import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import json
import os

# Utility function to convert Spotify MPD JSON slice to a simplified CSV format
# The resulting CSV will contain user_id, track_id, and timestamp for each track in a playlist
# This enables compatibility with the SASRec sequential recommendation format

def parse_mpd_to_csv(json_file, output_csv):
    with open(json_file, 'r') as f:
        mpd_data = json.load(f)

    rows = []
    for playlist in mpd_data['playlists']:
        pid = playlist['pid']  # Use playlist ID as user ID
        base_time = 1600000000  # Arbitrary base timestamp
        for i, track in enumerate(playlist['tracks']):
            track_uri = track['track_uri']
            # Convert track_uri string to integer ID for model compatibility
            track_id = abs(hash(track_uri)) % 100000
            timestamp = base_time + i * 60  # Increment timestamp per track
            rows.append([pid, track_id, timestamp])

    df = pd.DataFrame(rows, columns=["user_id", "track_id", "timestamp"])
    df.to_csv(output_csv, index=False)

# Dataset class used by PyTorch DataLoader for training the SASRec model
class SASRecDataset(Dataset):
    def __init__(self, interactions_file, max_seq_len=50):
        df = pd.read_csv(interactions_file)

        # Ensure expected columns are present
        expected_cols = {'user_id', 'track_id', 'timestamp'}
        if not expected_cols.issubset(set(df.columns)):
            raise ValueError(f"Dataset must contain columns: {expected_cols}, but got: {df.columns.tolist()}")

        # Sort data by user and timestamp to preserve interaction sequence
        df = df.sort_values(['user_id', 'timestamp'])
        self.user_sequences = df.groupby('user_id')['track_id'].apply(list).to_dict()
        self.max_seq_len = max_seq_len
        self.users = list(self.user_sequences.keys())

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        seq = self.user_sequences[user]

        # Truncate sequence if it's longer than max_seq_len
        if len(seq) >= self.max_seq_len:
            seq = seq[-self.max_seq_len:]

        # Create input and target sequences
        input_seq = seq[:-1]
        target = seq[1:]

        # Pad sequences with 0s if they're too short
        pad_len = self.max_seq_len - len(input_seq)
        if pad_len > 0:
            input_seq = [0] * pad_len + input_seq
            target = [0] * pad_len + target

        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

# SASRec: Self-Attentive Sequential Recommendation model using Transformer encoder layers
class SASRec(pl.LightningModule):
    def __init__(self, num_items, d_model=64, n_heads=2, num_layers=2, dropout=0.1, lr=1e-3, max_seq_len=50):
        super().__init__()
        self.save_hyperparameters()

        # Embedding layers for item IDs and positional encodings
        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Define the transformer encoder block with specified dimensions and heads
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            dim_feedforward=d_model * 4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Regularization and projection
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, num_items + 1)
        self.lr = lr
        self.max_seq_len = max_seq_len

    def forward(self, seq):
        batch_size, seq_len = seq.size()

        # Create positional indices and expand to match batch size
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)

        # Combine item and position embeddings
        x = self.item_emb(seq) + self.pos_emb(positions)

        # Reshape input for transformer: [seq_len, batch_size, embedding_dim]
        x = self.dropout(x).permute(1, 0, 2)

        # Apply transformer encoder
        x = self.transformer(x)

        # Restore original shape: [batch_size, seq_len, embedding_dim]
        x = x.permute(1, 0, 2)

        # Normalize and project to vocabulary size
        x = self.layer_norm(x)
        logits = self.output_proj(x)
        return logits

    def training_step(self, batch, batch_idx):
        seq, target = batch
        logits = self(seq)

        # Compute cross-entropy loss, ignoring padded values (0)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=0)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        seq, target = batch
        logits = self(seq)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=0)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

if __name__ == "__main__":
    # Automatically generate CSV if it doesn't exist
    if not os.path.exists("spotify_mpd_interactions.csv"):
        print("Generating CSV from mpd.slice.0-999.json...")
        parse_mpd_to_csv("mpd.slice.0-999.json", "spotify_mpd_interactions.csv")

    # Load dataset
    interactions_file = "spotify_mpd_interactions.csv"
    df = pd.read_csv(interactions_file)
    num_items = df['track_id'].max()

    # Prepare data loaders
    dataset = SASRecDataset(interactions_file, max_seq_len=50)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    # Initialize model
    model = SASRec(num_items=num_items, d_model=64, n_heads=2, num_layers=2, dropout=0.1, lr=1e-3, max_seq_len=50)

    # Train model using PyTorch Lightning
    trainer = pl.Trainer(max_epochs=5, gpus=1 if torch.cuda.is_available() else 0)
    trainer.fit(model, train_loader, val_loader)

    # Run inference on first user sequence in dataset
    sample_seq, _ = dataset[0]
    sample_seq = sample_seq.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        logits = model(sample_seq)
        next_item_preds = torch.topk(logits[0, -1], k=10).indices  # Get top-10 predictions
    print("Top-10 recommendations:", next_item_preds.tolist())
