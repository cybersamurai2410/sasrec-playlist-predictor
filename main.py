import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data_utils import parse_mpd_to_csv
from dataset import SASRecDataset
from model import SASRec

# Main entrypoint for running the full SASRec pipeline
if __name__ == "__main__":
    # Convert JSON to CSV if not already done
    if not os.path.exists("spotify_mpd_interactions.csv"):
        print("Generating CSV from mpd.slice.0-999.json...")
        parse_mpd_to_csv("mpd.slice.0-999.json", "spotify_mpd_interactions.csv")

    # Load dataset and count unique items
    interactions_file = "spotify_mpd_interactions.csv"
    df = pd.read_csv(interactions_file)
    num_items = df['track_id'].max()

    # Create PyTorch datasets and loaders
    dataset = SASRecDataset(interactions_file, max_seq_len=50)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    # Instantiate SASRec model
    model = SASRec(num_items=num_items, d_model=64, n_heads=2, num_layers=2, dropout=0.1, lr=1e-3, max_seq_len=50)

    # Train using PyTorch Lightning Trainer
    trainer = pl.Trainer(max_epochs=5, gpus=1 if torch.cuda.is_available() else 0)
    trainer.fit(model, train_loader, val_loader)

    # Run inference on one sample sequence
    sample_seq, _ = dataset[0]
    sample_seq = sample_seq.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        logits = model(sample_seq)
        next_item_preds = torch.topk(logits[0, -1], k=10).indices  # Top-10 predictions
    print("Top-10 recommendations:", next_item_preds.tolist())
