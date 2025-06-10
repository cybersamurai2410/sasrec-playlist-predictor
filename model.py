import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# SASRec model definition using PyTorch Lightning
class SASRec(pl.LightningModule):
    def __init__(self, num_items, d_model=64, n_heads=2, num_layers=2, dropout=0.1, lr=1e-3, max_seq_len=50):
        super().__init__()
        self.save_hyperparameters()

        # Embedding layers for items and positions
        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Transformer encoder block (masked self-attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            dim_feedforward=d_model * 4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, num_items + 1)  # Predict over all items

        self.lr = lr
        self.max_seq_len = max_seq_len

    def forward(self, seq):
        batch_size, seq_len = seq.size()

        # Create position IDs
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)

        # Combine embeddings
        x = self.item_emb(seq) + self.pos_emb(positions)

        # [seq_len, batch, d_model] format for transformer
        x = self.dropout(x).permute(1, 0, 2)

        # Transformer encoder
        x = self.transformer(x)

        # Restore original shape
        x = x.permute(1, 0, 2)

        # Normalize and predict
        x = self.layer_norm(x)
        logits = self.output_proj(x)
        return logits

    def training_step(self, batch, batch_idx):
        seq, target = batch
        logits = self(seq)

        # Compute loss (ignore padding index)
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
      
