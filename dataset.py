import torch
from torch.utils.data import Dataset
import pandas as pd

# Custom dataset for SASRec to format interaction sequences for training
class SASRecDataset(Dataset):
    def __init__(self, interactions_file, max_seq_len=50):
        df = pd.read_csv(interactions_file)

        # Validate that all required columns exist
        expected_cols = {'user_id', 'track_id', 'timestamp'}
        if not expected_cols.issubset(set(df.columns)):
            raise ValueError(f"Dataset must contain columns: {expected_cols}, but got: {df.columns.tolist()}")

        # Sort interactions by time
        df = df.sort_values(['user_id', 'timestamp'])

        # Group sequences by user
        self.user_sequences = df.groupby('user_id')['track_id'].apply(list).to_dict()
        self.max_seq_len = max_seq_len
        self.users = list(self.user_sequences.keys())

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        seq = self.user_sequences[user]

        # Clip sequence to max length
        if len(seq) >= self.max_seq_len:
            seq = seq[-self.max_seq_len:]

        # Create input and target sequences by shifting
        input_seq = seq[:-1]
        target = seq[1:]

        # Left-pad with 0s if shorter than max_seq_len
        pad_len = self.max_seq_len - len(input_seq)
        if pad_len > 0:
            input_seq = [0] * pad_len + input_seq
            target = [0] * pad_len + target

        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)
