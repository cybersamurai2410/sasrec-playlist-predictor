import torch
import numpy as np
from torch.utils.data import DataLoader
import time

# Compute Recall@K for predictions vs true targets
def recall_at_k(preds, targets, k):
    correct = 0
    for pred, target in zip(preds, targets):
        if target in pred[:k]:
            correct += 1
    return correct / len(preds)

# Compute NDCG@K for predictions vs true targets
def ndcg_at_k(preds, targets, k):
    ndcg = 0
    for pred, target in zip(preds, targets):
        if target in pred[:k]:
            index = pred[:k].index(target)
            ndcg += 1 / np.log2(index + 2)
    return ndcg / len(preds)

# Evaluate model using Recall@K, NDCG@K, and average inference time per user
def evaluate_model(model, dataset, k=10):
    model.eval()
    preds = []
    targets = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    start_time = time.time()

    with torch.no_grad():
        for seq, target_seq in loader:
            logits = model(seq)
            last_logits = logits[:, -1, :]
            topk = torch.topk(last_logits, k=k).indices.squeeze(0).tolist()
            target_item = target_seq[0, -1].item()
            preds.append(topk)
            targets.append(target_item)

    recall = recall_at_k(preds, targets, k)
    ndcg = ndcg_at_k(preds, targets, k)
    inference_time = (time.time() - start_time) / len(dataset)

    print(f"Recall@{k}: {recall:.4f}")
    print(f"NDCG@{k}: {ndcg:.4f}")
    print(f"Avg Inference Time per user: {inference_time * 1000:.2f} ms")

    return recall, ndcg, inference_time
  
