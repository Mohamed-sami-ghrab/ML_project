import torch

def calculate_f1(preds, targets, threshold=0.5):
    probs = torch.sigmoid(preds)
    pred_mask = (probs > threshold).float()
    
    tp = (pred_mask * targets).sum()
    fp = (pred_mask * (1 - targets)).sum()
    fn = ((1 - pred_mask) * targets).sum()
    
    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    return f1