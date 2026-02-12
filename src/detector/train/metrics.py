import torch

@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim = 1)
    return (pred == y).float().mean().item()
    # .item() converts single-value tensor to Python float, eg. tensor(0.6667) -> 0.6667  