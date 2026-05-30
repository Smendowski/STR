import torch


def attention_score(q: torch.Tensor, k: torch.Tensor) -> float:
    q_norm = q / q.norm(p=2)
    k_norm = k / k.norm(p=2)
    return torch.dot(q_norm, k_norm).item()
