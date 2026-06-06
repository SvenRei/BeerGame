import torch

def get_vocab_tensor(vocab_size, device):
    """Unified vocabulary mapping for all agents."""
    if vocab_size == 1: 
        return torch.tensor([0.0], device=device)
    elif vocab_size == 3: 
        return torch.tensor([-1.0, 0.0, 1.0], device=device)
    elif vocab_size == 5: 
        return torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device)
    raise ValueError("Vocab size must be 1, 3, or 5")