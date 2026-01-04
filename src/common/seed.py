"""Deterministic seeding utilities"""

import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms in PyTorch (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Note: Some operations may still be non-deterministic on GPU
        # For full reproducibility, run on CPU or use torch.use_deterministic_algorithms(True)

