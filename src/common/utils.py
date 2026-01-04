"""General utility functions"""

from typing import List, Tuple
import numpy as np
import torch


def parse_hidden_sizes(hidden_sizes_str: str) -> List[int]:
    """
    Parse comma-separated hidden sizes string.
    
    Args:
        hidden_sizes_str: Comma-separated string like "128,128"
    
    Returns:
        List of integers
    """
    return [int(s.strip()) for s in hidden_sizes_str.split(",") if s.strip()]


def normalize_returns(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize returns to have zero mean and unit variance.
    
    Args:
        returns: Tensor of returns
        eps: Small value to avoid division by zero
    
    Returns:
        Normalized returns
    """
    mean = returns.mean()
    std = returns.std()
    return (returns - mean) / (std + eps)


def compute_returns(
    rewards: List[float],
    gamma: float,
    normalize: bool = False
) -> torch.Tensor:
    """
    Compute discounted returns from rewards.
    
    Args:
        rewards: List of rewards for an episode
        gamma: Discount factor
        normalize: Whether to normalize returns
    
    Returns:
        Tensor of returns G_t for each time step
    """
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    
    returns_tensor = torch.tensor(returns, dtype=torch.float32)
    
    if normalize:
        returns_tensor = normalize_returns(returns_tensor)
    
    return returns_tensor


def get_space_size(space) -> Tuple[int, int]:
    """
    Get observation and action space sizes.
    
    Args:
        space: Gymnasium space (observation or action)
    
    Returns:
        Tuple of (size, is_discrete)
    """
    if hasattr(space, 'shape'):
        if len(space.shape) == 0:
            return (1, False)
        return (int(np.prod(space.shape)), False)
    elif hasattr(space, 'n'):
        return (space.n, True)
    else:
        raise ValueError(f"Unknown space type: {type(space)}")

