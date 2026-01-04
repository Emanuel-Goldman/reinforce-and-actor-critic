"""Action distribution utilities for discrete actions"""

import torch
from torch.distributions import Categorical
from typing import Tuple


def sample_action(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample an action from a categorical distribution.
    
    Args:
        logits: Action logits (batch_size, action_dim) or (action_dim,)
    
    Returns:
        Tuple of (action, log_prob)
    """
    dist = Categorical(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action, log_prob


def get_log_prob(logits: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    Get log probability of an action given logits.
    
    Args:
        logits: Action logits (batch_size, action_dim)
        action: Action indices (batch_size,)
    
    Returns:
        Log probabilities (batch_size,)
    """
    dist = Categorical(logits=logits)
    return dist.log_prob(action)


def get_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of the action distribution.
    
    Args:
        logits: Action logits (batch_size, action_dim) or (action_dim,)
    
    Returns:
        Entropy scalar or tensor
    """
    dist = Categorical(logits=logits)
    return dist.entropy()

