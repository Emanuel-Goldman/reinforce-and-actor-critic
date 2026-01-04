"""Trajectory buffers for storing episode data"""

from typing import List
import torch
import numpy as np


class TrajectoryBuffer:
    """Buffer for storing a single episode trajectory."""
    
    def __init__(self):
        """Initialize empty buffer."""
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []  # For baseline/AC
        self.dones: List[bool] = []
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float = 0.0,
        done: bool = False
    ) -> None:
        """
        Add a step to the buffer.
        
        Args:
            state: Observation
            action: Action taken
            reward: Reward received
            log_prob: Log probability of action
            value: Value estimate (optional)
            done: Whether episode terminated
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
    
    def get_episode_return(self) -> float:
        """Compute total undiscounted return for the episode."""
        return sum(self.rewards)
    
    def get_episode_length(self) -> int:
        """Get episode length."""
        return len(self.rewards)

