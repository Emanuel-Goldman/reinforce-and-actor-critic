"""Basic REINFORCE algorithm (Monte-Carlo Policy Gradient)"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from src.pg.networks import PolicyNetwork
from src.pg.distributions import sample_action, get_log_prob, get_entropy
from src.pg.buffers import TrajectoryBuffer
from src.common.utils import compute_returns


class REINFORCE:
    """Basic REINFORCE algorithm."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list = [128, 128],
        lr: float = 3e-4,
        gamma: float = 0.99,
        normalize_returns: bool = False,
        entropy_coef: float = 0.0,
        device: str = "cpu"
    ):
        """
        Initialize REINFORCE agent.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_sizes: Hidden layer sizes for policy network
            lr: Learning rate for policy
            gamma: Discount factor
            normalize_returns: Whether to normalize returns
            entropy_coef: Entropy bonus coefficient (for exploration)
            device: Device to run on ("cpu" or "cuda")
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.normalize_returns = normalize_returns
        self.entropy_coef = entropy_coef
        self.device = device
        
        # Policy network
        self.policy = PolicyNetwork(obs_dim, action_dim, hidden_sizes).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        # Buffer for current episode
        self.buffer = TrajectoryBuffer()
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """
        Select an action given an observation.
        
        Args:
            obs: Observation array
            deterministic: If True, select greedy action (not used in training)
        
        Returns:
            Action index
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.policy(obs_tensor)
            if deterministic:
                action = torch.argmax(logits, dim=-1).item()
                log_prob = get_log_prob(logits, torch.tensor([action]).to(self.device))
            else:
                action, log_prob = sample_action(logits)
                action = action.item()
                log_prob = log_prob.item()
        
        return action
    
    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        done: bool = False
    ) -> None:
        """
        Store a transition in the buffer.
        
        Args:
            obs: Observation
            action: Action taken
            reward: Reward received
            log_prob: Log probability of action
            done: Whether episode terminated
        """
        self.buffer.add(obs, action, reward, log_prob, done=done)
    
    def update(self) -> dict:
        """
        Update policy using REINFORCE algorithm.
        
        Returns:
            Dictionary with training metrics
        """
        if len(self.buffer.rewards) == 0:
            return {}
        
        # Compute returns
        returns = compute_returns(
            self.buffer.rewards,
            self.gamma,
            normalize=self.normalize_returns
        ).to(self.device)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        log_probs_old = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        
        # Get current log probabilities
        logits = self.policy(states)
        log_probs = get_log_prob(logits, actions)
        
        # Policy gradient loss: -log_prob * return
        policy_loss = -(log_probs * returns).mean()
        
        # Entropy bonus (optional)
        entropy = get_entropy(logits).mean()
        entropy_bonus = self.entropy_coef * entropy
        
        # Total loss
        loss = policy_loss - entropy_bonus
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        # Clear buffer
        self.buffer.clear()
        
        return {
            "policy_loss": policy_loss.item(),
            "entropy": entropy.item(),
            "mean_return": returns.mean().item()
        }
    
    def save(self, filepath: str) -> None:
        """Save policy network weights."""
        torch.save(self.policy.state_dict(), filepath)
    
    def load(self, filepath: str) -> None:
        """Load policy network weights."""
        self.policy.load_state_dict(torch.load(filepath, map_location=self.device))

