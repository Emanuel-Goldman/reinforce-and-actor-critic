"""REINFORCE with baseline (variance reduction)"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from src.pg.networks import PolicyNetwork, ValueNetwork
from src.pg.distributions import sample_action, get_log_prob, get_entropy
from src.pg.buffers import TrajectoryBuffer
from src.common.utils import compute_returns, normalize_returns


class REINFORCEBaseline:
    """REINFORCE with value function baseline."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list = [128, 128],
        lr_policy: float = 3e-4,
        lr_value: float = 1e-3,
        gamma: float = 0.99,
        normalize_advantages: bool = False,
        entropy_coef: float = 0.0,
        device: str = "cpu"
    ):
        """
        Initialize REINFORCE with baseline agent.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_sizes: Hidden layer sizes for networks
            lr_policy: Learning rate for policy
            lr_value: Learning rate for value function
            gamma: Discount factor
            normalize_advantages: Whether to normalize advantages
            entropy_coef: Entropy bonus coefficient
            device: Device to run on
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.normalize_advantages = normalize_advantages
        self.entropy_coef = entropy_coef
        self.device = device
        
        # Networks
        self.policy = PolicyNetwork(obs_dim, action_dim, hidden_sizes).to(device)
        self.value = ValueNetwork(obs_dim, hidden_sizes).to(device)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr_value)
        
        # Buffer
        self.buffer = TrajectoryBuffer()
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> tuple:
        """
        Select an action and estimate value.
        
        Args:
            obs: Observation array
            deterministic: If True, select greedy action
        
        Returns:
            Tuple of (action, value_estimate)
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.policy(obs_tensor)
            value = self.value(obs_tensor)
            
            if deterministic:
                action = torch.argmax(logits, dim=-1).item()
                log_prob = get_log_prob(logits, torch.tensor([action]).to(self.device))
            else:
                action, log_prob = sample_action(logits)
                action = action.item()
                log_prob = log_prob.item()
            
            value = value.item()
        
        return action, value
    
    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool = False
    ) -> None:
        """Store a transition in the buffer."""
        self.buffer.add(obs, action, reward, log_prob, value=value, done=done)
    
    def update(self) -> dict:
        """
        Update policy and value networks.
        
        Returns:
            Dictionary with training metrics
        """
        if len(self.buffer.rewards) == 0:
            return {}
        
        # Compute returns
        returns = compute_returns(
            self.buffer.rewards,
            self.gamma,
            normalize=False  # Don't normalize returns, normalize advantages instead
        ).to(self.device)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        values_old = torch.FloatTensor(self.buffer.values).to(self.device)
        log_probs_old = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        
        # Compute advantages: A_t = G_t - V(s_t)
        # Detach value estimates to prevent gradients flowing through baseline
        advantages = returns - values_old.detach()
        
        if self.normalize_advantages:
            advantages = normalize_returns(advantages)
        
        # Policy update: maximize log_prob * advantage
        logits = self.policy(states)
        log_probs = get_log_prob(logits, actions)
        policy_loss = -(log_probs * advantages).mean()
        
        # Entropy bonus
        entropy = get_entropy(logits).mean()
        entropy_bonus = self.entropy_coef * entropy
        
        policy_loss_total = policy_loss - entropy_bonus
        
        # Value update: minimize MSE(V(s_t), G_t)
        values = self.value(states)
        value_loss = nn.MSELoss()(values, returns)
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.policy_optimizer.step()
        
        # Update value
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=0.5)
        self.value_optimizer.step()
        
        # Clear buffer
        self.buffer.clear()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "mean_return": returns.mean().item(),
            "mean_advantage": advantages.mean().item(),
            "mean_value": values.mean().item()
        }
    
    def save(self, filepath_prefix: str) -> None:
        """Save both networks."""
        torch.save(self.policy.state_dict(), f"{filepath_prefix}_policy.pt")
        torch.save(self.value.state_dict(), f"{filepath_prefix}_value.pt")
    
    def load(self, filepath_prefix: str) -> None:
        """Load both networks."""
        self.policy.load_state_dict(torch.load(f"{filepath_prefix}_policy.pt", map_location=self.device))
        self.value.load_state_dict(torch.load(f"{filepath_prefix}_value.pt", map_location=self.device))

