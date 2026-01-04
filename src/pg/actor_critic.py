"""Actor-Critic algorithm (TD-based policy gradient)"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from src.pg.networks import PolicyNetwork, ValueNetwork
from src.pg.distributions import sample_action, get_log_prob, get_entropy
from src.pg.buffers import TrajectoryBuffer


class ActorCritic:
    """Actor-Critic algorithm with TD-error as advantage."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list = [128, 128],
        lr_policy: float = 3e-4,
        lr_value: float = 1e-3,
        gamma: float = 0.99,
        normalize_advantages: bool = False,
        entropy_coef: float = 0.01,
        device: str = "cpu"
    ):
        """
        Initialize Actor-Critic agent.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_sizes: Hidden layer sizes for networks
            lr_policy: Learning rate for policy
            lr_value: Learning rate for value function
            gamma: Discount factor
            normalize_advantages: Whether to normalize TD-errors
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
        
        # Buffer for online updates (can update after each step or batch)
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
        next_obs: Optional[np.ndarray] = None,
        next_value: Optional[float] = None,
        done: bool = False
    ) -> None:
        """
        Store a transition in the buffer.
        
        For episode-based updates, we store the current state and value.
        The next state and value will be stored when the next transition is added,
        or we can append them at the end of the episode.
        """
        self.buffer.add(obs, action, reward, log_prob, value=value, done=done)
        # For episode updates, we'll append next_obs and next_value at the end
        # This allows update_episode to compute TD-errors correctly
    
    def update_step(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        next_obs: np.ndarray,
        next_value: float,
        done: bool
    ) -> dict:
        """
        Update networks using TD-error from a single step.
        
        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            log_prob: Log probability of action
            value: Value estimate of current state
            next_obs: Next observation
            next_value: Value estimate of next state
            done: Whether episode terminated
        
        Returns:
            Dictionary with training metrics
        """
        # Convert to tensors
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_tensor = torch.BoolTensor([done]).to(self.device)
        
        # Compute TD-error: δ_t = r_t + γ * (1 - done) * V(s_{t+1}) - V(s_t)
        with torch.no_grad():
            next_value_tensor = torch.FloatTensor([next_value]).to(self.device)
            target = reward_tensor + self.gamma * (1 - done_tensor.float()) * next_value_tensor
        
        # Current value estimate
        value_tensor = self.value(obs_tensor)
        
        # TD-error (advantage)
        td_error = target - value_tensor
        
        # Normalize if requested (optional, usually not done for TD-error)
        if self.normalize_advantages:
            # This would require batch statistics, skip for single-step updates
            pass
        
        # Policy update: maximize log_prob * td_error
        logits = self.policy(obs_tensor)
        log_prob = get_log_prob(logits, action_tensor)
        policy_loss = -(log_prob * td_error.detach()).mean()
        
        # Entropy bonus
        entropy = get_entropy(logits).mean()
        entropy_bonus = self.entropy_coef * entropy
        
        policy_loss_total = policy_loss - entropy_bonus
        
        # Value update: minimize (td_error)^2 or MSE(V(s_t), target)
        value_loss = td_error.pow(2).mean()
        # Alternative: value_loss = nn.MSELoss()(value_tensor, target)
        
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
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "td_error": td_error.item(),
            "entropy": entropy.item()
        }
    
    def update_episode(self) -> dict:
        """
        Update networks using all transitions from the episode.
        Computes TD-errors for each step.
        
        Returns:
            Dictionary with training metrics
        """
        if len(self.buffer.rewards) == 0:
            return {}
        
        # For TD-error computation, we need current states and next states
        # states[i] is the state at step i, states[i+1] is the next state
        # So we use states[:-1] for current and states[1:] for next
        states = torch.FloatTensor(np.array(self.buffer.states[:-1])).to(self.device)  # Exclude last state
        next_states = torch.FloatTensor(np.array(self.buffer.states[1:])).to(self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        rewards = torch.FloatTensor(self.buffer.rewards).to(self.device)
        dones = torch.BoolTensor(self.buffer.dones).to(self.device)
        
        # Get value estimates for current and next states
        with torch.no_grad():
            current_values = self.value(states)
            next_values = self.value(next_states)
        log_probs_old = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        
        # Compute TD-errors: δ_t = r_t + γ * (1 - done) * V(s_{t+1}) - V(s_t)
        targets = rewards + self.gamma * (1 - dones.float()) * next_values
        td_errors = targets - current_values
        
        if self.normalize_advantages:
            td_errors = (td_errors - td_errors.mean()) / (td_errors.std() + 1e-8)
        
        # Policy update
        logits = self.policy(states)
        log_probs = get_log_prob(logits, actions)
        policy_loss = -(log_probs * td_errors.detach()).mean()
        
        # Entropy bonus
        entropy = get_entropy(logits).mean()
        entropy_bonus = self.entropy_coef * entropy
        policy_loss_total = policy_loss - entropy_bonus
        
        # Value update
        value_loss = td_errors.pow(2).mean()
        
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
            "mean_td_error": td_errors.mean().item(),
            "std_td_error": td_errors.std().item(),
            "entropy": entropy.item()
        }
    
    def save(self, filepath_prefix: str) -> None:
        """Save both networks."""
        torch.save(self.policy.state_dict(), f"{filepath_prefix}_policy.pt")
        torch.save(self.value.state_dict(), f"{filepath_prefix}_value.pt")
    
    def load(self, filepath_prefix: str) -> None:
        """Load both networks."""
        self.policy.load_state_dict(torch.load(f"{filepath_prefix}_policy.pt", map_location=self.device))
        self.value.load_state_dict(torch.load(f"{filepath_prefix}_value.pt", map_location=self.device))

