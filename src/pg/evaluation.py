"""Evaluation utilities for trained policies"""

import torch
import numpy as np
from typing import List, Optional
import gymnasium as gym

from src.pg.networks import PolicyNetwork
from src.pg.distributions import get_log_prob


def evaluate_policy(
    policy: PolicyNetwork,
    env: gym.Env,
    n_episodes: int = 10,
    deterministic: bool = True,
    device: str = "cpu",
    render: bool = False
) -> dict:
    """
    Evaluate a policy over multiple episodes.
    
    Args:
        policy: Policy network
        env: Environment
        n_episodes: Number of episodes to run
        deterministic: Whether to use deterministic (greedy) actions
        device: Device to run on
        render: Whether to render episodes
    
    Returns:
        Dictionary with evaluation metrics
    """
    episode_returns = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_return = 0.0
        episode_length = 0
        done = False
        
        while not done:
            if render:
                env.render()
            
            # Select action
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = policy(obs_tensor)
                if deterministic:
                    action = torch.argmax(logits, dim=-1).item()
                else:
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample().item()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_return += reward
            episode_length += 1
        
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
    
    return {
        "mean_return": np.mean(episode_returns),
        "std_return": np.std(episode_returns),
        "min_return": np.min(episode_returns),
        "max_return": np.max(episode_returns),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths
    }

