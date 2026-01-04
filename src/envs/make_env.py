"""Environment creation utilities"""

import gymnasium as gym
from typing import Optional, Tuple
import numpy as np


def make_env(
    env_name: str,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None
) -> gym.Env:
    """
    Create and configure a Gymnasium environment.
    
    Args:
        env_name: Name of the environment (e.g., "CartPole-v1")
        seed: Random seed for environment
        render_mode: Rendering mode ("human", "rgb_array", or None)
    
    Returns:
        Configured environment
    """
    env = gym.make(env_name, render_mode=render_mode)
    
    if seed is not None:
        env.reset(seed=seed)
    
    return env


def get_env_info(env: gym.Env) -> dict:
    """
    Extract environment information.
    
    Args:
        env: Gymnasium environment
    
    Returns:
        Dictionary with obs_dim, action_dim, is_discrete, max_episode_steps
    """
    obs_space = env.observation_space
    action_space = env.action_space
    
    # Get observation dimension
    if hasattr(obs_space, 'shape'):
        obs_dim = int(np.prod(obs_space.shape))
    else:
        obs_dim = obs_space.n
    
    # Get action dimension and type
    if hasattr(action_space, 'n'):
        action_dim = action_space.n
        is_discrete = True
    else:
        action_dim = int(np.prod(action_space.shape))
        is_discrete = False
    
    # Get max episode steps (if available)
    max_episode_steps = None
    if hasattr(env, '_max_episode_steps'):
        max_episode_steps = env._max_episode_steps
    elif hasattr(env, 'spec') and env.spec is not None:
        max_episode_steps = env.spec.max_episode_steps
    
    return {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "is_discrete": is_discrete,
        "max_episode_steps": max_episode_steps
    }

