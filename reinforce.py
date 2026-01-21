#!/usr/bin/env python3
"""Training script for basic REINFORCE algorithm (Monte-Carlo Policy Gradient)"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import matplotlib.pyplot as plt


# ============================================================================
# Utility Functions
# ============================================================================

def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_env(env_name: str, seed: Optional[int] = None, render_mode: Optional[str] = None) -> gym.Env:
    """Create and configure a Gymnasium environment."""
    env = gym.make(env_name, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
    return env


def get_env_info(env: gym.Env) -> dict:
    """Extract environment information."""
    obs_space = env.observation_space
    action_space = env.action_space
    
    if hasattr(obs_space, 'shape'):
        obs_dim = int(np.prod(obs_space.shape))
    else:
        obs_dim = obs_space.n
    
    if hasattr(action_space, 'n'):
        action_dim = action_space.n
        is_discrete = True
    else:
        action_dim = int(np.prod(action_space.shape))
        is_discrete = False
    
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


def parse_hidden_sizes(hidden_sizes_str: str) -> List[int]:
    """Parse comma-separated hidden sizes string."""
    return [int(s.strip()) for s in hidden_sizes_str.split(",") if s.strip()]


def normalize_returns(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize returns to have zero mean and unit variance."""
    mean = returns.mean()
    std = returns.std()
    return (returns - mean) / (std + eps)


def compute_returns(rewards: List[float], gamma: float, normalize: bool = False) -> torch.Tensor:
    """Compute discounted returns from rewards."""
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    
    returns_tensor = torch.tensor(returns, dtype=torch.float32)
    if normalize:
        returns_tensor = normalize_returns(returns_tensor)
    
    return returns_tensor


# ============================================================================
# Neural Networks
# ============================================================================

class MLP(nn.Module):
    """Multi-layer perceptron with configurable hidden layers."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: List[int], 
                 activation: str = "relu", output_activation: Optional[str] = None):
        super().__init__()
        layers = []
        prev_size = input_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "elu":
                layers.append(nn.ELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_dim))
        if output_activation == "tanh":
            layers.append(nn.Tanh())
        elif output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif output_activation is not None:
            raise ValueError(f"Unknown output activation: {output_activation}")
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyNetwork(nn.Module):
    """Policy network that outputs action logits for discrete actions."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: List[int] = [128, 128]):
        super().__init__()
        self.mlp = MLP(
            input_dim=obs_dim,
            output_dim=action_dim,
            hidden_sizes=hidden_sizes,
            activation="relu",
            output_activation=None
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


# ============================================================================
# Distribution Utilities
# ============================================================================

def sample_action(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample an action from a categorical distribution."""
    dist = Categorical(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action, log_prob


def get_log_prob(logits: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Get log probability of an action given logits."""
    dist = Categorical(logits=logits)
    return dist.log_prob(action)


def get_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy of the action distribution."""
    dist = Categorical(logits=logits)
    return dist.entropy()


# ============================================================================
# Buffer
# ============================================================================

class TrajectoryBuffer:
    """Buffer for storing a single episode trajectory."""
    
    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []
    
    def add(self, state: np.ndarray, action: int, reward: float, log_prob: float, done: bool = False) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.dones.clear()


# ============================================================================
# REINFORCE Algorithm
# ============================================================================

class REINFORCE:
    """Basic REINFORCE algorithm."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: list = [128, 128],
                 lr: float = 3e-4, gamma: float = 0.99, normalize_returns: bool = False,
                 entropy_coef: float = 0.0, device: str = "cpu"):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.normalize_returns = normalize_returns
        self.entropy_coef = entropy_coef
        self.device = device
        
        self.policy = PolicyNetwork(obs_dim, action_dim, hidden_sizes).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = TrajectoryBuffer()
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """Select an action given an observation."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.policy(obs_tensor)
            if deterministic:
                action = torch.argmax(logits, dim=-1).item()
            else:
                action, _ = sample_action(logits)
                action = action.item()
        return action
    
    def store_transition(self, obs: np.ndarray, action: int, reward: float, 
                        log_prob: float, done: bool = False) -> None:
        """Store a transition in the buffer."""
        self.buffer.add(obs, action, reward, log_prob, done=done)
    
    def update(self) -> dict:
        """Update policy using REINFORCE algorithm."""
        if len(self.buffer.rewards) == 0:
            return {}
        
        returns = compute_returns(self.buffer.rewards, self.gamma, normalize=self.normalize_returns).to(self.device)
        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        
        logits = self.policy(states)
        log_probs = get_log_prob(logits, actions)
        policy_loss = -(log_probs * returns).mean()
        
        entropy = get_entropy(logits).mean()
        entropy_bonus = self.entropy_coef * entropy
        loss = policy_loss - entropy_bonus
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()
        
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


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_policy(policy: PolicyNetwork, env: gym.Env, n_episodes: int = 10,
                   deterministic: bool = True, device: str = "cpu", render: bool = False) -> dict:
    """Evaluate a policy over multiple episodes."""
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
            
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = policy(obs_tensor)
                if deterministic:
                    action = torch.argmax(logits, dim=-1).item()
                else:
                    dist = Categorical(logits=logits)
                    action = dist.sample().item()
            
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


# ============================================================================
# Logging and Plotting
# ============================================================================

class MetricsLogger:
    """Logs metrics to TensorBoard and saves to JSON."""
    
    def __init__(self, log_dir: str, artifact_dir: str, run_name: str = "run"):
        self.log_dir = Path(log_dir)
        self.artifact_dir = Path(artifact_dir)
        self.run_name = run_name
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        (self.artifact_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.artifact_dir / "plots").mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.metrics: Dict[str, list] = {}
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value to TensorBoard and store in memory."""
        self.writer.add_scalar(tag, value, step)
        if tag not in self.metrics:
            self.metrics[tag] = []
        self.metrics[tag].append({"step": step, "value": value})
    
    def save_metrics(self, filename: str = "metrics.json") -> None:
        """Save all logged metrics to JSON file."""
        metrics_file = self.artifact_dir / filename
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to {metrics_file}")
    
    def close(self) -> None:
        """Close the TensorBoard writer."""
        self.writer.close()
    
    def get_metrics(self) -> Dict[str, list]:
        """Get all logged metrics."""
        return self.metrics.copy()


def moving_average(data: np.ndarray, window: int = 100) -> np.ndarray:
    """Compute moving average of data."""
    if len(data) < window:
        return np.convolve(data, np.ones(window) / window, mode='valid')
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_metrics(metrics: Dict[str, List[Dict[str, float]]], output_dir: Path, window: int = 100) -> None:
    """Generate plots from metrics dictionary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if "episode_return" in metrics:
        returns = [m["value"] for m in metrics["episode_return"]]
        steps = [m["step"] for m in metrics["episode_return"]]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, returns, alpha=0.3, label="Episode Return", color="blue")
        
        if len(returns) >= window:
            ma_returns = moving_average(np.array(returns), window)
            ma_steps = steps[window-1:]
            plt.plot(ma_steps, ma_returns, label=f"{window}-Episode Moving Average", 
                    color="red", linewidth=2)
        
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("Episode Returns Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "episode_returns.png", dpi=150)
        plt.close()
    
    if "policy_loss" in metrics:
        losses = [m["value"] for m in metrics["policy_loss"]]
        steps = [m["step"] for m in metrics["policy_loss"]]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, alpha=0.7, label="Policy Loss", color="green")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title("Policy Loss Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "policy_loss.png", dpi=150)
        plt.close()
    
    if "entropy" in metrics:
        entropies = [m["value"] for m in metrics["entropy"]]
        steps = [m["step"] for m in metrics["entropy"]]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, entropies, alpha=0.7, label="Policy Entropy", color="brown")
        plt.xlabel("Episode")
        plt.ylabel("Entropy")
        plt.title("Policy Entropy Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "entropy.png", dpi=150)
        plt.close()
    
    print(f"Plots saved to {output_dir}")


# ============================================================================
# Main Training Script
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train REINFORCE agent")
    
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr_policy", type=float, default=3e-4, help="Policy learning rate")
    parser.add_argument("--hidden_sizes", type=str, default="128,128", help="Hidden layer sizes (comma-separated)")
    parser.add_argument("--normalize_returns", action="store_true", help="Normalize returns")
    parser.add_argument("--entropy_coef", type=float, default=0.0, help="Entropy bonus coefficient")
    parser.add_argument("--max_episodes", type=int, default=1000, help="Maximum training episodes")
    parser.add_argument("--eval_interval", type=int, default=100, help="Evaluation interval (episodes)")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Number of episodes for evaluation")
    parser.add_argument("--log_dir", type=str, default=None, help="TensorBoard log directory")
    parser.add_argument("--artifact_dir", type=str, default=None, help="Artifact output directory")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    set_seed(args.seed)
    
    env = make_env(args.env, seed=args.seed)
    env_info = get_env_info(env)
    
    print(f"Environment: {args.env}")
    print(f"Observation dim: {env_info['obs_dim']}")
    print(f"Action dim: {env_info['action_dim']}")
    print(f"Discrete actions: {env_info['is_discrete']}")
    
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    
    agent = REINFORCE(
        obs_dim=env_info['obs_dim'],
        action_dim=env_info['action_dim'],
        hidden_sizes=hidden_sizes,
        lr=args.lr_policy,
        gamma=args.gamma,
        normalize_returns=args.normalize_returns,
        entropy_coef=args.entropy_coef,
        device="cpu"
    )
    
    if args.log_dir is None:
        args.log_dir = f"results/reinforce/tensorboard/seed_{args.seed}"
    if args.artifact_dir is None:
        args.artifact_dir = f"results/reinforce"
    
    logger = MetricsLogger(
        log_dir=args.log_dir,
        artifact_dir=args.artifact_dir,
        run_name=f"reinforce_{args.env}_{args.seed}"
    )
    
    best_mean_return = float('-inf')
    episodes_to_threshold = None
    threshold = 475.0
    
    print("\nStarting training...")
    print(f"Max episodes: {args.max_episodes}")
    print(f"Gamma: {args.gamma}, LR: {args.lr_policy}")
    print("-" * 50)
    
    for episode in range(1, args.max_episodes + 1):
        obs, info = env.reset()
        episode_return = 0.0
        episode_length = 0
        done = False
        
        while not done and episode_length < args.max_steps:
            action = agent.select_action(obs, deterministic=False)
            logits = agent.policy(torch.FloatTensor(obs).unsqueeze(0))
            log_prob = get_log_prob(logits, torch.LongTensor([action])).item()
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(obs, action, reward, log_prob, done=done)
            
            obs = next_obs
            episode_return += reward
            episode_length += 1
        
        metrics = agent.update()
        
        logger.log_scalar("episode_return", episode_return, episode)
        logger.log_scalar("episode_length", episode_length, episode)
        if metrics:
            logger.log_scalar("policy_loss", metrics.get("policy_loss", 0), episode)
            if "entropy" in metrics:
                logger.log_scalar("entropy", metrics["entropy"], episode)
        
        if episode % args.eval_interval == 0 or episode == args.max_episodes:
            eval_results = evaluate_policy(
                agent.policy,
                env,
                n_episodes=args.eval_episodes,
                deterministic=True,
                device="cpu",
                render=False
            )
            mean_return = eval_results["mean_return"]
            logger.log_scalar("eval_mean_return", mean_return, episode)
            
            print(f"Episode {episode:4d} | "
                  f"Return: {episode_return:7.2f} | "
                  f"Eval Return: {mean_return:7.2f} ± {eval_results['std_return']:.2f} | "
                  f"Loss: {metrics.get('policy_loss', 0):.4f}")
            
            if mean_return > best_mean_return:
                best_mean_return = mean_return
                agent.save(f"{args.artifact_dir}/checkpoints/policy_best.pt")
            
            if episodes_to_threshold is None and mean_return >= threshold:
                episodes_to_threshold = episode
                print(f"  ✓ Reached threshold {threshold} at episode {episode}")
    
    agent.save(f"{args.artifact_dir}/checkpoints/policy_final.pt")
    
    logger.save_metrics()
    plot_metrics(logger.get_metrics(), Path(args.artifact_dir) / "plots")
    logger.close()
    
    print("\n" + "=" * 50)
    print("Training Summary")
    print("=" * 50)
    print(f"Best eval return: {best_mean_return:.2f}")
    if episodes_to_threshold:
        print(f"Episodes to reach {threshold}: {episodes_to_threshold}")
    else:
        print(f"Did not reach threshold {threshold}")
    print(f"Metrics saved to: {args.artifact_dir}/metrics.json")
    print(f"Plots saved to: {args.artifact_dir}/plots/")
    print(f"TensorBoard logs: {args.log_dir}")
    print("=" * 50)
    
    env.close()


if __name__ == "__main__":
    main()
