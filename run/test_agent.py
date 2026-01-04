#!/usr/bin/env python3
"""Test a trained agent"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import gymnasium as gym

from src.common.seed import set_seed
from src.envs.make_env import make_env, get_env_info
from src.pg.networks import PolicyNetwork, ValueNetwork
from src.pg.evaluation import evaluate_policy


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test a trained agent")
    
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--n_episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--render", action="store_true", help="Render episodes")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic policy")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create environment
    env = make_env(args.env, seed=args.seed, render_mode="human" if args.render else None)
    env_info = get_env_info(env)
    
    print(f"Environment: {args.env}")
    print(f"Observation dim: {env_info['obs_dim']}")
    print(f"Action dim: {env_info['action_dim']}")
    print(f"Loading checkpoint: {args.checkpoint}")
    
    # Load policy network
    policy = PolicyNetwork(
        obs_dim=env_info['obs_dim'],
        action_dim=env_info['action_dim'],
        hidden_sizes=[128, 128]  # TODO: Match training config
    )
    
    # Check if it's a baseline/AC checkpoint (has _policy suffix) or basic REINFORCE
    checkpoint_path = Path(args.checkpoint)
    if "_policy" in checkpoint_path.name or "_value" in checkpoint_path.name:
        # Baseline or AC checkpoint
        if "_policy" in checkpoint_path.name:
            policy_path = args.checkpoint
        else:
            # Assume both files exist with same prefix
            prefix = str(checkpoint_path).replace("_value.pt", "").replace("_policy.pt", "")
            policy_path = f"{prefix}_policy.pt"
        
        policy.load_state_dict(torch.load(policy_path, map_location="cpu"))
        print(f"Loaded policy from: {policy_path}")
    else:
        # Basic REINFORCE checkpoint
        policy.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        print(f"Loaded policy from: {args.checkpoint}")
    
    policy.eval()
    
    # Evaluate
    print(f"\nRunning {args.n_episodes} episodes...")
    results = evaluate_policy(
        policy,
        env,
        n_episodes=args.n_episodes,
        deterministic=args.deterministic,
        device="cpu",
        render=args.render
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Mean return: {results['mean_return']:.2f} ± {results['std_return']:.2f}")
    print(f"Min return: {results['min_return']:.2f}")
    print(f"Max return: {results['max_return']:.2f}")
    print(f"Mean length: {results['mean_length']:.2f} ± {results['std_length']:.2f}")
    print("=" * 50)
    
    env.close()


if __name__ == "__main__":
    main()

