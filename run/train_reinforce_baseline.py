#!/usr/bin/env python3
"""Training script for REINFORCE with baseline"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import gymnasium as gym

from src.common.seed import set_seed
from src.common.logging import MetricsLogger
from src.common.plotting import plot_metrics
from src.envs.make_env import make_env, get_env_info
from src.pg.reinforce_baseline import REINFORCEBaseline
from src.pg.evaluation import evaluate_policy


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train REINFORCE with baseline agent")
    
    # Environment
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
    
    # Algorithm
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr_policy", type=float, default=3e-4, help="Policy learning rate")
    parser.add_argument("--lr_value", type=float, default=1e-3, help="Value learning rate")
    parser.add_argument("--hidden_sizes", type=str, default="128,128", help="Hidden layer sizes (comma-separated)")
    parser.add_argument("--normalize_advantages", action="store_true", help="Normalize advantages")
    parser.add_argument("--entropy_coef", type=float, default=0.0, help="Entropy bonus coefficient")
    
    # Training
    parser.add_argument("--max_episodes", type=int, default=1000, help="Maximum training episodes")
    parser.add_argument("--eval_interval", type=int, default=100, help="Evaluation interval (episodes)")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Number of episodes for evaluation")
    
    # Output
    parser.add_argument("--log_dir", type=str, default=None, help="TensorBoard log directory")
    parser.add_argument("--artifact_dir", type=str, default=None, help="Artifact output directory")
    
    # TODO: Tune learning rates / hidden sizes / gamma / entropy coef
    # TODO: Choose final hyperparameters for report
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create environment
    env = make_env(args.env, seed=args.seed)
    env_info = get_env_info(env)
    
    print(f"Environment: {args.env}")
    print(f"Observation dim: {env_info['obs_dim']}")
    print(f"Action dim: {env_info['action_dim']}")
    print(f"Discrete actions: {env_info['is_discrete']}")
    
    # Parse hidden sizes
    from src.common.utils import parse_hidden_sizes
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    
    # Create agent
    agent = REINFORCEBaseline(
        obs_dim=env_info['obs_dim'],
        action_dim=env_info['action_dim'],
        hidden_sizes=hidden_sizes,
        lr_policy=args.lr_policy,
        lr_value=args.lr_value,
        gamma=args.gamma,
        normalize_advantages=args.normalize_advantages,
        entropy_coef=args.entropy_coef,
        device="cpu"
    )
    
    # Setup logging
    if args.log_dir is None:
        args.log_dir = f"artifacts/section1_reinforce_baseline/tensorboard/seed_{args.seed}"
    if args.artifact_dir is None:
        args.artifact_dir = f"artifacts/section1_reinforce_baseline"
    
    logger = MetricsLogger(
        log_dir=args.log_dir,
        artifact_dir=args.artifact_dir,
        run_name=f"reinforce_baseline_{args.env}_{args.seed}"
    )
    
    # Training loop
    best_mean_return = float('-inf')
    episodes_to_threshold = None
    threshold = 475.0  # CartPole-v1 threshold
    
    print("\nStarting training...")
    print(f"Max episodes: {args.max_episodes}")
    print(f"Gamma: {args.gamma}, Policy LR: {args.lr_policy}, Value LR: {args.lr_value}")
    print("-" * 50)
    
    for episode in range(1, args.max_episodes + 1):
        obs, info = env.reset()
        episode_return = 0.0
        episode_length = 0
        done = False
        
        # Collect episode
        while not done and episode_length < args.max_steps:
            # Select action and estimate value
            action, value = agent.select_action(obs, deterministic=False)
            logits = agent.policy(torch.FloatTensor(obs).unsqueeze(0))
            from src.pg.distributions import get_log_prob
            log_prob = get_log_prob(logits, torch.LongTensor([action])).item()
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(obs, action, reward, log_prob, value, done=done)
            
            obs = next_obs
            episode_return += reward
            episode_length += 1
        
        # Update policy and value
        metrics = agent.update()
        
        # Log metrics
        logger.log_scalar("episode_return", episode_return, episode)
        logger.log_scalar("episode_length", episode_length, episode)
        if metrics:
            logger.log_scalar("policy_loss", metrics.get("policy_loss", 0), episode)
            logger.log_scalar("value_loss", metrics.get("value_loss", 0), episode)
            if "entropy" in metrics:
                logger.log_scalar("entropy", metrics["entropy"], episode)
            if "mean_advantage" in metrics:
                logger.log_scalar("mean_advantage", metrics["mean_advantage"], episode)
        
        # Evaluation
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
                  f"Policy Loss: {metrics.get('policy_loss', 0):.4f} | "
                  f"Value Loss: {metrics.get('value_loss', 0):.4f}")
            
            # Track best performance
            if mean_return > best_mean_return:
                best_mean_return = mean_return
                agent.save(f"{args.artifact_dir}/checkpoints/agent_best")
            
            # Track convergence
            if episodes_to_threshold is None and mean_return >= threshold:
                episodes_to_threshold = episode
                print(f"  ✓ Reached threshold {threshold} at episode {episode}")
    
    # Save final model
    agent.save(f"{args.artifact_dir}/checkpoints/agent_final")
    
    # Save metrics and generate plots
    logger.save_metrics()
    plot_metrics(logger.get_metrics(), args.artifact_dir)
    logger.close()
    
    # Print summary
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

