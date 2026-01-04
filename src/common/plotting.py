"""Plotting utilities for training metrics"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def moving_average(data: np.ndarray, window: int = 100) -> np.ndarray:
    """Compute moving average of data."""
    if len(data) < window:
        return np.convolve(data, np.ones(window) / window, mode='valid')
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_metrics(
    metrics: Dict[str, List[Dict[str, float]]],
    output_dir: Path,
    window: int = 100
) -> None:
    """
    Generate plots from metrics dictionary.
    
    Args:
        metrics: Dictionary with metric names as keys and lists of {step, value} dicts
        output_dir: Directory to save plots
        window: Window size for moving average
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract episode returns
    if "episode_return" in metrics:
        returns = [m["value"] for m in metrics["episode_return"]]
        steps = [m["step"] for m in metrics["episode_return"]]
        
        # Plot raw returns
        plt.figure(figsize=(10, 6))
        plt.plot(steps, returns, alpha=0.3, label="Episode Return", color="blue")
        
        # Plot moving average
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
    
    # Plot policy loss
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
    
    # Plot value loss
    if "value_loss" in metrics:
        losses = [m["value"] for m in metrics["value_loss"]]
        steps = [m["step"] for m in metrics["value_loss"]]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, alpha=0.7, label="Value Loss", color="orange")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title("Value Loss Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "value_loss.png", dpi=150)
        plt.close()
    
    # Plot TD error (for actor-critic)
    if "td_error" in metrics:
        td_errors = [m["value"] for m in metrics["td_error"]]
        steps = [m["step"] for m in metrics["td_error"]]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, td_errors, alpha=0.7, label="TD Error", color="purple")
        plt.xlabel("Step")
        plt.ylabel("TD Error")
        plt.title("TD Error Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "td_error.png", dpi=150)
        plt.close()
    
    # Plot entropy (if available)
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


def plot_comparison(
    metrics_files: List[str],
    labels: List[str],
    output_file: Path,
    metric_name: str = "episode_return",
    window: int = 100
) -> None:
    """
    Plot comparison of multiple runs.
    
    Args:
        metrics_files: List of paths to metrics JSON files
        labels: List of labels for each run
        output_file: Path to save comparison plot
        metric_name: Name of metric to compare
        window: Window size for moving average
    """
    plt.figure(figsize=(12, 6))
    
    for metrics_file, label in zip(metrics_files, labels):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        if metric_name in metrics:
            values = [m["value"] for m in metrics[metric_name]]
            steps = [m["step"] for m in metrics[metric_name]]
            
            if len(values) >= window:
                ma_values = moving_average(np.array(values), window)
                ma_steps = steps[window-1:]
                plt.plot(ma_steps, ma_values, label=label, linewidth=2)
    
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(f"Comparison: {metric_name.replace('_', ' ').title()}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Comparison plot saved to {output_file}")

