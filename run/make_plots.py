#!/usr/bin/env python3
"""Generate plots from saved metrics"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.plotting import plot_metrics, load_metrics


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate plots from metrics")
    
    parser.add_argument("--metrics_file", type=str, required=True, help="Path to metrics JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load metrics
    print(f"Loading metrics from: {args.metrics_file}")
    metrics = load_metrics(args.metrics_file)
    
    # Generate plots
    print(f"Generating plots in: {args.output_dir}")
    plot_metrics(metrics, Path(args.output_dir))
    
    print("Done!")


if __name__ == "__main__":
    main()

