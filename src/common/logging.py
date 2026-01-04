"""TensorBoard logging and metrics saving utilities"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class MetricsLogger:
    """Logs metrics to TensorBoard and saves to JSON/CSV"""
    
    def __init__(
        self,
        log_dir: str,
        artifact_dir: str,
        run_name: str = "run"
    ):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
            artifact_dir: Directory for artifacts (checkpoints, metrics, plots)
            run_name: Name of this run
        """
        self.log_dir = Path(log_dir)
        self.artifact_dir = Path(artifact_dir)
        self.run_name = run_name
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        (self.artifact_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.artifact_dir / "plots").mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # In-memory metrics storage
        self.metrics: Dict[str, list] = {}
        
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value to TensorBoard and store in memory."""
        self.writer.add_scalar(tag, value, step)
        
        if tag not in self.metrics:
            self.metrics[tag] = []
        self.metrics[tag].append({"step": step, "value": value})
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int) -> None:
        """Log a histogram to TensorBoard."""
        self.writer.add_histogram(tag, values, step)
    
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


def load_metrics(metrics_file: str) -> Dict[str, list]:
    """Load metrics from a JSON file."""
    with open(metrics_file, 'r') as f:
        return json.load(f)

