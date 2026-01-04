"""Neural network architectures for policy and value functions"""

import torch
import torch.nn as nn
from typing import List, Optional


class MLP(nn.Module):
    """Multi-layer perceptron with configurable hidden layers."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: List[int],
        activation: str = "relu",
        output_activation: Optional[str] = None
    ):
        """
        Initialize MLP.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_sizes: List of hidden layer sizes
            activation: Activation function for hidden layers ("relu", "tanh", etc.)
            output_activation: Activation function for output layer (None, "tanh", "sigmoid", etc.)
        """
        super().__init__()
        
        # Build layers
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
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_dim))
        if output_activation == "tanh":
            layers.append(nn.Tanh())
        elif output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif output_activation is not None:
            raise ValueError(f"Unknown output activation: {output_activation}")
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)


class PolicyNetwork(nn.Module):
    """Policy network that outputs action logits for discrete actions."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = [128, 128]
    ):
        """
        Initialize policy network.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension (number of discrete actions)
            hidden_sizes: List of hidden layer sizes
        """
        super().__init__()
        self.mlp = MLP(
            input_dim=obs_dim,
            output_dim=action_dim,
            hidden_sizes=hidden_sizes,
            activation="relu",
            output_activation=None  # Raw logits
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            obs: Observation tensor (batch_size, obs_dim) or (obs_dim,)
        
        Returns:
            Action logits (batch_size, action_dim) or (action_dim,)
        """
        return self.mlp(obs)


class ValueNetwork(nn.Module):
    """Value network that estimates V(s)."""
    
    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: List[int] = [128, 128]
    ):
        """
        Initialize value network.
        
        Args:
            obs_dim: Observation dimension
            hidden_sizes: List of hidden layer sizes
        """
        super().__init__()
        self.mlp = MLP(
            input_dim=obs_dim,
            output_dim=1,
            hidden_sizes=hidden_sizes,
            activation="relu",
            output_activation=None  # Raw value estimate
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            obs: Observation tensor (batch_size, obs_dim) or (obs_dim,)
        
        Returns:
            Value estimate (batch_size, 1) or (1,)
        """
        return self.mlp(obs).squeeze(-1)  # Remove last dimension if it's 1

