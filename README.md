# Assignment 2 – Policy Gradient Methods

This repository implements three policy gradient algorithms:
1. **REINFORCE** (Monte-Carlo Policy Gradient)
2. **REINFORCE with Baseline** (Variance Reduction)
3. **Actor-Critic** (TD-based Policy Gradient)

## Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Training

**Basic REINFORCE:**
```bash
python reinforce.py --env CartPole-v1 --seed 0 --max_episodes 1000
```

**REINFORCE with Baseline:**
```bash
python reinforce_baseline.py --env CartPole-v1 --seed 0 --max_episodes 1000
```

**Actor-Critic:**
```bash
python actor_critic.py --env CartPole-v1 --seed 0 --max_episodes 1000
```

### Viewing TensorBoard Logs

```bash
tensorboard --logdir results/
```

Then open `http://localhost:6006` in your browser.

## Output Locations

All outputs are saved under `results/`:

- `results/reinforce/`
  - `checkpoints/` - Saved model weights
  - `metrics.json` - Training metrics (episode returns, losses, etc.)
  - `plots/` - Generated plots (PNG files)
  - `tensorboard/` - TensorBoard event files

- `results/reinforce_baseline/` - Same structure
- `results/actor_critic/` - Same structure

## Command-Line Arguments

All training scripts support:

- `--env`: Environment name (default: `CartPole-v1`)
- `--seed`: Random seed (default: `0`)
- `--gamma`: Discount factor (default: `0.99`)
- `--lr_policy`: Policy learning rate (default: `3e-4`)
- `--lr_value`: Value learning rate (default: `1e-3`, baseline/AC only)
- `--hidden_sizes`: Comma-separated hidden layer sizes (default: `128,128`)
- `--max_episodes`: Maximum training episodes (default: `1000`)
- `--max_steps`: Maximum steps per episode (default: `500`)
- `--log_dir`: TensorBoard log directory (default: auto-generated)
- `--artifact_dir`: Artifact output directory (default: auto-generated)
- `--normalize_returns`: Normalize returns (REINFORCE only, default: `False`)
- `--normalize_advantages`: Normalize advantages (baseline/AC only, default: `False`)
- `--entropy_coef`: Entropy bonus coefficient (default: `0.0` for REINFORCE/baseline, `0.01` for AC)
- `--eval_interval`: Evaluation interval in episodes (default: `100`)
- `--eval_episodes`: Number of episodes for evaluation (default: `10`)
- `--update_mode`: Update mode for Actor-Critic (default: `episode`, choices: `step` or `episode`)

## Project Structure

```
reinforce-and-actor-critic/
├── reinforce.py              # REINFORCE algorithm (standalone)
├── reinforce_baseline.py     # REINFORCE with baseline (standalone)
├── actor_critic.py           # Actor-Critic algorithm (standalone)
├── README.md                 # This file
├── requirements.txt          # Python dependencies
└── results/                  # Outputs (checkpoints, logs, plots)
```

## Notes

- All algorithms use deterministic seeding for reproducibility
- Metrics are saved to JSON for easy analysis
- TensorBoard logs include scalar metrics and histograms
- Checkpoints are saved periodically and at the end of training
- Each script is completely self-contained with no local imports
