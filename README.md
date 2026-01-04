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
python run/train_basic_reinforce.py --env CartPole-v1 --seed 0 --max_episodes 1000
```

**REINFORCE with Baseline:**
```bash
python run/train_reinforce_baseline.py --env CartPole-v1 --seed 0 --max_episodes 1000
```

**Actor-Critic:**
```bash
python run/train_actor_critic.py --env CartPole-v1 --seed 0 --max_episodes 1000
```

### Testing a Trained Agent

```bash
python run/test_agent.py --checkpoint artifacts/section1_basic_reinforce/checkpoints/policy_final.pt --env CartPole-v1 --render
```

### Viewing TensorBoard Logs

```bash
tensorboard --logdir artifacts/
```

Then open `http://localhost:6006` in your browser.

### Generating Plots

```bash
python run/make_plots.py --metrics_file artifacts/section1_basic_reinforce/metrics.json --output_dir artifacts/section1_basic_reinforce/plots
```

## Output Locations

All outputs are saved under `artifacts/`:

- `artifacts/section1_basic_reinforce/`
  - `checkpoints/` - Saved model weights
  - `metrics.json` - Training metrics (episode returns, losses, etc.)
  - `plots/` - Generated plots (PNG files)
  - `tensorboard/` - TensorBoard event files

- `artifacts/section1_reinforce_baseline/` - Same structure
- `artifacts/section2_actor_critic/` - Same structure

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
- `--normalize_returns`: Normalize returns/advantages (default: `False`)
- `--entropy_coef`: Entropy bonus coefficient (default: `0.0`, AC only)

## Project Structure

```
reinforce-and-actor-critic/
├── src/              # Source code modules
│   ├── common/       # Utilities (logging, plotting, seeding)
│   ├── envs/         # Environment creation
│   └── pg/           # Policy gradient algorithms
├── run/              # Training and testing scripts
├── scripts/          # Utility scripts
├── report/           # Report template
└── artifacts/        # Outputs (checkpoints, logs, plots)
```

## Notes

- All algorithms use deterministic seeding for reproducibility
- Metrics are saved to JSON for easy analysis
- TensorBoard logs include scalar metrics and histograms
- Checkpoints are saved periodically and at the end of training

