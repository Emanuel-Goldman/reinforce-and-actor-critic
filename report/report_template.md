# Assignment 2 – Policy Gradient Methods

**Student Name:** TODO: Your Name  
**Student ID:** TODO: Your Student ID  
**Date:** TODO: Submission Date

**Note:** The assignment text contains conflicting due dates (26/11/2025 at top vs 24/12/2024 inside).  
**TODO: Verify the correct due date from Moodle**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Theoretical Background](#theoretical-background)
3. [Implementation](#implementation)
4. [Experiments](#experiments)
5. [Results and Analysis](#results-and-analysis)
6. [Conclusion](#conclusion)
7. [How to Run](#how-to-run)

---

## 1. Introduction

This report presents the implementation and evaluation of three policy gradient methods:
1. **REINFORCE** (Monte-Carlo Policy Gradient)
2. **REINFORCE with Baseline** (Variance Reduction)
3. **Actor-Critic** (TD-based Policy Gradient)

All algorithms were implemented using PyTorch and evaluated on the CartPole-v1 environment from Gymnasium.

---

## 2. Theoretical Background

### 2.1 Policy Gradient Methods

Policy gradient methods directly optimize the policy parameters θ to maximize expected return:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t \right]$$

where $G_t$ is the return from time step $t$.

### 2.2 REINFORCE

REINFORCE is a Monte-Carlo policy gradient method that:
- Collects full episodes using the current policy
- Computes returns $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$
- Updates policy parameters using the gradient:

$$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t$$

**Advantages:**
- Simple to implement
- Unbiased gradient estimate

**Disadvantages:**
- High variance in gradient estimates
- Requires full episodes before updating

### 2.3 REINFORCE with Baseline

To reduce variance, we subtract a baseline $b(s_t)$ from the return:

$$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot (G_t - b(s_t))$$

**Why use advantage instead of return?**

TODO: Explain why using advantage $A_t = G_t - V(s_t)$ instead of return $G_t$ reduces variance while maintaining an unbiased gradient estimate. Discuss the variance reduction property and provide intuition.

**Baseline no-bias prerequisite and proof:**

TODO: Prove that subtracting a baseline that depends only on the state (not the action) does not introduce bias in the policy gradient estimate. Show mathematically that:

$$\mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot b(s_t) \right] = 0$$

when $b(s_t)$ is independent of the action $a_t$.

### 2.4 Actor-Critic

Actor-Critic methods use a value function (critic) to estimate advantages using TD-error:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

The policy (actor) is updated using:

$$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot \delta_t$$

**TD-error as advantage approximation:**

TODO: Explain why TD-error $\delta_t$ is an approximation of the advantage $A_t$. Show the relationship between TD-error and the true advantage, and discuss when this approximation is accurate. Provide a proof or derivation.

**Actor vs Critic explanation:**

TODO: Explain the roles of the actor and critic networks:
- **Actor (Policy Network):** What it does, what it outputs, how it's updated
- **Critic (Value Network):** What it does, what it outputs, how it's updated
- How they work together in the algorithm

**Advantages:**
- Lower variance than REINFORCE (uses bootstrapping)
- Can update online (after each step)
- More sample-efficient

**Disadvantages:**
- Introduces bias (due to bootstrapping)
- Requires tuning two networks

---

## 3. Implementation

### 3.1 Architecture

The implementation follows a modular structure:

- **`src/common/`**: Utilities for seeding, logging, plotting
- **`src/envs/`**: Environment creation and configuration
- **`src/pg/`**: Policy gradient algorithms
  - `networks.py`: Policy and value network architectures
  - `distributions.py`: Action sampling utilities
  - `buffers.py`: Trajectory storage
  - `reinforce.py`: Basic REINFORCE
  - `reinforce_baseline.py`: REINFORCE with baseline
  - `actor_critic.py`: Actor-Critic
  - `evaluation.py`: Policy evaluation utilities

### 3.2 Network Architecture

Both policy and value networks use multi-layer perceptrons (MLPs) with:
- Input layer: Observation dimension
- Hidden layers: Configurable (default: [128, 128] with ReLU activation)
- Output layer:
  - Policy: Action logits (softmax for discrete actions)
  - Value: Scalar value estimate

### 3.3 Key Implementation Details

1. **Discrete Actions**: Using `torch.distributions.Categorical` for action sampling
2. **Gradient Clipping**: Applied to prevent exploding gradients (max_norm=0.5)
3. **Deterministic Seeding**: For reproducibility
4. **Return Normalization**: Optional normalization of returns/advantages to reduce variance
5. **Entropy Bonus**: Optional entropy regularization for exploration

---

## 4. Experiments

### 4.1 Environment

**Environment:** TODO: Specify the environment used (e.g., CartPole-v1)

**Environment Details:**
- Observation space: TODO
- Action space: TODO
- Episode length: TODO
- Reward structure: TODO

### 4.2 Hyperparameters

**TODO: Fill in the final hyperparameters used for each algorithm after tuning**

#### REINFORCE
- Learning rate (policy): TODO
- Discount factor (γ): TODO
- Hidden layer sizes: TODO
- Normalize returns: TODO (True/False)
- Entropy coefficient: TODO
- Max episodes: TODO

#### REINFORCE with Baseline
- Learning rate (policy): TODO
- Learning rate (value): TODO
- Discount factor (γ): TODO
- Hidden layer sizes: TODO
- Normalize advantages: TODO (True/False)
- Entropy coefficient: TODO
- Max episodes: TODO

#### Actor-Critic
- Learning rate (policy): TODO
- Learning rate (value): TODO
- Discount factor (γ): TODO
- Hidden layer sizes: TODO
- Normalize advantages: TODO (True/False)
- Entropy coefficient: TODO
- Update mode: TODO (step/episode)
- Max episodes: TODO

### 4.3 Experimental Setup

- **Random seeds:** TODO (e.g., seeds 0, 1, 2 for multiple runs)
- **Evaluation:** TODO (e.g., evaluated every 100 episodes over 10 test episodes)
- **Hardware:** TODO (e.g., CPU, GPU model)
- **Software versions:** TODO (PyTorch version, Gymnasium version, etc.)

---

## 5. Results and Analysis

### 5.1 Training Curves

**TODO: Include plots and analysis**

1. **Episode Returns Over Time**
   - Plot showing raw episode returns and moving average
   - Compare all three algorithms
   - Discuss convergence behavior

2. **Loss Curves**
   - Policy loss for all algorithms
   - Value loss for baseline and AC
   - TD-error statistics for AC

3. **Entropy Over Time** (if applicable)
   - Shows exploration behavior

### 5.2 Performance Comparison

**TODO: Create a comparison table**

| Algorithm | Mean Return (Final) | Episodes to 475 | Convergence Time | Sample Efficiency |
|-----------|---------------------|------------------|-------------------|-------------------|
| REINFORCE | TODO | TODO | TODO | TODO |
| REINFORCE+Baseline | TODO | TODO | TODO | TODO |
| Actor-Critic | TODO | TODO | TODO | TODO |

### 5.3 Analysis

**TODO: Provide detailed analysis**

1. **Convergence Speed:**
   - Which algorithm converges fastest?
   - Why? (discuss variance, bias, sample efficiency)

2. **Final Performance:**
   - Which algorithm achieves the best final performance?
   - Are there differences? Why?

3. **Variance Reduction:**
   - Compare variance in returns between REINFORCE and REINFORCE+Baseline
   - How does the baseline help?

4. **Sample Efficiency:**
   - Compare sample efficiency between MC methods (REINFORCE) and TD methods (AC)
   - Discuss the bias-variance trade-off

5. **Hyperparameter Sensitivity:**
   - Which hyperparameters were most critical?
   - How did tuning affect performance?

### 5.4 Discussion

**TODO: Discuss findings, limitations, and potential improvements**

- Strengths and weaknesses of each method
- When would you use each algorithm?
- Potential improvements (e.g., GAE, PPO, etc.)

---

## 6. Conclusion

**TODO: Write a conclusion summarizing:**
- Key findings
- Comparison of the three methods
- Lessons learned
- Future work

---

## 7. How to Run

### 7.1 Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 7.2 Training

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

### 7.3 Testing

```bash
python run/test_agent.py --checkpoint artifacts/section1_basic_reinforce/checkpoints/policy_final.pt --env CartPole-v1 --render
```

### 7.4 Visualization

**TensorBoard:**
```bash
tensorboard --logdir artifacts/
```

**Generate Plots:**
```bash
python run/make_plots.py --metrics_file artifacts/section1_basic_reinforce/metrics.json --output_dir artifacts/section1_basic_reinforce/plots
```

### 7.5 Output Locations

All outputs are saved under `artifacts/`:
- `artifacts/section1_basic_reinforce/` - REINFORCE results
- `artifacts/section1_reinforce_baseline/` - REINFORCE+Baseline results
- `artifacts/section2_actor_critic/` - Actor-Critic results

Each directory contains:
- `checkpoints/` - Saved model weights
- `metrics.json` - Training metrics
- `plots/` - Generated plots (PNG)
- `tensorboard/` - TensorBoard event files

---

## References

TODO: Add references (papers, textbooks, etc.)

1. Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3-4), 229-256.

2. Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.

3. TODO: Add more references as needed

---

**End of Report**

