# GRPO Robots

Reinforcement learning for robotic manipulation tasks using Metaworld.

## Overview

This repository implements behavioral cloning (BC) and proximal policy optimization (PPO) algorithms for robotic manipulation tasks from the Metaworld environment suite. It can train agents in two ways:

1. Pure PPO training from scratch
2. BC initialization followed by PPO fine-tuning (BC-PPO)

## Requirements

- Python 3.12+
- JAX
- Flax
- MetaWorld

## Installation

```bash
# Install dependencies
uv pip install -e .
```

## Training

### Train PPO from scratch

```bash
# Basic PPO training
uv run python src/grpo_robots/train_ppo.py --env_name peg-insert-side-v2

# PPO with sparse rewards
uv run python src/grpo_robots/train_ppo.py --env_name peg-insert-side-v2 --sparse_reward=True
```

### Train BC + PPO (BC initialization, then PPO fine-tuning)

```bash
# BC-PPO with default settings (2000 BC steps, then PPO)
uv run python src/grpo_robots/train_ppo.py --env_name peg-insert-side-v2 --use_bc_init=True --demo_file=peg-insert-side-v2_expert.hdf5

# BC-PPO with custom steps and sparse rewards
uv run python src/grpo_robots/train_ppo.py --env_name peg-insert-side-v2 --use_bc_init=True --demo_file=peg-insert-side-v2_expert.hdf5 --bc_steps=5000 --sparse_reward=True
```

### Train BC Only

```bash
# Pure BC training
uv run python src/grpo_robots/train_bc.py --env_name peg-insert-side-v2 --demo_file=peg-insert-side-v2_expert.hdf5
```

## Important Parameters

- `--env_name`: Metaworld environment name
- `--use_bc_init`: Whether to initialize with BC before PPO
- `--bc_steps`: Number of BC gradient steps for initialization (default: 2000)
- `--demo_file`: Path to demonstration file for BC
- `--sparse_reward`: Use sparse rewards (10.0 for success, 0 otherwise)
- `--num_iterations`: Number of PPO iterations
- `--save_dir`: Directory to save models

## Experiment Tracking

The code uses Weights & Biases for experiment tracking. Set `--use_wandb=True` to enable logging.