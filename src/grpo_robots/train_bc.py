#!/usr/bin/env python3
"""Training script for Behavioral Cloning on MetaWorld tasks."""
import os
import time
import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from tqdm import tqdm
import h5py

# Set device
device = jax.devices("gpu")[0] if jax.devices("gpu") else jax.devices("cpu")[0]
print(f"Using device: {device}")

# Import your modules
from grpo_robots.evaluation import evaluate
from grpo_robots.bc_learner import BCAgent
from grpo_robots.utils import set_random_seed, get_metaworld_env, wandb_init, log_metrics

# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'peg-insert-side-v2', 'MetaWorld environment name')
flags.DEFINE_string('demo_file', 'peg-insert-side-v2_expert.hdf5', 'Path to demonstration file')
flags.DEFINE_integer('seed', 42, 'Random seed')
flags.DEFINE_integer('eval_episodes', 10, 'Number of episodes for evaluation')
flags.DEFINE_float('learning_rate', 3e-4, 'Learning rate for BC training')
flags.DEFINE_integer('batch_size', 256, 'Batch size for training')
flags.DEFINE_integer('num_steps', 10000, 'Number of training steps')
flags.DEFINE_integer('log_interval', 100, 'Logging interval')
flags.DEFINE_integer('eval_interval', 1000, 'Evaluation interval')
flags.DEFINE_string('save_dir', 'models', 'Directory to save models')
flags.DEFINE_bool('use_wandb', True, 'Whether to use wandb for logging')
flags.DEFINE_string('wandb_project', 'grpo-robots', 'WandB project name')
flags.DEFINE_string('wandb_name', None, 'WandB run name')
flags.DEFINE_bool('state_dependent_std', False, 'Whether to use state-dependent std in policy')
flags.DEFINE_bool('quiet', False, 'Suppress tqdm progress bar')
flags.DEFINE_integer('subsample_size', None, 'Number of samples to subsample from dataset')

def load_hdf5_demonstrations(file_path):
    """Load demonstrations from HDF5 file."""
    print(f"Loading demonstrations from {file_path}")
    with h5py.File(file_path, 'r') as f:
        # Extract data
        observations = f['observations'][:]
        actions = f['actions'][:]

        # Return data dictionary
        data = {
            'observations': observations,
            'actions': actions,
            'size': observations.shape[0]  # Store dataset size for convenience
        }
        return data

def subsample_dataset(dataset, num_samples):
    """Subsample dataset to a fixed number of samples."""
    return {k: v[:num_samples] for k, v in dataset.items()}

def main(argv):
    # Set random seed
    rng = set_random_seed(FLAGS.seed)

    # Ensure save directory exists
    if FLAGS.save_dir:
        os.makedirs(FLAGS.save_dir, exist_ok=True)

    # Initialize wandb if requested
    wandb = None
    if FLAGS.use_wandb:
        config = {
            'env_name': FLAGS.env_name,
            'demo_file': FLAGS.demo_file,
            'seed': FLAGS.seed,
            'learning_rate': FLAGS.learning_rate,
            'batch_size': FLAGS.batch_size,
            'num_steps': FLAGS.num_steps,
            'state_dependent_std': FLAGS.state_dependent_std
        }
        wandb_name = FLAGS.wandb_name or f"{FLAGS.env_name}_bc_{int(time.time())}"
        wandb = wandb_init(FLAGS.wandb_project, config, name=wandb_name)

    # Load demonstrations
    dataset = load_hdf5_demonstrations(FLAGS.demo_file)
    print(f"Dataset loaded with {dataset['size']} transitions")

    # Create environment for evaluation
    print(f"Creating environment: {FLAGS.env_name}")
    eval_env = get_metaworld_env(FLAGS.env_name, seed=FLAGS.seed)

    # Determine observation and action dimensions
    observation_dim = dataset['observations'].shape[1]
    action_dim = dataset['actions'].shape[1]

    print(f"Observation dimension: {observation_dim}")
    print(f"Action dimension: {action_dim}")

    # Create BC agent
    print("Initializing BC agent")
    bc_agent = BCAgent.create(
        seed=FLAGS.seed,
        observation_dim=observation_dim,
        action_dim=action_dim,
        learning_rate=FLAGS.learning_rate,
        state_dependent_std=FLAGS.state_dependent_std
    )

    # Get dataset arrays for training
    observations = dataset['observations']
    actions = dataset['actions']
    dataset_size = dataset['size']

    # Training loop
    print(f"Starting training for {FLAGS.num_steps} steps")
    best_success_rate = 0.0

    progress_bar = range(1, FLAGS.num_steps + 1)
    if not FLAGS.quiet:
        progress_bar = tqdm(progress_bar)

    for step in progress_bar:
        # Sample batch
        indices = np.random.randint(0, dataset_size, size=FLAGS.batch_size)
        batch = {
            'observations': observations[indices],
            'actions': actions[indices],
        }

        # Update BC agent
        bc_agent, info = bc_agent.update(batch)

        # Log metrics
        if step % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in info.items()}
            if not FLAGS.quiet:
                desc = f"Step {step}: MSE Loss: {info['mse_loss']:.6f}"
                progress_bar.set_description(desc)

            if wandb:
                wandb.log(train_metrics, step=step)

        # Evaluate periodically
        if step % FLAGS.eval_interval == 0 or step == FLAGS.num_steps:
            policy_fn = bc_agent.get_actions
            eval_stats = evaluate(policy_fn, eval_env, num_episodes=FLAGS.eval_episodes)

            # Log evaluation metrics
            print(f"\nEvaluation at step {step}:")
            print(f"  Episode Return: {eval_stats.get('episode_reward', 0):.2f}")
            success_rate = eval_stats.get('final.success', 0)
            print(f"  Success Rate: {success_rate:.2f}")

            # Save best model
            if FLAGS.save_dir and success_rate > best_success_rate:
                best_success_rate = success_rate
                model_path = os.path.join(FLAGS.save_dir, f"bc_{FLAGS.env_name}_best.pkl")
                bc_agent.save(model_path)
                print(f"  Saved best model with success rate {best_success_rate:.2f}")

            if wandb:
                eval_metrics = {f'evaluation/{k}': v for k, v in eval_stats.items()}
                eval_metrics['evaluation/best_success_rate'] = best_success_rate
                wandb.log(eval_metrics, step=step)

    # Final evaluation
    policy_fn = bc_agent.get_actions
    eval_stats = evaluate(policy_fn, eval_env, num_episodes=FLAGS.eval_episodes)

    print("\nFinal evaluation:")
    print(f"  Episode Return: {eval_stats.get('episode_reward', 0):.2f}")
    print(f"  Success Rate: {eval_stats.get('success', 0):.2f}")

    # Save final model
    if FLAGS.save_dir:
        model_path = os.path.join(FLAGS.save_dir, f"bc_{FLAGS.env_name}_final.pkl")
        bc_agent.save(model_path)
        print(f"Saved final model to {model_path}")

    # Clean up
    if wandb:
        wandb.finish()

if __name__ == '__main__':
    app.run(main)