#!/usr/bin/env python3
"""Training script for Proximal Policy Optimization (PPO) on MetaWorld tasks with BC initialization."""
import os
import time
import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from tqdm import tqdm
import gymnasium as gym
import h5py

# Set device
device = jax.devices("gpu")[0] if jax.devices("gpu") else jax.devices("cpu")[0]
print(f"Using device: {device}")

# Import modules
from grpo_robots.evaluation import evaluate, EpisodeMonitor
from grpo_robots.ppo_learner import PPOAgent
from grpo_robots.bc_learner import BCAgent
from grpo_robots.utils import set_random_seed, get_metaworld_env, wandb_init

# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'peg-insert-side-v2', 'MetaWorld environment name')
flags.DEFINE_integer('seed', 42, 'Random seed')
flags.DEFINE_integer('eval_episodes', 10, 'Number of episodes for evaluation')
flags.DEFINE_float('actor_lr', 3e-4, 'Learning rate for actor network')
flags.DEFINE_float('critic_lr', 1e-3, 'Learning rate for critic network')
flags.DEFINE_float('bc_lr', 3e-4, 'Learning rate for BC initialization')
flags.DEFINE_integer('batch_size', 1024, 'Batch size for updating')
flags.DEFINE_integer('rollout_length', 8192, 'Number of steps to collect per iteration')
flags.DEFINE_integer('num_iterations', 100, 'Number of training iterations')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs to update on the same data')
flags.DEFINE_integer('log_interval', 1, 'Logging interval (iterations)')
flags.DEFINE_integer('eval_interval', 5, 'Evaluation interval (iterations)')
flags.DEFINE_string('save_dir', 'models', 'Directory to save models')
flags.DEFINE_bool('use_wandb', True, 'Whether to use wandb for logging')
flags.DEFINE_string('wandb_project', 'grpo-robots', 'WandB project name')
flags.DEFINE_string('wandb_name', None, 'WandB run name')
flags.DEFINE_bool('state_dependent_std', True, 'Whether to use state-dependent std in policy')
flags.DEFINE_bool('quiet', False, 'Suppress tqdm progress bar')
flags.DEFINE_float('discount', 0.99, 'Discount factor (gamma)')
flags.DEFINE_float('gae_lambda', 0.99, 'GAE lambda parameter')
flags.DEFINE_float('clip_ratio', 0.2, 'PPO clipping parameter')
flags.DEFINE_float('target_kl', 0.01, 'Target KL divergence for early stopping')
flags.DEFINE_float('entropy_coef', 0.01, 'Entropy bonus coefficient')
flags.DEFINE_float('vf_coef', 0.5, 'Value function loss coefficient')
flags.DEFINE_bool('sparse_reward', False, 'Whether to use sparse rewards')
flags.DEFINE_float('max_grad_norm', 0.5, 'Maximum gradient norm for clipping')
flags.DEFINE_string('demo_file', None, 'Path to demonstration file for BC initialization')
flags.DEFINE_integer('bc_steps', 2000, 'Number of BC gradient steps for initialization')
flags.DEFINE_bool('use_bc_init', True, 'Whether to use BC initialization before PPO')
flags.DEFINE_float('action_smoothing', 0.0, 'Action smoothing factor (0-1) for stability')
flags.DEFINE_float('action_clip', 0.9, 'Action clipping range (-value to value)')
flags.DEFINE_bool('reset_optimizer', False, 'Whether to reset optimizer state after BC initialization')

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

def train_with_bc(observation_dim, action_dim, demo_data, seed, bc_steps=2000, state_dependent_std=True, learning_rate=3e-4, batch_size=1024, use_wandb=False, wandb=None, global_step=0):
    """Train a policy with Behavioral Cloning."""
    print(f"Starting BC initialization for {bc_steps} steps")

    # Create BC agent
    bc_agent = BCAgent.create(
        seed=seed,
        observation_dim=observation_dim,
        action_dim=action_dim,
        learning_rate=learning_rate,
        state_dependent_std=state_dependent_std
    )

    # Training progress bar
    progress_bar = range(1, bc_steps + 1)
    if not FLAGS.quiet:
        progress_bar = tqdm(progress_bar, desc="BC Training")

    # Get dataset
    observations = demo_data['observations']
    actions = demo_data['actions']
    dataset_size = demo_data['size']

    # BC training loop
    for step in progress_bar:
        # Sample batch
        indices = np.random.randint(0, dataset_size, size=batch_size)
        batch = {
            'observations': observations[indices],
            'actions': actions[indices],
        }

        # Update BC agent
        bc_agent, info = bc_agent.update(batch)

        # Log metrics
        if step % 100 == 0:
            if not FLAGS.quiet:
                desc = f"BC Step {step}: MSE Loss: {info['mse_loss']:.6f}"
                progress_bar.set_description(desc)

            if use_wandb and wandb:
                train_metrics = {f'bc_training/{k}': v for k, v in info.items()}
                train_metrics['step'] = global_step  # Add global step for reference
                wandb.log(train_metrics, step=global_step)
                global_step += 1

    print("BC training completed")
    return bc_agent, global_step

def collect_rollout(env, agent, rng, rollout_length=2048, sparse_reward=False, action_smoothing=0.7):
    """Collect a rollout of experience from the environment."""
    observations = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []

    # Reset environment
    observation, info = env.reset()
    done = False
    episode_rewards = []
    episode_reward = 0

    # Add action smoothing with exponential moving average
    last_action = np.zeros(env.action_space.shape)
    action_smoothing_alpha = action_smoothing  # Controls smoothing amount (0.0-1.0)

    collected_steps = 0
    while collected_steps < rollout_length:
        # Sample action
        rng, action_key = jax.random.split(rng)
        action, log_prob, value = agent.sample_actions(
            jnp.array(observation)[None],
            seed=action_key
        )
        action = np.array(action[0])

        # Apply action smoothing to reduce extreme changes
        smoothed_action = action_smoothing_alpha * action + (1 - action_smoothing_alpha) * last_action
        last_action = smoothed_action

        log_prob = np.array(log_prob[0])
        value = np.array(value[0])

        # Store data
        observations.append(observation)
        actions.append(action)  # Store original action for learning
        log_probs.append(log_prob)
        values.append(value)

        # Step environment with smoothed action
        try:
            next_observation, reward, terminated, truncated, info = env.step(smoothed_action)

            # Apply sparse reward if requested
            if sparse_reward:
                reward = 10.0 if info.get('success', False) else 0.0

            episode_reward += reward
            rewards.append(reward)

            # Handle episode termination
            done = terminated or truncated or info.get('success', False)
            dones.append(done)

            if done:
                # Store episode reward and reset environment
                episode_rewards.append(episode_reward)
                episode_reward = 0
                observation, info = env.reset()
                # Reset smoothing on new episode
                last_action = np.zeros(env.action_space.shape)
            else:
                observation = next_observation
        except Exception as e:
            print(f"Error during environment step: {e}")
            # Handle simulation errors by ending episode and resetting
            done = True
            rewards.append(0.0)  # Safe reward
            dones.append(True)
            episode_rewards.append(episode_reward)
            episode_reward = 0
            try:
                observation, info = env.reset()
                # Reset smoothing on new episode
                last_action = np.zeros(env.action_space.shape)
            except:
                # If reset also fails, use zero observation as fallback
                print("Environment reset failed after error")
                observation = np.zeros_like(observation)

        collected_steps += 1

    # Get last value for bootstrapping
    try:
        last_value = agent.get_values(jnp.array(observation)[None])[0]
    except:
        print("Error getting value for final state, using zero")
        last_value = 0.0

    # Convert to numpy arrays
    observations = np.array(observations, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)
    dones = np.array(dones, dtype=np.float32)
    log_probs = np.array(log_probs, dtype=np.float32)
    values = np.array(values, dtype=np.float32)

    # Handle potential NaN values
    observations = np.nan_to_num(observations, nan=0.0)
    actions = np.nan_to_num(actions, nan=0.0)
    rewards = np.nan_to_num(rewards, nan=0.0)
    values = np.nan_to_num(values, nan=0.0)
    log_probs = np.nan_to_num(log_probs, nan=0.0)

    # Compute returns and advantages
    advantages, returns = agent.compute_gae(
        jnp.array(rewards),
        jnp.concatenate([jnp.array(values), jnp.array([last_value])]),
        jnp.array(dones),
        jnp.array(last_value)
    )

    # Return rollout data
    rollout_data = {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'dones': dones,
        'log_probs': log_probs,
        'values': values,
        'advantages': advantages,
        'returns': returns,
        'episode_rewards': episode_rewards,
    }

    return rollout_data

def main(argv):
    # Set random seed
    rng = set_random_seed(FLAGS.seed)

    # Ensure save directory exists
    if FLAGS.save_dir:
        os.makedirs(FLAGS.save_dir, exist_ok=True)

    # Initialize wandb if requested
    wandb = None
    global_step = 0  # Track steps across BC and PPO

    if FLAGS.use_wandb:
        config = {
            'env_name': FLAGS.env_name,
            'seed': FLAGS.seed,
            'actor_lr': FLAGS.actor_lr,
            'critic_lr': FLAGS.critic_lr,
            'bc_lr': FLAGS.bc_lr,
            'batch_size': FLAGS.batch_size,
            'rollout_length': FLAGS.rollout_length,
            'num_iterations': FLAGS.num_iterations,
            'num_epochs': FLAGS.num_epochs,
            'discount': FLAGS.discount,
            'gae_lambda': FLAGS.gae_lambda,
            'clip_ratio': FLAGS.clip_ratio,
            'target_kl': FLAGS.target_kl,
            'entropy_coef': FLAGS.entropy_coef,
            'vf_coef': FLAGS.vf_coef,
            'state_dependent_std': FLAGS.state_dependent_std,
            'sparse_reward': FLAGS.sparse_reward,
            'use_bc_init': FLAGS.use_bc_init,
            'bc_steps': FLAGS.bc_steps,
            'demo_file': FLAGS.demo_file,
            'action_smoothing': FLAGS.action_smoothing,
            'action_clip': FLAGS.action_clip,
            'reset_optimizer': FLAGS.reset_optimizer,
        }
        wandb_name = FLAGS.wandb_name or f"{FLAGS.env_name}_bc_ppo_{int(time.time())}"
        wandb = wandb_init(FLAGS.wandb_project, config, name=wandb_name)

    # Create environment for training and evaluation
    print(f"Creating environment: {FLAGS.env_name}")
    train_env = get_metaworld_env(FLAGS.env_name, seed=FLAGS.seed)
    train_env = EpisodeMonitor(train_env)  # Track episode stats

    eval_env = get_metaworld_env(FLAGS.env_name, seed=FLAGS.seed + 100)  # Different seed for eval

    # Determine observation and action dimensions
    observation_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]

    print(f"Observation dimension: {observation_dim}")
    print(f"Action dimension: {action_dim}")

    # First train with BC if requested
    if FLAGS.use_bc_init:
        if FLAGS.demo_file is None:
            # Use default demo file if not specified
            demo_file = f"{FLAGS.env_name}_expert.hdf5"
            print(f"No demo file specified, using default: {demo_file}")
        else:
            demo_file = FLAGS.demo_file

        # Load demonstrations
        try:
            demo_data = load_hdf5_demonstrations(demo_file)

            # Train with BC
            bc_agent, global_step = train_with_bc(
                observation_dim=observation_dim,
                action_dim=action_dim,
                demo_data=demo_data,
                seed=FLAGS.seed,
                bc_steps=FLAGS.bc_steps,
                state_dependent_std=FLAGS.state_dependent_std,
                learning_rate=FLAGS.bc_lr,
                batch_size=FLAGS.batch_size,
                use_wandb=FLAGS.use_wandb,
                wandb=wandb,
                global_step=global_step
            )

            # Save BC model
            if FLAGS.save_dir:
                bc_model_path = os.path.join(FLAGS.save_dir, f"bc_{FLAGS.env_name}_init.pkl")
                bc_agent.save(bc_model_path)
                print(f"Saved BC model to {bc_model_path}")

            # Evaluate BC policy
            policy_fn = lambda obs: np.array(bc_agent.get_actions(jnp.array(obs)[None])[0])
            bc_stats = evaluate(policy_fn, eval_env, num_episodes=FLAGS.eval_episodes)
            print(f"BC policy evaluation:")
            print(f"  Episode Return: {bc_stats.get('episode_reward', 0):.2f}")
            print(f"  Success Rate: {bc_stats.get('final.success', 0):.2f}")

            if FLAGS.use_wandb and wandb:
                bc_eval_metrics = {f'bc_evaluation/{k}': v for k, v in bc_stats.items()}
                wandb.log(bc_eval_metrics, step=global_step)
                global_step += 1

            # Create PPO agent with BC initialization
            print("Initializing PPO agent from BC policy")
            ppo_agent = PPOAgent.from_bc_agent(
                bc_agent=bc_agent,
                observation_dim=observation_dim,
                action_dim=action_dim,
                actor_lr=FLAGS.actor_lr,
                critic_lr=FLAGS.critic_lr,
                discount=FLAGS.discount,
                gae_lambda=FLAGS.gae_lambda,
                clip_ratio=FLAGS.clip_ratio,
                target_kl=FLAGS.target_kl,
                entropy_coef=FLAGS.entropy_coef,
                vf_coef=FLAGS.vf_coef,
                state_dependent_std=FLAGS.state_dependent_std,
                action_clip=FLAGS.action_clip,
            )
        except (FileNotFoundError, KeyError) as e:
            print(f"Error loading or training with demonstration file: {e}")
            print("Falling back to random initialization for PPO")
            FLAGS.use_bc_init = False

    # If not using BC or BC failed, create PPO agent from scratch
    if not FLAGS.use_bc_init:
        print("Initializing PPO agent from scratch")
        ppo_agent = PPOAgent.create(
            seed=FLAGS.seed,
            observation_dim=observation_dim,
            action_dim=action_dim,
            actor_lr=FLAGS.actor_lr,
            critic_lr=FLAGS.critic_lr,
            discount=FLAGS.discount,
            gae_lambda=FLAGS.gae_lambda,
            clip_ratio=FLAGS.clip_ratio,
            target_kl=FLAGS.target_kl,
            entropy_coef=FLAGS.entropy_coef,
            vf_coef=FLAGS.vf_coef,
            state_dependent_std=FLAGS.state_dependent_std,
            action_clip=FLAGS.action_clip,
        )

    # Training loop
    print(f"Starting training for {FLAGS.num_iterations} iterations")
    best_success_rate = 0.0
    best_return = -float('inf')

    progress_bar = range(1, FLAGS.num_iterations + 1)
    if not FLAGS.quiet:
        progress_bar = tqdm(progress_bar)

    for iteration in progress_bar:
        iteration_start_time = time.time()

        # Collect rollout
        rng, collect_key = jax.random.split(rng)
        rollout_data = collect_rollout(
            train_env,
            ppo_agent,
            collect_key,
            rollout_length=FLAGS.rollout_length,
            sparse_reward=FLAGS.sparse_reward,
            action_smoothing=FLAGS.action_smoothing
        )

        # Process collected data
        observations = rollout_data['observations']
        actions = rollout_data['actions']
        log_probs = rollout_data['log_probs']
        advantages = rollout_data['advantages']
        returns = rollout_data['returns']
        episode_rewards = rollout_data['episode_rewards']

        # Average episode statistics
        avg_episode_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        avg_episode_length = FLAGS.rollout_length / max(1, len(episode_rewards))

        # Multiple epochs of PPO updates on the same data
        kl_div = 0.0
        update_actor = True

        # Mini-batch indices
        indices = np.arange(FLAGS.rollout_length)
        mini_batch_size = FLAGS.batch_size

        train_metrics = {}

        for epoch in range(FLAGS.num_epochs):
            # Shuffle data for mini-batches
            np.random.shuffle(indices)

            # Mini-batch updates
            for start in range(0, FLAGS.rollout_length, mini_batch_size):
                end = start + mini_batch_size
                batch_indices = indices[start:end]

                # Extract mini-batch
                mb_observations = observations[batch_indices]
                mb_actions = actions[batch_indices]
                mb_old_log_probs = log_probs[batch_indices]
                mb_advantages = advantages[batch_indices]
                mb_returns = returns[batch_indices]

                # Update agent on mini-batch
                ppo_agent, info = ppo_agent.update(
                    jnp.array(mb_observations),
                    jnp.array(mb_actions),
                    jnp.array(mb_old_log_probs),
                    jnp.array(mb_advantages),
                    jnp.array(mb_returns),
                    update_actor=update_actor
                )

                # Track metrics
                for k, v in info.items():
                    if k in train_metrics:
                        train_metrics[k].append(float(v))
                    else:
                        train_metrics[k] = [float(v)]

                # Early stopping based on KL divergence
                kl_div = float(info.get('kl', 0.0))
                if kl_div > FLAGS.target_kl * 1.5:
                    update_actor = False
                    # Break early since we're not updating actor anymore
                    break

            # Break if KL too high across all mini-batches
            if not update_actor:
                break

        # Compute mean of training metrics
        train_metrics = {k: np.mean(v) for k, v in train_metrics.items()}
        train_metrics.update({
            'episode_reward': avg_episode_reward,
            'episode_length': avg_episode_length,
            'kl_divergence': kl_div,
            'iterations': iteration,
            'epochs_completed': epoch + 1,
            'time_per_iteration': time.time() - iteration_start_time,
        })

        # Log metrics
        if iteration % FLAGS.log_interval == 0:
            # Print progress
            if not FLAGS.quiet:
                desc = f"Iter {iteration}: Reward: {avg_episode_reward:.2f}, KL: {kl_div:.4f}"
                progress_bar.set_description(desc)

            # Log to wandb
            if wandb:
                wandb_metrics = {f'training/{k}': v for k, v in train_metrics.items()}
                wandb_metrics['step'] = global_step  # Add global step for reference
                wandb.log(wandb_metrics, step=global_step)
                global_step += 1

        # Evaluate periodically
        if iteration % FLAGS.eval_interval == 0 or iteration == FLAGS.num_iterations:
            policy_fn = lambda obs: np.array(ppo_agent.get_actions(jnp.array(obs)[None])[0])
            eval_stats = evaluate(policy_fn, eval_env, num_episodes=FLAGS.eval_episodes)

            # Log evaluation metrics
            print(f"\nEvaluation at iteration {iteration}:")
            eval_return = eval_stats.get('episode_reward', 0)
            success_rate = eval_stats.get('final.success', 0)
            print(f"  Episode Return: {eval_return:.2f}")
            print(f"  Success Rate: {success_rate:.2f}")

            # Save best model based on success rate
            if FLAGS.save_dir and success_rate > best_success_rate:
                best_success_rate = success_rate
                model_path = os.path.join(FLAGS.save_dir, f"ppo_{FLAGS.env_name}_best_success.pkl")
                ppo_agent.save(model_path)
                print(f"  Saved best model with success rate {best_success_rate:.2f}")

            # Also save based on return for comparison
            if FLAGS.save_dir and eval_return > best_return:
                best_return = eval_return
                model_path = os.path.join(FLAGS.save_dir, f"ppo_{FLAGS.env_name}_best_return.pkl")
                ppo_agent.save(model_path)
                print(f"  Saved best model with return {best_return:.2f}")

            if wandb:
                eval_metrics = {f'evaluation/{k}': v for k, v in eval_stats.items()}
                eval_metrics['evaluation/best_success_rate'] = best_success_rate
                eval_metrics['evaluation/best_return'] = best_return
                eval_metrics['step'] = global_step  # Add global step for reference
                wandb.log(eval_metrics, step=global_step)
                global_step += 1

    # Final evaluation
    policy_fn = lambda obs: np.array(ppo_agent.get_actions(jnp.array(obs)[None])[0])
    eval_stats = evaluate(policy_fn, eval_env, num_episodes=FLAGS.eval_episodes)

    print("\nFinal evaluation:")
    print(f"  Episode Return: {eval_stats.get('episode_reward', 0):.2f}")
    print(f"  Success Rate: {eval_stats.get('final.success', 0):.2f}")

    # Save final model
    if FLAGS.save_dir:
        model_path = os.path.join(FLAGS.save_dir, f"ppo_{FLAGS.env_name}_final.pkl")
        ppo_agent.save(model_path)
        print(f"Saved final model to {model_path}")

    # Clean up
    if wandb:
        wandb.finish()

    train_env.close()
    eval_env.close()

if __name__ == '__main__':
    app.run(main)
