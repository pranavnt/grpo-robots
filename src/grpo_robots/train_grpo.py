#!/usr/bin/env python3
"""Training script for Group Relative Policy Optimization (GRPO) on MetaWorld tasks with BC initialization.

This script closely resembles your train_ppo file but:
  • Uses GRPOAgent (which has no critic) instead of PPOAgent.
  • Computes advantages by group–normalizing returns from terminal transitions.
  • Supports sparse rewards and resetting the optimizer state after BC initialization.
"""

import os
import time
import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from tqdm import tqdm
import gymnasium as gym
import h5py

# Import modules
from grpo_robots.evaluation import evaluate, EpisodeMonitor
from grpo_robots.grpo_learner import GRPOAgent
from grpo_robots.bc_learner import BCAgent
from grpo_robots.utils import set_random_seed, get_metaworld_env, wandb_init

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'peg-insert-side-v2', 'MetaWorld environment name')
flags.DEFINE_integer('seed', 42, 'Random seed')
flags.DEFINE_integer('eval_episodes', 10, 'Number of episodes for evaluation')
flags.DEFINE_float('actor_lr', 3e-4, 'Learning rate for actor network')
flags.DEFINE_float('bc_lr', 3e-4, 'Learning rate for BC initialization')
flags.DEFINE_integer('batch_size', 64, 'Batch size for updating')
flags.DEFINE_integer('rollout_length', 2048, 'Number of steps to collect per iteration')
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
flags.DEFINE_float('clip_ratio', 0.2, 'GRPO clipping parameter')
flags.DEFINE_float('target_kl', 0.01, 'Target KL divergence for early stopping')
flags.DEFINE_float('entropy_coef', 0.01, 'Entropy bonus coefficient')
flags.DEFINE_string('demo_file', None, 'Path to demonstration file for BC initialization')
flags.DEFINE_integer('bc_steps', 2000, 'Number of BC gradient steps for initialization')
flags.DEFINE_bool('use_bc_init', True, 'Whether to use BC initialization before GRPO')
flags.DEFINE_float('action_smoothing', 0.0, 'Action smoothing factor (0-1) for stability')
flags.DEFINE_float('action_clip', 0.9, 'Action clipping range (-value to value)')
flags.DEFINE_bool('sparse_reward', False, 'Whether to use sparse rewards')
flags.DEFINE_bool('reset_optimizer', False, 'Whether to reset optimizer state after BC initialization')


def load_hdf5_demonstrations(file_path):
    print(f"Loading demonstrations from {file_path}")
    with h5py.File(file_path, 'r') as f:
        observations = f['observations'][:]
        actions = f['actions'][:]
        data = {
            'observations': observations,
            'actions': actions,
            'size': observations.shape[0]
        }
        return data


def train_with_bc(observation_dim, action_dim, demo_data, seed, bc_steps=2000,
                  state_dependent_std=True, learning_rate=3e-4, batch_size=256,
                  use_wandb=False, wandb=None, global_step=0):
    print(f"Starting BC initialization for {bc_steps} steps")
    bc_agent = BCAgent.create(
        seed=seed,
        observation_dim=observation_dim,
        action_dim=action_dim,
        learning_rate=learning_rate,
        state_dependent_std=state_dependent_std
    )
    progress_bar = range(1, bc_steps + 1)
    if not FLAGS.quiet:
        progress_bar = tqdm(progress_bar, desc="BC Training")
    observations = demo_data['observations']
    actions = demo_data['actions']
    dataset_size = demo_data['size']
    for step in progress_bar:
        indices = np.random.randint(0, dataset_size, size=batch_size)
        batch = {
            'observations': observations[indices],
            'actions': actions[indices],
        }
        bc_agent, info = bc_agent.update(batch)
        if step % 100 == 0:
            if not FLAGS.quiet:
                progress_bar.set_description(f"BC Step {step}: MSE Loss: {info['mse_loss']:.6f}")
            if use_wandb and wandb:
                wandb.log({f'bc_training/{k}': v for k, v in info.items()}, step=global_step)
                global_step += 1
    print("BC training completed")
    return bc_agent, global_step


def collect_rollout(env, agent, rng, rollout_length=2048, action_smoothing=0.7):
    """Collect a rollout of experience from the environment.

    For GRPO, we mimic the RLHF setting: only terminal transitions yield a nonzero reward.
    """
    observations = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    episode_rewards = []

    observation, info = env.reset()
    episode_reward = 0
    last_action = np.zeros(env.action_space.shape)
    action_smoothing_alpha = action_smoothing
    collected_steps = 0

    while collected_steps < rollout_length:
        rng, action_key = jax.random.split(rng)
        action, log_prob = agent.sample_actions(jnp.array(observation)[None], seed=action_key)
        action = np.array(action[0])
        smoothed_action = action_smoothing_alpha * action + (1 - action_smoothing_alpha) * last_action
        last_action = smoothed_action
        log_prob = np.array(log_prob[0])
        observations.append(observation)
        actions.append(action)
        log_probs.append(log_prob)
        try:
            next_observation, reward, terminated, truncated, info = env.step(smoothed_action)
            # Support for sparse rewards: override reward if enabled
            if FLAGS.sparse_reward:
                reward = 10.0 if info.get('success', False) else 0.0
            episode_reward += reward
            rewards.append(reward)
            d = terminated or truncated or info.get('success', False)
            dones.append(float(d))
            if d:
                episode_rewards.append(episode_reward)
                episode_reward = 0
                observation, info = env.reset()
                last_action = np.zeros(env.action_space.shape)
            else:
                observation = next_observation
        except Exception as e:
            print(f"Error during environment step: {e}")
            rewards.append(0.0)
            dones.append(1.0)
            episode_rewards.append(episode_reward)
            episode_reward = 0
            observation, info = env.reset()
            last_action = np.zeros(env.action_space.shape)
        collected_steps += 1

    observations = np.array(observations, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)
    dones = np.array(dones, dtype=np.float32)
    log_probs = np.array(log_probs, dtype=np.float32)

    rollout_data = {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'dones': dones,
        'log_probs': log_probs,
        'episode_rewards': episode_rewards,
    }
    # For GRPO, set returns to be nonzero only on terminal transitions.
    returns = np.where(dones == 1, rewards, 0.0)
    rollout_data['returns'] = returns
    return rollout_data


def main(argv):
    rng = set_random_seed(FLAGS.seed)
    if FLAGS.save_dir:
        os.makedirs(FLAGS.save_dir, exist_ok=True)
    wandb = None
    global_step = 0
    if FLAGS.use_wandb:
        config = {
            'env_name': FLAGS.env_name,
            'seed': FLAGS.seed,
            'actor_lr': FLAGS.actor_lr,
            'batch_size': FLAGS.batch_size,
            'rollout_length': FLAGS.rollout_length,
            'num_iterations': FLAGS.num_iterations,
            'clip_ratio': FLAGS.clip_ratio,
            'target_kl': FLAGS.target_kl,
            'entropy_coef': FLAGS.entropy_coef,
            'state_dependent_std': FLAGS.state_dependent_std,
            'use_bc_init': FLAGS.use_bc_init,
            'bc_steps': FLAGS.bc_steps,
            'action_smoothing': FLAGS.action_smoothing,
            'action_clip': FLAGS.action_clip,
            'sparse_reward': FLAGS.sparse_reward,
            'reset_optimizer': FLAGS.reset_optimizer,
        }
        wandb_name = FLAGS.wandb_name or f"{FLAGS.env_name}_bc_grpo_{int(time.time())}"
        wandb = wandb_init(FLAGS.wandb_project, config, name=wandb_name)

    print(f"Creating environment: {FLAGS.env_name}")
    train_env = get_metaworld_env(FLAGS.env_name, seed=FLAGS.seed)
    train_env = EpisodeMonitor(train_env)
    eval_env = get_metaworld_env(FLAGS.env_name, seed=FLAGS.seed + 100)
    observation_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]
    print(f"Observation dimension: {observation_dim}")
    print(f"Action dimension: {action_dim}")

    if FLAGS.use_bc_init:
        demo_file = FLAGS.demo_file if FLAGS.demo_file else f"{FLAGS.env_name}_expert.hdf5"
        try:
            demo_data = load_hdf5_demonstrations(demo_file)
            bc_agent, global_step = train_with_bc(
                observation_dim, action_dim, demo_data, FLAGS.seed,
                bc_steps=FLAGS.bc_steps,
                state_dependent_std=FLAGS.state_dependent_std,
                learning_rate=FLAGS.bc_lr,
                batch_size=FLAGS.batch_size,
                use_wandb=FLAGS.use_wandb,
                wandb=wandb,
                global_step=global_step
            )
            if FLAGS.save_dir:
                bc_model_path = os.path.join(FLAGS.save_dir, f"bc_{FLAGS.env_name}_init.pkl")
                bc_agent.save(bc_model_path)
                print(f"Saved BC model to {bc_model_path}")
            policy_fn = lambda obs: np.array(bc_agent.get_actions(jnp.array(obs)[None])[0])
            bc_stats = evaluate(policy_fn, eval_env, num_episodes=FLAGS.eval_episodes)
            print(f"BC policy evaluation: Return: {bc_stats.get('episode_reward', 0):.2f}, Success: {bc_stats.get('final.success', 0):.2f}")
            if FLAGS.use_wandb and wandb:
                wandb.log({f'bc_evaluation/{k}': v for k, v in bc_stats.items()}, step=global_step)
                global_step += 1
            print("Initializing GRPO agent from BC policy")
            grpo_agent = GRPOAgent.from_bc_agent(
                bc_agent, observation_dim, action_dim,
                actor_lr=FLAGS.actor_lr,
                clip_ratio=FLAGS.clip_ratio,
                target_kl=FLAGS.target_kl,
                entropy_coef=FLAGS.entropy_coef,
                state_dependent_std=FLAGS.state_dependent_std,
                action_clip=FLAGS.action_clip
            )
            # If requested, reset the optimizer state after BC initialization.
            if FLAGS.reset_optimizer:
                grpo_agent = grpo_agent.replace(
                    actor=grpo_agent.actor.replace(
                        opt_state=grpo_agent.actor.tx.init(grpo_agent.actor.params)
                    )
                )
        except (FileNotFoundError, KeyError) as e:
            print(f"Error during BC initialization: {e}")
            print("Falling back to random initialization for GRPO")
            FLAGS.use_bc_init = False

    if not FLAGS.use_bc_init:
        print("Initializing GRPO agent from scratch")
        grpo_agent = GRPOAgent.create(
            seed=FLAGS.seed,
            observation_dim=observation_dim,
            action_dim=action_dim,
            actor_lr=FLAGS.actor_lr,
            clip_ratio=FLAGS.clip_ratio,
            target_kl=FLAGS.target_kl,
            entropy_coef=FLAGS.entropy_coef,
            state_dependent_std=FLAGS.state_dependent_std,
            action_clip=FLAGS.action_clip
        )

    print(f"Starting training for {FLAGS.num_iterations} iterations")
    best_success_rate = 0.0
    best_return = -float('inf')
    progress_bar = range(1, FLAGS.num_iterations + 1)
    if not FLAGS.quiet:
        progress_bar = tqdm(progress_bar)

    for iteration in progress_bar:
        iter_start = time.time()
        rng, collect_key = jax.random.split(rng)
        rollout_data = collect_rollout(
            train_env,
            grpo_agent,
            collect_key,
            rollout_length=FLAGS.rollout_length,
            action_smoothing=FLAGS.action_smoothing
        )
        observations = rollout_data['observations']
        actions = rollout_data['actions']
        log_probs = rollout_data['log_probs']
        returns = rollout_data['returns']
        dones = rollout_data['dones']
        episode_rewards = rollout_data['episode_rewards']

        # Compute group-normalized advantages only on terminal transitions.
        if np.sum(dones) > 0:
            terminal_returns = returns[dones == 1]
            mean_r = terminal_returns.mean()
            std_r = terminal_returns.std() + 1e-8
            norm_advantages = np.where(dones == 1, (returns - mean_r) / std_r, 0.0)
        else:
            norm_advantages = returns

        avg_episode_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        avg_episode_length = FLAGS.rollout_length / max(1, len(episode_rewards))
        indices = np.arange(FLAGS.rollout_length)
        mini_batch_size = FLAGS.batch_size
        train_metrics = {}

        for epoch in range(FLAGS.num_epochs):
            np.random.shuffle(indices)
            for start in range(0, FLAGS.rollout_length, mini_batch_size):
                end = start + mini_batch_size
                batch_indices = indices[start:end]
                mb_obs = observations[batch_indices]
                mb_actions = actions[batch_indices]
                mb_old_log_probs = log_probs[batch_indices]
                mb_advantages = norm_advantages[batch_indices]
                grpo_agent, info = grpo_agent.update(
                    jnp.array(mb_obs),
                    jnp.array(mb_actions),
                    jnp.array(mb_old_log_probs),
                    jnp.array(mb_advantages)
                )
                for k, v in info.items():
                    train_metrics.setdefault(k, []).append(float(v))
        train_metrics = {k: np.mean(v) for k, v in train_metrics.items()}
        train_metrics.update({
            'episode_reward': avg_episode_reward,
            'episode_length': avg_episode_length,
            'iterations': iteration,
            'epochs_completed': epoch + 1,
            'time_per_iteration': time.time() - iter_start,
        })

        if iteration % FLAGS.log_interval == 0:
            if not FLAGS.quiet:
                progress_bar.set_description(f"Iter {iteration}: Reward: {avg_episode_reward:.2f}")
            if wandb:
                wandb.log({f'training/{k}': v for k, v in train_metrics.items()}, step=global_step)
                global_step += 1

        if iteration % FLAGS.eval_interval == 0 or iteration == FLAGS.num_iterations:
            policy_fn = lambda obs: np.array(grpo_agent.get_actions(jnp.array(obs)[None])[0])
            eval_stats = evaluate(policy_fn, eval_env, num_episodes=FLAGS.eval_episodes)
            print(f"\nEvaluation at iteration {iteration}: Return: {eval_stats.get('episode_reward', 0):.2f}, Success: {eval_stats.get('final.success', 0):.2f}")

            if FLAGS.save_dir and eval_stats.get('final.success', 0) > best_success_rate:
                best_success_rate = eval_stats.get('final.success', 0)
                model_path = os.path.join(FLAGS.save_dir, f"grpo_{FLAGS.env_name}_best_success.pkl")
                grpo_agent.save(model_path)
                print(f"Saved best model with success rate {best_success_rate:.2f}")

            if FLAGS.save_dir and eval_stats.get('episode_reward', 0) > best_return:
                best_return = eval_stats.get('episode_reward', 0)
                model_path = os.path.join(FLAGS.save_dir, f"grpo_{FLAGS.env_name}_best_return.pkl")
                grpo_agent.save(model_path)
                print(f"Saved best model with return {best_return:.2f}")

            if wandb:
                eval_metrics = {f'evaluation/{k}': v for k, v in eval_stats.items()}
                eval_metrics['evaluation/best_success_rate'] = best_success_rate
                eval_metrics['evaluation/best_return'] = best_return
                wandb.log(eval_metrics, step=global_step)
                global_step += 1

    policy_fn = lambda obs: np.array(grpo_agent.get_actions(jnp.array(obs)[None])[0])
    eval_stats = evaluate(policy_fn, eval_env, num_episodes=FLAGS.eval_episodes)
    print(f"\nFinal evaluation: Return: {eval_stats.get('episode_reward', 0):.2f}, Success: {eval_stats.get('final.success', 0):.2f}")
    if FLAGS.save_dir:
        model_path = os.path.join(FLAGS.save_dir, f"grpo_{FLAGS.env_name}_final.pkl")
        grpo_agent.save(model_path)
        print(f"Saved final model to {model_path}")
    if wandb:
        wandb.finish()
    train_env.close()
    eval_env.close()


if __name__ == '__main__':
    app.run(main)
