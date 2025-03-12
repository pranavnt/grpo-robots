"""Proximal Policy Optimization (PPO) implementation."""
import jax
import jax.numpy as jnp
import flax
import optax
from typing import Tuple, Dict, Any, Optional, NamedTuple, Union
import pickle
import os

from grpo_robots.networks import Policy, ValueCritic, MLP
from grpo_robots.train_state import TrainState

class PPOTrainState(NamedTuple):
    """PPO train state containing both actor and critic states."""
    actor: TrainState
    critic: TrainState

class PPOAgent(flax.struct.PyTreeNode):
    """Proximal Policy Optimization agent."""
    actor: TrainState
    critic: TrainState
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    target_kl: float = 0.01
    entropy_coef: float = 0.01
    vf_coef: float = 0.5
    action_clip: float = 0.9  # Prevent extreme actions

    @classmethod
    def create(
        cls,
        seed: int,
        observation_dim: int,
        action_dim: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        hidden_dims: Tuple[int, ...] = (256, 256),
        discount: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        target_kl: float = 0.01,
        entropy_coef: float = 0.01,
        vf_coef: float = 0.5,
        state_dependent_std: bool = True,
        action_clip: float = 0.9,
    ):
        """Create a new PPO agent.

        Args:
            seed: Random seed
            observation_dim: Dimension of observation space
            action_dim: Dimension of action space
            actor_lr: Learning rate for policy optimization
            critic_lr: Learning rate for value function optimization
            hidden_dims: Hidden dimensions for networks
            discount: Reward discount factor (gamma)
            gae_lambda: GAE lambda parameter
            clip_ratio: PPO clipping parameter
            target_kl: Target KL divergence
            entropy_coef: Entropy bonus coefficient
            vf_coef: Value function loss coefficient
            state_dependent_std: Whether std is state-dependent or learned
            action_clip: Value to clip actions to prevent extreme values

        Returns:
            A new PPOAgent
        """
        rng = jax.random.PRNGKey(seed)
        actor_key, critic_key = jax.random.split(rng)

        # Configure actor network (policy)
        actor_def = Policy(
            hidden_dims=hidden_dims,
            action_dim=action_dim,
            log_std_min=-10.0,
            log_std_max=0.0,  # More conservative upper bound
            state_dependent_std=state_dependent_std,
            tanh_squash_distribution=True,
            final_fc_init_scale=0.01
        )

        # Configure critic network (value function)
        critic_def = ValueCritic(hidden_dims=hidden_dims)

        # Initialize actor
        dummy_obs = jnp.zeros((1, observation_dim))
        actor_params = actor_def.init(actor_key, dummy_obs)['params']

        # Initialize critic
        critic_params = critic_def.init(critic_key, dummy_obs)['params']

        # Setup optimizers with gradient clipping and warm-up
        # Use smaller learning rate at beginning
        warmup_steps = 1000
        actor_lr_schedule = optax.join_schedules([
            optax.linear_schedule(actor_lr * 0.1, actor_lr, warmup_steps),
            optax.constant_schedule(actor_lr)
        ], [warmup_steps])

        critic_lr_schedule = optax.join_schedules([
            optax.linear_schedule(critic_lr * 0.1, critic_lr, warmup_steps),
            optax.constant_schedule(critic_lr)
        ], [warmup_steps])

        actor_tx = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(learning_rate=actor_lr_schedule, eps=1e-5)
        )

        critic_tx = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(learning_rate=critic_lr_schedule, eps=1e-5)
        )

        # Create train states
        actor = TrainState.create(actor_def, actor_params, tx=actor_tx)
        critic = TrainState.create(critic_def, critic_params, tx=critic_tx)

        return cls(
            actor=actor,
            critic=critic,
            discount=discount,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            target_kl=target_kl,
            entropy_coef=entropy_coef,
            vf_coef=vf_coef,
            action_clip=action_clip
        )

    @jax.jit
    def compute_gae(
        self,
        rewards: jnp.ndarray,
        values: jnp.ndarray,
        dones: jnp.ndarray,
        last_value: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: (T) rewards
            values: (T+1) value estimates (includes last_value)
            dones: (T) done flags
            last_value: Value estimate for final state

        Returns:
            advantages: (T) advantage estimates
            returns: (T) return estimates
        """
        # First handle any NaN values in the inputs
        rewards = jnp.nan_to_num(rewards, nan=0.0)
        values = jnp.nan_to_num(values, nan=0.0)
        dones = jnp.nan_to_num(dones, nan=1.0)
        last_value = jnp.nan_to_num(last_value, nan=0.0)

        # Use jax.lax.scan for computing GAE recursively
        def gae_step(carry, transition_and_mask):
            # Unpack values
            delta, mask = transition_and_mask
            advantage, last_advantage = carry

            # Current advantage
            new_advantage = delta + self.discount * self.gae_lambda * mask * last_advantage

            # Return the new advantage and use it for next iteration
            return (new_advantage, new_advantage), new_advantage

        # Compute TD errors: r_t + γV(s_{t+1}) - V(s_t)
        deltas = rewards + self.discount * (1.0 - dones) * values[1:] - values[:-1]

        # Masks for done states
        masks = 1.0 - dones

        # Reverse the sequences for backward calculation
        reversed_deltas = jnp.flip(deltas)
        reversed_masks = jnp.flip(masks)

        # Initial advantage
        init_carry = (jnp.array(0.0), jnp.array(0.0))

        # Compute advantages using scan
        _, advantages = jax.lax.scan(
            gae_step,
            init_carry,
            (reversed_deltas, reversed_masks)
        )

        # Flip back to original order
        advantages = jnp.flip(advantages)

        # Compute returns
        returns = advantages + values[:-1]

        # Clip advantages to prevent extreme values
        advantages = jnp.clip(advantages, -10.0, 10.0)

        return advantages, returns

    def update(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        old_log_probs: jnp.ndarray,
        advantages: jnp.ndarray,
        returns: jnp.ndarray,
        update_actor: bool = True,
    ):
        """Update agent parameters using PPO.

        Args:
            observations: Batch of observations
            actions: Batch of actions
            old_log_probs: Log probs of actions under old policy
            advantages: Advantage estimates
            returns: Return estimates
            update_actor: Whether to update actor network

        Returns:
            Updated agent and info dictionary
        """
        # Detect and handle NaN values
        if jnp.isnan(advantages).any() or jnp.isnan(returns).any():
            print("Warning: NaN detected in advantages or returns")
            advantages = jnp.nan_to_num(advantages, nan=0.0)
            returns = jnp.nan_to_num(returns, nan=0.0)

        return self._update_impl(observations, actions, old_log_probs,
                                advantages, returns, update_actor)

    @jax.jit
    def _update_impl(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        old_log_probs: jnp.ndarray,
        advantages: jnp.ndarray,
        returns: jnp.ndarray,
        update_actor: bool,
    ):
        """Jitted implementation of update."""
        # Safe advantage normalization
        advantages_mean = jax.lax.stop_gradient(jnp.mean(advantages))
        advantages_std = jax.lax.stop_gradient(jnp.std(advantages) + 1e-8)
        normalized_advantages = (advantages - advantages_mean) / advantages_std

        # Clip normalized advantages for numerical stability
        normalized_advantages = jnp.clip(normalized_advantages, -5.0, 5.0)

        def actor_loss_fn(actor_params):
            # Get action distribution
            dist = self.actor(observations, params=actor_params)

            # Calculate log probs of actions
            log_probs = dist.log_prob(actions)

            # Check for NaN in log probs and replace with old log probs
            log_probs = jnp.nan_to_num(log_probs, nan=old_log_probs)

            # Calculate ratio of new and old action probabilities: π_new(a|s) / π_old(a|s)
            # More numerically stable ratio calculation
            ratio = jnp.exp(jnp.clip(log_probs - old_log_probs, -20.0, 20.0))

            # Clipped objective: min(r * A, clip(r, 1-ε, 1+ε) * A)
            clip_low, clip_high = 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
            clipped_ratio = jnp.clip(ratio, clip_low, clip_high)

            surrogate1 = ratio * normalized_advantages
            surrogate2 = clipped_ratio * normalized_advantages
            surrogate_loss = -jnp.minimum(surrogate1, surrogate2).mean()

            # Calculate entropy bonus for exploration
            entropy = dist.entropy().mean()
            entropy_loss = -self.entropy_coef * entropy

            # Calculate KL divergence between old and new policy for early stopping
            # More stable KL calculation
            approx_kl = jnp.clip(((ratio - 1) - jnp.log(jnp.clip(ratio, 1e-5, 1e5))).mean(), 0.0, 100.0)

            # Total loss
            total_loss = surrogate_loss + entropy_loss

            return total_loss, {
                'actor_loss': surrogate_loss,
                'entropy': entropy,
                'entropy_loss': entropy_loss,
                'kl': approx_kl,
                'ratio': ratio.mean(),
                'ratio_min': ratio.min(),
                'ratio_max': ratio.max(),
            }

        def critic_loss_fn(critic_params):
            # Get value predictions
            values = self.critic(observations, params=critic_params)

            # Huber loss instead of MSE for better stability
            delta = 1.0
            diff = values - returns
            vf_loss = jnp.where(
                jnp.abs(diff) < delta,
                0.5 * diff ** 2,
                delta * (jnp.abs(diff) - 0.5 * delta)
            ).mean()

            scaled_vf_loss = self.vf_coef * vf_loss

            return scaled_vf_loss, {
                'vf_loss': vf_loss,
                'values': values.mean(),
                'returns': returns.mean(),
                'values_min': values.min(),
                'values_max': values.max(),
                'returns_min': returns.min(),
                'returns_max': returns.max(),
            }

        # Update critic
        new_critic, critic_info = self.critic.apply_loss_fn(loss_fn=critic_loss_fn, has_aux=True)

        # Update actor using jax.lax.cond to handle the conditional
        def update_actor_fn(_):
            return self.actor.apply_loss_fn(loss_fn=actor_loss_fn, has_aux=True)

        def no_update_fn(_):
            # Must match the structure of update_actor_fn return value
            empty_info = {
                'actor_loss': jnp.array(0.0),
                'entropy': jnp.array(0.0),
                'entropy_loss': jnp.array(0.0),
                'kl': jnp.array(0.0),
                'ratio': jnp.array(0.0),
                'ratio_min': jnp.array(0.0),
                'ratio_max': jnp.array(0.0)
            }
            return self.actor, empty_info

        new_actor, actor_info = jax.lax.cond(
            update_actor,
            update_actor_fn,
            no_update_fn,
            None
        )

        # Combine info
        info = {**actor_info, **critic_info}

        # Create new agent
        return self.replace(actor=new_actor, critic=new_critic), info

    @jax.jit
    def sample_actions(
        self,
        observations: jnp.ndarray,
        *,
        seed: jax.random.PRNGKey,
        temperature: float = 1.0
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample actions from the policy.

        Args:
            observations: Batch of observations
            seed: Random seed
            temperature: Temperature for exploration

        Returns:
            actions: Sampled actions
            log_probs: Log probabilities of sampled actions
            values: Value estimates
        """
        # Get action distribution and sample actions
        dist = self.actor(observations, temperature=temperature)
        actions = dist.sample(seed=seed)
        log_probs = dist.log_prob(actions)

        # Get value estimates
        values = self.critic(observations)

        # Use safer action clipping with the configurable limit
        return jnp.clip(actions, -self.action_clip, self.action_clip), log_probs, values

    @jax.jit
    def evaluate_actions(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate actions given observations.

        Args:
            observations: Batch of observations
            actions: Batch of actions

        Returns:
            log_probs: Log probabilities of actions
            values: Value estimates
        """
        # Get action distribution
        dist = self.actor(observations)
        log_probs = dist.log_prob(actions)

        # Get value estimates
        values = self.critic(observations)

        return log_probs, values

    @jax.jit
    def get_values(self, observations: jnp.ndarray) -> jnp.ndarray:
        """Get value estimates for observations.

        Args:
            observations: Batch of observations

        Returns:
            values: Value estimates
        """
        return self.critic(observations)

    @jax.jit
    def get_actions(
        self,
        observations: jnp.ndarray,
        temperature: float = 1.0
    ) -> jnp.ndarray:
        """Get deterministic actions (distribution mode).

        Args:
            observations: Batch of observations
            temperature: Temperature for exploration

        Returns:
            actions: Deterministic actions
        """
        dist = self.actor(observations, temperature=temperature)
        actions = dist.mode()
        # Use safer action clipping with the configurable limit
        return jnp.clip(actions, -self.action_clip, self.action_clip)

    def save(self, path):
        """Save model parameters."""
        with open(path, "wb") as f:
            params_dict = {
                'actor_params': self.actor.params,
                'actor_def': self.actor.model_def,
                'critic_params': self.critic.params,
                'critic_def': self.critic.model_def,
                'discount': self.discount,
                'gae_lambda': self.gae_lambda,
                'clip_ratio': self.clip_ratio,
                'target_kl': self.target_kl,
                'entropy_coef': self.entropy_coef,
                'vf_coef': self.vf_coef,
                'action_clip': self.action_clip,
            }
            pickle.dump(params_dict, f)

    @classmethod
    def from_bc_agent(cls, bc_agent, observation_dim, action_dim, critic_lr=1e-3, actor_lr=3e-4, **kwargs):
        """Initialize a PPO agent from a trained BC agent.

        Args:
            bc_agent: Trained BC agent
            observation_dim: Dimension of observation space
            action_dim: Dimension of action space
            critic_lr: Learning rate for critic
            actor_lr: Learning rate for actor fine-tuning
            **kwargs: Additional arguments for PPO agent

        Returns:
            A new PPO agent with actor initialized from BC
        """
        # Create a new PPO agent
        ppo_agent = cls.create(
            seed=0,  # Doesn't matter since we'll override params
            observation_dim=observation_dim,
            action_dim=action_dim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            **kwargs
        )

        # Replace actor with BC agent's actor
        new_actor = ppo_agent.actor.replace(
            params=bc_agent.actor.params,
            model_def=bc_agent.actor.model_def
        )

        # Return PPO agent with BC-initialized actor
        return ppo_agent.replace(actor=new_actor)

    @classmethod
    def load(cls, path, observation_dim, action_dim, actor_lr=3e-4, critic_lr=1e-3):
        """Load model parameters."""
        with open(path, "rb") as f:
            params_dict = pickle.load(f)

        # Create a new agent with the loaded parameters
        agent = cls.create(
            seed=0,  # Doesn't matter for loading
            observation_dim=observation_dim,
            action_dim=action_dim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            discount=params_dict.get('discount', 0.99),
            gae_lambda=params_dict.get('gae_lambda', 0.95),
            clip_ratio=params_dict.get('clip_ratio', 0.2),
            target_kl=params_dict.get('target_kl', 0.01),
            entropy_coef=params_dict.get('entropy_coef', 0.01),
            vf_coef=params_dict.get('vf_coef', 0.5),
            action_clip=params_dict.get('action_clip', 0.9),
        )

        # Replace actor parameters
        new_actor = agent.actor.replace(
            params=params_dict['actor_params'],
            model_def=params_dict['actor_def']
        )

        # Replace critic parameters
        new_critic = agent.critic.replace(
            params=params_dict['critic_params'],
            model_def=params_dict['critic_def']
        )

        return agent.replace(actor=new_actor, critic=new_critic)