#!/usr/bin/env python3
"""Group Relative Policy Optimization (GRPO) implementation.

This agent is similar to PPO but removes the critic and computes the advantage
for policy updates via group–normalization of rewards.
"""

import os
import pickle
from typing import Tuple, Dict, Any, Tuple, Optional
import jax
import jax.numpy as jnp
import flax
import optax

from grpo_robots.networks import Policy
from grpo_robots.train_state import TrainState


class GRPOAgent(flax.struct.PyTreeNode):
    """Group Relative Policy Optimization (GRPO) agent.

    The agent only maintains an actor (policy) network. Its update step uses a
    clipped surrogate loss similar to PPO but the advantage is provided externally
    (typically computed as group-normalized returns over complete trajectories).
    """
    actor: TrainState
    clip_ratio: float = 0.2
    target_kl: float = 0.01
    entropy_coef: float = 0.01
    action_clip: float = 0.9

    @classmethod
    def create(
        cls,
        seed: int,
        observation_dim: int,
        action_dim: int,
        actor_lr: float = 3e-4,
        hidden_dims: Tuple[int, ...] = (256, 256),
        clip_ratio: float = 0.2,
        target_kl: float = 0.01,
        entropy_coef: float = 0.01,
        state_dependent_std: bool = True,
        action_clip: float = 0.9,
    ):
        rng = jax.random.PRNGKey(seed)
        # Configure actor network (policy)
        actor_def = Policy(
            hidden_dims=hidden_dims,
            action_dim=action_dim,
            log_std_min=-10.0,
            log_std_max=0.0,
            state_dependent_std=state_dependent_std,
            tanh_squash_distribution=True,
            final_fc_init_scale=0.01
        )
        dummy_obs = jnp.zeros((1, observation_dim))
        actor_params = actor_def.init(rng, dummy_obs)['params']

        actor_tx = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(learning_rate=actor_lr, eps=1e-5)
        )

        actor = TrainState.create(actor_def, actor_params, tx=actor_tx)

        return cls(
            actor=actor,
            clip_ratio=clip_ratio,
            target_kl=target_kl,
            entropy_coef=entropy_coef,
            action_clip=action_clip
        )

    @jax.jit
    def update(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        old_log_probs: jnp.ndarray,
        advantages: jnp.ndarray,
    ):
        """Update the agent using the GRPO objective.

        Args:
            observations: Batch of observations.
            actions: Batch of actions.
            old_log_probs: Log probabilities under the old policy.
            advantages: Advantage estimates (group–normalized).

        Returns:
            A tuple of the updated agent and an info dict.
        """
        # (Optionally re–normalize advantages; here we clip them for numerical stability.)
        advantages_mean = jax.lax.stop_gradient(jnp.mean(advantages))
        advantages_std = jax.lax.stop_gradient(jnp.std(advantages) + 1e-8)
        normalized_advantages = (advantages - advantages_mean) / advantages_std
        normalized_advantages = jnp.clip(normalized_advantages, -5.0, 5.0)

        def actor_loss_fn(actor_params):
            dist = self.actor(observations, params=actor_params)
            log_probs = dist.log_prob(actions)
            # Replace any NaN log–probs with the old values
            log_probs = jnp.nan_to_num(log_probs, nan=old_log_probs)
            ratio = jnp.exp(jnp.clip(log_probs - old_log_probs, -20.0, 20.0))
            clip_low, clip_high = 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
            clipped_ratio = jnp.clip(ratio, clip_low, clip_high)
            surrogate1 = ratio * normalized_advantages
            surrogate2 = clipped_ratio * normalized_advantages
            surrogate_loss = -jnp.minimum(surrogate1, surrogate2).mean()

            entropy = dist.entropy().mean()
            entropy_loss = -self.entropy_coef * entropy

            # Approximate KL divergence (for monitoring)
            approx_kl = jnp.clip(
                ((ratio - 1) - jnp.log(jnp.clip(ratio, 1e-5, 1e5))).mean(), 0.0, 100.0
            )

            total_loss = surrogate_loss + entropy_loss
            return total_loss, {
                'actor_loss': surrogate_loss,
                'entropy': entropy,
                'entropy_loss': entropy_loss,
                'kl': approx_kl,
                'ratio_mean': ratio.mean(),
                'ratio_min': ratio.min(),
                'ratio_max': ratio.max(),
            }

        new_actor, info = self.actor.apply_loss_fn(loss_fn=actor_loss_fn, has_aux=True)
        return self.replace(actor=new_actor), info

    @jax.jit
    def sample_actions(
        self,
        observations: jnp.ndarray,
        *,
        seed: jax.random.PRNGKey,
        temperature: float = 1.0
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sample actions from the policy.

        Args:
            observations: Batch of observations.
            seed: PRNG key.
            temperature: Exploration temperature.

        Returns:
            A tuple of (actions, log_probs).
        """
        dist = self.actor(observations, temperature=temperature)
        actions = dist.sample(seed=seed)
        log_probs = dist.log_prob(actions)
        return jnp.clip(actions, -self.action_clip, self.action_clip), log_probs

    @jax.jit
    def get_actions(
        self,
        observations: jnp.ndarray,
        temperature: float = 1.0
    ) -> jnp.ndarray:
        """Get deterministic actions (mode of the distribution)."""
        dist = self.actor(observations, temperature=temperature)
        actions = dist.mode()
        return jnp.clip(actions, -self.action_clip, self.action_clip)

    def save(self, path: str):
        """Save the GRPO agent to disk."""
        with open(path, "wb") as f:
            params_dict = {
                'actor_params': self.actor.params,
                'actor_def': self.actor.model_def,
                'clip_ratio': self.clip_ratio,
                'target_kl': self.target_kl,
                'entropy_coef': self.entropy_coef,
                'action_clip': self.action_clip,
            }
            pickle.dump(params_dict, f)

    @classmethod
    def from_bc_agent(cls, bc_agent, observation_dim, action_dim, actor_lr=3e-4, **kwargs):
        """Initialize a GRPO agent from a BC agent.

        The actor is initialized from the BC agent.
        """
        grpo_agent = cls.create(
            seed=0,  # seed is irrelevant since we override the parameters
            observation_dim=observation_dim,
            action_dim=action_dim,
            actor_lr=actor_lr,
            **kwargs
        )
        new_actor = grpo_agent.actor.replace(
            params=bc_agent.actor.params,
            model_def=bc_agent.actor.model_def
        )
        return grpo_agent.replace(actor=new_actor)

    @classmethod
    def load(cls, path: str, observation_dim, action_dim, actor_lr=3e-4):
        """Load a GRPO agent from disk."""
        with open(path, "rb") as f:
            params_dict = pickle.load(f)
        agent = cls.create(
            seed=0,
            observation_dim=observation_dim,
            action_dim=action_dim,
            actor_lr=actor_lr,
            clip_ratio=params_dict.get('clip_ratio', 0.2),
            target_kl=params_dict.get('target_kl', 0.01),
            entropy_coef=params_dict.get('entropy_coef', 0.01),
            action_clip=params_dict.get('action_clip', 0.9)
        )
        new_actor = agent.actor.replace(
            params=params_dict['actor_params'],
            model_def=params_dict['actor_def']
        )
        return agent.replace(actor=new_actor)
