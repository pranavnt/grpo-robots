"""Behavioral Cloning (BC) implementation."""
import jax
import jax.numpy as jnp
import flax
import optax
from typing import Tuple, Dict, Any, Optional
import pickle

from grpo_robots.networks import Policy, MLP
from grpo_robots.train_state import TrainState

class BCAgent(flax.struct.PyTreeNode):
    """Behavioral Cloning agent."""
    actor: TrainState

    @classmethod
    def create(
        cls,
        seed: int,
        observation_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        hidden_dims: Tuple[int, ...] = (256, 256),
        state_dependent_std: bool = False,
    ):
        """Create a new BC agent.

        Args:
            seed: Random seed
            observation_dim: Dimension of observation space
            action_dim: Dimension of action space
            learning_rate: Learning rate for optimization
            hidden_dims: Hidden dimensions for the policy network
            state_dependent_std: Whether std is state-dependent or learned directly

        Returns:
            A new BCAgent
        """
        rng = jax.random.PRNGKey(seed)

        # Configure actor network
        actor_def = Policy(
            hidden_dims=hidden_dims,
            action_dim=action_dim,
            log_std_min=-10.0,
            log_std_max=0.0,  # More conservative upper bound
            state_dependent_std=state_dependent_std,
            tanh_squash_distribution=True,
            final_fc_init_scale=0.01
        )

        # Initialize actor
        dummy_obs = jnp.zeros((1, observation_dim))
        actor_params = actor_def.init(rng, dummy_obs)['params']

        # Use gradient clipping for stability
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=learning_rate, eps=1e-5)
        )

        actor = TrainState.create(actor_def, actor_params, tx=tx)

        return cls(actor=actor)

    @jax.jit
    def update(self, batch):
        """Update the agent using BC loss."""
        def loss_fn(actor_params):
            dist = self.actor(batch['observations'], params=actor_params)

            # MSE loss is more stable than negative log likelihood for BC
            actions_mean = dist.mode()
            mse_loss = ((actions_mean - batch['actions']) ** 2).sum(-1).mean()

            return mse_loss, {
                'actor_loss': mse_loss,
                'mse_loss': mse_loss,
            }

        new_actor, info = self.actor.apply_loss_fn(loss_fn=loss_fn, has_aux=True)
        return self.replace(actor=new_actor), info

    @jax.jit
    def sample_actions(self,
                      observations: jnp.ndarray,
                      *,
                      seed: jax.random.PRNGKey,
                      temperature: float = 1.0) -> jnp.ndarray:
        """Sample actions from the policy distribution."""
        dist = self.actor(observations, temperature=temperature)
        actions = dist.sample(seed=seed)
        actions = jnp.clip(actions, -0.9, 0.9)  # Use slightly more conservative clipping
        return actions

    @jax.jit
    def get_actions(self,
                   observations: jnp.ndarray,
                   temperature: float = 1.0) -> jnp.ndarray:
        """Get deterministic actions (distribution mode)."""
        dist = self.actor(observations, temperature=temperature)
        actions = dist.mode()
        actions = jnp.clip(actions, -0.9, 0.9)  # Use slightly more conservative clipping
        return actions

    def save(self, path):
        """Save model parameters."""
        with open(path, "wb") as f:
            params_dict = {
                'actor_params': self.actor.params,
                'actor_def': self.actor.model_def
            }
            pickle.dump(params_dict, f)

    @classmethod
    def load(cls, path, observation_dim, action_dim, learning_rate=3e-4):
        """Load model parameters."""
        with open(path, "rb") as f:
            params_dict = pickle.load(f)

        # Create a new agent with the loaded parameters
        agent = cls.create(
            seed=0,  # Doesn't matter for loading
            observation_dim=observation_dim,
            action_dim=action_dim,
            learning_rate=learning_rate
        )

        # Replace actor parameters
        new_actor = agent.actor.replace(
            params=params_dict['actor_params'],
            model_def=params_dict['actor_def']
        )

        return agent.replace(actor=new_actor)