"""Common neural network architectures for RL."""
from typing import Callable, Dict, Optional, Sequence, Tuple, Union
import jax
from jax.random import PRNGKey
import jax.numpy as jnp
import flax.linen as nn
import distrax
from typing import Any, Sequence
from grpo_robots.grpo_types import Shape, Dtype, Array

def default_init(scale: Optional[float] = 1.0):
    """Default initialization for layers."""
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")

class MLP(nn.Module):
    """Multi-layer perceptron module."""
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
        return x

class TransformedWithMode(distrax.Transformed):
    """Distribution transformation that supports mode calculation and entropy approximation."""
    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())
        
    def entropy(self) -> jnp.ndarray:
        """Approximate entropy for the transformed distribution.
        
        For tanh-transformed normal, we just use the base distribution entropy
        as a proxy. This is a common approximation in RL.
        """
        return self.distribution.entropy()

class Policy(nn.Module):
    """Gaussian policy network."""
    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    tanh_squash_distribution: bool = True
    state_dependent_std: bool = True
    final_fc_init_scale: float = 1e-2

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, temperature: float = 1.0
    ) -> distrax.Distribution:
        outputs = MLP(
            self.hidden_dims,
            activate_final=True,
        )(observations)

        means = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )(outputs)

        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
            )(outputs)
        else:
            log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )

        if self.tanh_squash_distribution:
            distribution = TransformedWithMode(
                distribution, distrax.Block(distrax.Tanh(), ndims=1)
            )

        return distribution

class ValueCritic(nn.Module):
    """Value function network."""
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        critic = MLP((*self.hidden_dims, 1))(observations)
        return jnp.squeeze(critic, -1)

class Critic(nn.Module):
    """Q-function network."""
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, 1), activations=self.activations, use_layer_norm=self.use_layer_norm)(inputs)
        return jnp.squeeze(critic, -1)

def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    """Create an ensemble of Q-functions."""
    split_rngs = kwargs.pop("split_rngs", {})
    return nn.vmap(
        cls,
        variable_axes={"params": 0},
        split_rngs={**split_rngs, "params": True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs
    )
