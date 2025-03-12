"""Training state utilities."""
from typing import Any, Callable, Optional, Dict
import flax
import flax.linen as nn
import jax
import optax
import functools

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)

def target_update(
    model: "TrainState", target_model: "TrainState", tau: float
) -> "TrainState":
    """Perform soft target update."""
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), model.params, target_model.params
    )
    return target_model.replace(params=new_target_params)


class TrainState(flax.struct.PyTreeNode):
    """Enhanced TrainState for managing model parameters and optimization state."""
    step: int
    apply_fn: Callable[..., Any] = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Dict
    tx: Optional[optax.GradientTransformation] = nonpytree_field()
    opt_state: Optional[optax.OptState] = None
    extra_variables: Optional[Dict] = None

    @classmethod
    def create(
        cls,
        model_def: nn.Module,
        params: Dict,
        tx: Optional[optax.GradientTransformation] = None,
        extra_variables: Optional[dict] = None,
        **kwargs,
    ) -> "TrainState":
        """Create a new TrainState instance."""
        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        if extra_variables is None:
            extra_variables = flax.core.FrozenDict()

        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            extra_variables=extra_variables,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def __call__(
        self,
        *args,
        params=None,
        extra_variables=None,
        method=None,
        **kwargs,
    ):
        """Call the model with specified parameters and method."""
        if params is None:
            params = self.params

        if extra_variables is None:
            extra_variables = self.extra_variables
        variables = {"params": params, **self.extra_variables}

        if isinstance(method, str):
            method = getattr(self.model_def, method)

        return self.apply_fn(variables, *args, method=method, **kwargs)

    def apply_gradients(self, *, grads, **kwargs):
        """Apply gradients to update model parameters."""
        if self.tx is None:
            raise ValueError("No optimizer (tx) provided in TrainState.")

        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    def apply_loss_fn(self, *, loss_fn, pmap_axis=None, has_aux=False):
        """Apply a loss function and compute gradients."""
        if has_aux:
            grads, info = jax.grad(loss_fn, has_aux=has_aux)(self.params)
            if pmap_axis is not None:
                grads = jax.lax.pmean(grads, axis_name=pmap_axis)
                info = jax.lax.pmean(info, axis_name=pmap_axis)
            return self.apply_gradients(grads=grads), info

        else:
            grads = jax.grad(loss_fn)(self.params)
            if pmap_axis is not None:
                grads = jax.lax.pmean(grads, axis_name=pmap_axis)
            return self.apply_gradients(grads=grads)

    def __getattr__(self, name):
        """Provide syntax sugar for calling methods of the model_def directly."""
        method = getattr(self.model_def, name)
        return functools.partial(self.__call__, method=method)
