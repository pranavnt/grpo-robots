"""Miscellaneous utility functions."""
import os
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
from typing import Dict, Tuple, Optional, List, Any, Union, Callable

def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass
    
    return jax.random.PRNGKey(seed)

def create_log_dir(base_dir: str, experiment_name: str) -> str:
    """Create a directory for logging experiment results."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def save_config(config: Dict[str, Any], log_dir: str):
    """Save experiment configuration to a file."""
    config_path = os.path.join(log_dir, "config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(config, f)
    
    # Also save as text for easy inspection
    config_txt_path = os.path.join(log_dir, "config.txt")
    with open(config_txt_path, "w") as f:
        for k, v in config.items():
            f.write(f"{k}: {v}\n")

def get_metaworld_env(env_name: str, seed: Optional[int] = None):
    """Create a MetaWorld environment."""
    try:
        from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
        
        mw_env_name = f"{env_name}-goal-observable"
        env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[mw_env_name]()
        
        if seed is not None:
            env.seed(seed)
            
        return env
    except ImportError:
        raise ImportError("MetaWorld not installed. Please install it with: pip install git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld")

def linear_schedule(start_value: float, end_value: float, duration: int):
    """Create a linear schedule function."""
    def schedule(step: int) -> float:
        progress = min(step / duration, 1.0)
        return start_value + progress * (end_value - start_value)
    return schedule

def cosine_schedule(start_value: float, end_value: float, duration: int):
    """Create a cosine annealing schedule function."""
    def schedule(step: int) -> float:
        progress = min(step / duration, 1.0)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        return end_value + (start_value - end_value) * cosine_decay
    return schedule

def wandb_init(project: str, config: Dict[str, Any], name: Optional[str] = None):
    """Initialize Weights & Biases for experiment tracking."""
    try:
        import wandb
        wandb.init(project=project, config=config, name=name)
        return wandb
    except ImportError:
        print("Weights & Biases not installed. Run 'pip install wandb' to use it.")
        return None

def log_metrics(metrics: Dict[str, Any], step: int, wandb_instance=None):
    """Log metrics to console and optionally to wandb."""
    # Print to console
    print(f"Step {step}:")
    for k, v in metrics.items():
        if isinstance(v, (float, int)):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # Log to wandb if available
    if wandb_instance:
        wandb_instance.log(metrics, step=step)

class Timer:
    """Simple timer for measuring execution time."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """Stop the timer."""
        self.end_time = time.time()
        return self
    
    def elapsed(self):
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time
    
    def __enter__(self):
        """Support for 'with' statement."""
        self.start()
        return self
    
    def __exit__(self, *args):
        """Support for 'with' statement."""
        self.stop()

def save_model(model, path: str):
    """Save a model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model(path: str):
    """Load a model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
