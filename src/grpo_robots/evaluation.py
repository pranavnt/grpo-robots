"""Evaluation utilities for RL agents."""
from typing import Dict, Callable, Optional, List, Any
import jax
import numpy as np
import gymnasium as gym
import time
from collections import defaultdict

def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """
    Wrapper that supplies a jax random key to a function (using keyword `seed`).
    """
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)
    return wrapped

def flatten(d, parent_key="", sep="."):
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def add_to(dict_of_lists, single_dict):
    """Add values from a dict to corresponding lists in a dict of lists."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)

def evaluate(policy_fn, env: gym.Env, num_episodes: int, save_video: bool = False) -> Dict[str, float]:
    """
    Evaluate a policy in an environment for a number of episodes.
    
    Args:
        policy_fn: Function that takes observation and returns action
        env: Gym environment
        num_episodes: Number of episodes to evaluate
        save_video: Whether to save video of episodes
        
    Returns:
        Dictionary of evaluation statistics
    """
    stats = defaultdict(list)
    
    if save_video:
        try:
            from gym.wrappers.monitoring.video_recorder import VideoRecorder
            video_recorder = VideoRecorder(env, base_path=f"./videos/eval_{time.time()}")
        except ImportError:
            print("Warning: video recording requires gym.wrappers.monitoring.video_recorder")
            save_video = False
    
    for _ in range(num_episodes):
        observation, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = policy_fn(observation)
            
            if save_video:
                video_recorder.capture_frame()
                
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated or info.get('success', False)
            add_to(stats, flatten(info))
        
        # Add episode statistics
        add_to(stats, {'episode_reward': episode_reward})
        add_to(stats, flatten(info, parent_key="final"))
    
    if save_video:
        video_recorder.close()
    
    # Compute means of collected statistics
    for k, v in stats.items():
        stats[k] = np.mean(v)
        
    return stats

class EpisodeMonitor(gym.ActionWrapper):
    """A wrapper that computes episode returns and lengths."""
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0
        
    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()
        
    def step(self, action: np.ndarray):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}
        
        if done:
            info["episode"] = {}
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time
            
        return observation, reward, terminated, truncated, info
        
    def reset(self, **kwargs):
        self._reset_stats()
        return self.env.reset(**kwargs)
