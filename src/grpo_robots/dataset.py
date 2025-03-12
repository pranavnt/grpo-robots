"""Replay buffer for RL agents."""
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
import jax
import jax.numpy as jnp

class ReplayBuffer:
    """Simple numpy-based replay buffer for RL agents."""
    
    def __init__(
        self, 
        observation_dim: int, 
        action_dim: int, 
        capacity: int = 100000,
    ):
        """Initialize a replay buffer.
        
        Args:
            observation_dim: Dimension of observations
            action_dim: Dimension of actions
            capacity: Buffer capacity
        """
        self.capacity = capacity
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        # Initialize buffer storage
        self.observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.next_observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.size = 0
        self.insert_index = 0
        
    def add(
        self, 
        observation: np.ndarray, 
        action: np.ndarray, 
        reward: float, 
        next_observation: np.ndarray, 
        done: bool,
    ):
        """Add a transition to the buffer."""
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.next_observations[self.insert_index] = next_observation
        self.dones[self.insert_index] = done
        
        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def add_batch(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
    ):
        """Add a batch of transitions to the buffer."""
        batch_size = observations.shape[0]
        
        if batch_size > self.capacity:
            batch_size = self.capacity
            # Only keep most recent transitions
            observations = observations[-batch_size:]
            actions = actions[-batch_size:]
            rewards = rewards[-batch_size:]
            next_observations = next_observations[-batch_size:]
            dones = dones[-batch_size:]
        
        # If batch would exceed capacity, reset insert_index to 0
        if self.insert_index + batch_size > self.capacity:
            # Add as many transitions as possible at the end
            space_left = self.capacity - self.insert_index
            self.observations[self.insert_index:] = observations[:space_left]
            self.actions[self.insert_index:] = actions[:space_left]
            self.rewards[self.insert_index:] = rewards[:space_left]
            self.next_observations[self.insert_index:] = next_observations[:space_left]
            self.dones[self.insert_index:] = dones[:space_left]
            
            # Add remaining transitions at the beginning
            remaining = batch_size - space_left
            self.observations[:remaining] = observations[space_left:]
            self.actions[:remaining] = actions[space_left:]
            self.rewards[:remaining] = rewards[space_left:]
            self.next_observations[:remaining] = next_observations[space_left:]
            self.dones[:remaining] = dones[space_left:]
            
            self.insert_index = remaining
        else:
            # Add all transitions sequentially
            end_idx = self.insert_index + batch_size
            self.observations[self.insert_index:end_idx] = observations
            self.actions[self.insert_index:end_idx] = actions
            self.rewards[self.insert_index:end_idx] = rewards
            self.next_observations[self.insert_index:end_idx] = next_observations
            self.dones[self.insert_index:end_idx] = dones
            
            self.insert_index = end_idx % self.capacity
        
        self.size = min(self.size + batch_size, self.capacity)
        
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Dictionary of sampled batch
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            'observations': self.observations[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_observations': self.next_observations[indices],
            'dones': self.dones[indices],
        }
        
    def get_all(self) -> Dict[str, np.ndarray]:
        """Get all transitions in the buffer."""
        return {
            'observations': self.observations[:self.size],
            'actions': self.actions[:self.size],
            'rewards': self.rewards[:self.size],
            'next_observations': self.next_observations[:self.size],
            'dones': self.dones[:self.size],
        }
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for a batch."""
        return self.size >= batch_size
