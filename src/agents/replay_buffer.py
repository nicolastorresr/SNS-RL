import numpy as np
from collections import deque
from typing import Tuple, List

class ReplayBuffer:
    def __init__(self, capacity: int):
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience: Tuple[np.ndarray, int, float, np.ndarray, bool]):
        """Add experience to buffer.
        
        Args:
            experience: Tuple of (state, action, reward, next_state, done)
        """
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        indices = np.random.choice(
            len(self.buffer),
            size=batch_size,
            replace=False
        )
        
        states, actions, rewards, next_states, dones = zip(
            *[self.buffer[idx] for idx in indices]
        )
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self) -> int:
        """Get current size of buffer."""
        return len(self.buffer)
