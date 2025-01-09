import numpy as np
import tensorflow as tf
from typing import Tuple

def epsilon_greedy_policy(
    state: np.ndarray,
    network: tf.keras.Model,
    action_dim: int,
    epsilon: float
) -> int:
    """Implement epsilon-greedy policy for action selection.
    
    Args:
        state: Current state
        network: Q-network
        action_dim: Number of possible actions
        epsilon: Exploration rate
        
    Returns:
        Selected action index
    """
    # With probability epsilon, choose random action
    if np.random.random() < epsilon:
        return np.random.randint(action_dim)
    
    # Otherwise, choose greedy action
    state_tensor = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
    q_values = network(state_tensor)[0]
    return tf.argmax(q_values).numpy()

def boltzmann_policy(
    state: np.ndarray,
    network: tf.keras.Model,
    temperature: float = 1.0
) -> Tuple[int, np.ndarray]:
    """Implement Boltzmann (softmax) policy for action selection.
    
    Args:
        state: Current state
        network: Q-network
        temperature: Temperature parameter for softmax
        
    Returns:
        Tuple of (selected action index, action probabilities)
    """
    # Get Q-values
    state_tensor = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
    q_values = network(state_tensor)[0].numpy()
    
    # Apply softmax with temperature
    scaled_q_values = q_values / temperature
    exp_q_values = np.exp(scaled_q_values - np.max(scaled_q_values))
    probabilities = exp_q_values / np.sum(exp_q_values)
    
    # Sample action from probability distribution
    action = np.random.choice(len(probabilities), p=probabilities)
    
    return action, probabilities

def ucb_policy(
    state: np.ndarray,
    network: tf.keras.Model,
    action_counts: np.ndarray,
    total_steps: int,
    c: float = 2.0
) -> int:
    """Implement Upper Confidence Bound (UCB) policy for action selection.
    
    Args:
        state: Current state
        network: Q-network
        action_counts: Array counting how often each action was chosen
        total_steps: Total number of steps taken
        c: Exploration parameter
        
    Returns:
        Selected action index
    """
    state_tensor = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
    q_values = network(state_tensor)[0].numpy()
    
    # Calculate UCB scores
    uncertainty = c * np.sqrt(np.log(total_steps) / (action_counts + 1e-6))
    ucb_scores = q_values + uncertainty
    
    return np.argmax(ucb_scores)
