import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict
from .replay_buffer import ReplayBuffer
from .policy_utils import epsilon_greedy_policy

class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 10000,
        batch_size: int = 64,
        buffer_size: int = 100000,
        update_target_every: int = 1000
    ):
        """Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Starting exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay_steps: Steps over which to decay epsilon
            batch_size: Size of training batches
            buffer_size: Size of replay buffer
            update_target_every: Steps between target network updates
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.train_step_counter = 0
        
        # Initialize networks
        self.main_network = self._build_network()
        self.target_network = self._build_network()
        self.target_network.set_weights(self.main_network.get_weights())
        
        # Initialize optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Initialize epsilon decay
        self.epsilon_decay = tf.keras.optimizers.schedules.PolynomialDecay(
            epsilon_start,
            epsilon_decay_steps,
            epsilon_end
        )
        
    def _build_network(self) -> tf.keras.Model:
        """Build neural network for Q-function approximation."""
        inputs = tf.keras.layers.Input(shape=(self.state_dim,))
        
        # Hidden layers
        x = tf.keras.layers.Dense(256, activation='relu')(inputs)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(self.action_dim)(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        epsilon = self.epsilon_decay(self.train_step_counter) if training else 0.01
        return epsilon_greedy_policy(
            state,
            self.main_network,
            self.action_dim,
            epsilon
        )
    
    def train(self, experience: Tuple[np.ndarray, int, float, np.ndarray, bool]):
        """Store experience and train the agent if enough samples are available."""
        # Store experience in replay buffer
        self.replay_buffer.add(experience)
        
        # Only train if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return
            
        # Sample batch and train
        self._train_step()
        
        # Update target network if needed
        if self.train_step_counter % self.update_target_every == 0:
            self.target_network.set_weights(self.main_network.get_weights())
            
        self.train_step_counter += 1
    
    @tf.function
    def _train_step(self):
        """Execute one training step."""
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            # Calculate target Q-values
            next_q_values = self.target_network(next_states)
            max_next_q_values = tf.reduce_max(next_q_values, axis=1)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
            
            # Calculate current Q-values
            q_values = self.main_network(states)
            action_masks = tf.one_hot(actions, self.action_dim)
            current_q_values = tf.reduce_sum(q_values * action_masks, axis=1)
            
            # Calculate loss
            loss = tf.keras.losses.MSE(target_q_values, current_q_values)
            
        # Update weights
        gradients = tape.gradient(loss, self.main_network.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.main_network.trainable_variables)
        )
        
        return loss
    
    def save_weights(self, filepath: str):
        """Save network weights."""
        self.main_network.save_weights(filepath)
        
    def load_weights(self, filepath: str):
        """Load network weights."""
        self.main_network.load_weights(filepath)
        self.target_network.set_weights(self.main_network.get_weights())
