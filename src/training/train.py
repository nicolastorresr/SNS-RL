import numpy as np
from typing import Dict, List, Optional
import tensorflow as tf
from ..environment.social_network_env import SocialNetworkEnv
from ..agents.dqn_agent import DQNAgent
from .logger import TrainingLogger
from .evaluation import evaluate_network

def train_network(
    env: SocialNetworkEnv,
    agent: DQNAgent,
    num_episodes: int = 1000,
    max_steps_per_episode: int = 100,
    eval_frequency: int = 50,
    logger: Optional[TrainingLogger] = None,
) -> Dict:
    """
    Main training loop for the social network simulation.
    
    Args:
        env: The social network environment
        agent: The DQN agent
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        eval_frequency: How often to run evaluation
        logger: Training logger instance
    
    Returns:
        Dict containing training statistics
    """
    if logger is None:
        logger = TrainingLogger()
    
    best_reward = float('-inf')
    training_stats = {
        'episode_rewards': [],
        'network_metrics': [],
        'loss_history': []
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss = []
        
        for step in range(max_steps_per_episode):
            # Select action using epsilon-greedy policy
            action = agent.select_action(state)
            
            # Take action and observe next state and reward
            next_state, reward, done, info = env.step(action)
            
            # Store experience in replay buffer
            agent.store_experience(state, action, reward, next_state, done)
            
            # Train the agent
            if len(agent.replay_buffer) >= agent.batch_size:
                loss = agent.train_step()
                episode_loss.append(loss)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Log episode statistics
        training_stats['episode_rewards'].append(episode_reward)
        if episode_loss:
            training_stats['loss_history'].append(np.mean(episode_loss))
        
        # Periodic evaluation
        if (episode + 1) % eval_frequency == 0:
            eval_metrics = evaluate_network(env, agent)
            training_stats['network_metrics'].append(eval_metrics)
            
            # Log progress
            logger.log_episode(episode + 1, {
                'reward': episode_reward,
                'loss': np.mean(episode_loss) if episode_loss else None,
                'eval_metrics': eval_metrics
            })
            
            # Save best model
            if eval_metrics['average_reward'] > best_reward:
                best_reward = eval_metrics['average_reward']
                agent.save_model(f"best_model_episode_{episode+1}")
    
    return training_stats
