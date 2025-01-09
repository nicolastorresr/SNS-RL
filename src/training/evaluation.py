from typing import Dict
import numpy as np
from ..environment.social_network_env import SocialNetworkEnv
from ..agents.dqn_agent import DQNAgent
from ..network.metrics import compute_network_metrics

def evaluate_network(
    env: SocialNetworkEnv,
    agent: DQNAgent,
    num_episodes: int = 5,
    max_steps_per_episode: int = 100
) -> Dict:
    """
    Evaluate the current policy in the environment.
    
    Args:
        env: The social network environment
        agent: The trained DQN agent
        num_episodes: Number of evaluation episodes
        max_steps_per_episode: Maximum steps per evaluation episode
    
    Returns:
        Dict containing evaluation metrics
    """
    rewards = []
    network_metrics_list = []
    
    # Disable exploration during evaluation
    original_epsilon = agent.epsilon
    agent.epsilon = 0
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for _ in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        rewards.append(episode_reward)
        
        # Compute network metrics at the end of each episode
        network_metrics = compute_network_metrics(env.network)
        network_metrics_list.append(network_metrics)
    
    # Restore original exploration rate
    agent.epsilon = original_epsilon
    
    # Aggregate metrics
    eval_metrics = {
        'average_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'network_metrics': {
            key: np.mean([m[key] for m in network_metrics_list])
            for key in network_metrics_list[0].keys()
        }
    }
    
    return eval_metrics
