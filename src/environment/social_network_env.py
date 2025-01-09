import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from .state_representation import StateRepresentation
from .rewards import RewardFunction

class SocialNetworkEnvironment:
    def __init__(
        self,
        num_agents: int,
        initial_connections: int,
        max_connections: int,
        personality_dim: int = 5,
        content_topics: int = 10
    ):
        """Initialize the social network environment.
        
        Args:
            num_agents: Number of agents in the network
            initial_connections: Initial number of connections per agent
            max_connections: Maximum number of connections an agent can have
            personality_dim: Dimension of personality vectors
            content_topics: Number of different content topics
        """
        self.num_agents = num_agents
        self.max_connections = max_connections
        self.personality_dim = personality_dim
        self.content_topics = content_topics
        
        # Initialize network structure
        self.network = nx.DiGraph()
        self._initialize_network(initial_connections)
        
        # Initialize state representation
        self.state_representation = StateRepresentation(
            personality_dim=personality_dim,
            content_topics=content_topics
        )
        
        # Initialize reward function
        self.reward_function = RewardFunction()
        
        # Track agent states and metrics
        self.agent_states = {}
        self._initialize_agent_states()
        
        # Message history
        self.message_history = []
        
    def _initialize_network(self, initial_connections: int):
        """Initialize the network structure with random connections."""
        # Add nodes
        for i in range(self.num_agents):
            self.network.add_node(i)
        
        # Add random initial connections
        for i in range(self.num_agents):
            possible_connections = list(range(self.num_agents))
            possible_connections.remove(i)
            num_connections = min(initial_connections, len(possible_connections))
            connections = np.random.choice(
                possible_connections,
                size=num_connections,
                replace=False
            )
            for j in connections:
                self.network.add_edge(i, j)
    
    def _initialize_agent_states(self):
        """Initialize agent states with random personality traits and preferences."""
        for i in range(self.num_agents):
            self.agent_states[i] = {
                'personality': np.random.normal(0, 1, self.personality_dim),
                'influence_score': 1.0,
                'content_preferences': np.random.dirichlet(np.ones(self.content_topics)),
                'recent_interactions': []
            }
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state.
        
        Returns:
            Initial state representation for agent 0
        """
        self._initialize_network(initial_connections=2)
        self._initialize_agent_states()
        self.message_history = []
        return self.get_state(0)
    
    def step(self, agent_id: int, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment.
        
        Args:
            agent_id: ID of the agent taking the action
            action: Action index (0: form connection, 1: remove connection,
                   2: post content, 3: share content)
        
        Returns:
            next_state: Next state representation
            reward: Reward value
            done: Whether episode is finished
            info: Additional information
        """
        prev_state = self.get_state(agent_id)
        
        # Execute action
        if action == 0:  # Form connection
            self._handle_form_connection(agent_id)
        elif action == 1:  # Remove connection
            self._handle_remove_connection(agent_id)
        elif action == 2:  # Post content
            self._handle_post_content(agent_id)
        elif action == 3:  # Share content
            self._handle_share_content(agent_id)
            
        # Get new state and calculate reward
        next_state = self.get_state(agent_id)
        reward = self.reward_function.calculate_reward(
            prev_state,
            next_state,
            self.network,
            agent_id
        )
        
        # Check if episode is done (e.g., max steps reached)
        done = False
        
        # Additional info
        info = {
            'network_density': nx.density(self.network),
            'clustering_coefficient': nx.average_clustering(self.network.to_undirected()),
            'degree_centrality': nx.degree_centrality(self.network)[agent_id]
        }
        
        return next_state, reward, done, info
    
    def get_state(self, agent_id: int) -> np.ndarray:
        """Get the current state representation for an agent."""
        return self.state_representation.get_state(
            self.network,
            self.agent_states,
            agent_id
        )
    
    def _handle_form_connection(self, agent_id: int):
        """Handle forming a new connection."""
        current_connections = list(self.network.successors(agent_id))
        if len(current_connections) >= self.max_connections:
            return
            
        possible_connections = [
            i for i in range(self.num_agents)
            if i != agent_id and i not in current_connections
        ]
        
        if possible_connections:
            # Choose connection based on personality similarity
            similarities = [
                np.dot(
                    self.agent_states[agent_id]['personality'],
                    self.agent_states[i]['personality']
                )
                for i in possible_connections
            ]
            probabilities = np.exp(similarities) / np.sum(np.exp(similarities))
            new_connection = np.random.choice(
                possible_connections,
                p=probabilities
            )
            self.network.add_edge(agent_id, new_connection)
    
    def _handle_remove_connection(self, agent_id: int):
        """Handle removing an existing connection."""
        current_connections = list(self.network.successors(agent_id))
        if current_connections:
            connection_to_remove = np.random.choice(current_connections)
            self.network.remove_edge(agent_id, connection_to_remove)
    
    def _handle_post_content(self, agent_id: int):
        """Handle posting new content."""
        content = {
            'author': agent_id,
            'topic': np.random.choice(
                self.content_topics,
                p=self.agent_states[agent_id]['content_preferences']
            ),
            'timestamp': len(self.message_history)
        }
        self.message_history.append(content)
        
        # Update influence scores based on content posting
        self.agent_states[agent_id]['influence_score'] *= 1.05
    
    def _handle_share_content(self, agent_id: int):
        """Handle sharing existing content."""
        if not self.message_history:
            return
            
        # Choose content to share based on topic preferences
        content_scores = [
            np.dot(
                self.agent_states[agent_id]['content_preferences'],
                np.eye(self.content_topics)[content['topic']]
            )
            for content in self.message_history[-10:]  # Consider recent messages
        ]
        
        if content_scores:
            probabilities = np.exp(content_scores) / np.sum(np.exp(content_scores))
            chosen_idx = np.random.choice(len(content_scores), p=probabilities)
            original_content = self.message_history[-10:][chosen_idx]
            
            # Create shared content
            shared_content = {
                'author': agent_id,
                'original_author': original_content['author'],
                'topic': original_content['topic'],
                'timestamp': len(self.message_history)
            }
            self.message_history.append(shared_content)
            
            # Update influence scores
            self.agent_states[agent_id]['influence_score'] *= 1.02
            self.agent_states[original_content['author']]['influence_score'] *= 1.03
