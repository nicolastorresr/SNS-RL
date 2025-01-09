import numpy as np
import networkx as nx
from typing import Dict

class StateRepresentation:
    def __init__(self, personality_dim: int, content_topics: int):
        """Initialize state representation parameters.
        
        Args:
            personality_dim: Dimension of personality vectors
            content_topics: Number of different content topics
        """
        self.personality_dim = personality_dim
        self.content_topics = content_topics
        
    def get_state(
        self,
        network: nx.DiGraph,
        agent_states: Dict,
        agent_id: int
    ) -> np.ndarray:
        """Generate state representation for an agent.
        
        Args:
            network: Current network structure
            agent_states: Dictionary of agent states
            agent_id: ID of the agent
            
        Returns:
            State vector containing:
            - Agent's personality traits
            - Agent's content preferences
            - Current influence score
            - Network features (degree, clustering, etc.)
            - Recent interaction statistics
        """
        # Get agent's personality and preferences
        personality = agent_states[agent_id]['personality']
        content_prefs = agent_states[agent_id]['content_preferences']
        influence = agent_states[agent_id]['influence_score']
        
        # Calculate network features
        degree = network.degree(agent_id)
        in_degree = network.in_degree(agent_id)
        out_degree = network.out_degree(agent_id)
        
        # Local clustering coefficient
        try:
            clustering = nx.clustering(network.to_undirected(), agent_id)
        except:
            clustering = 0.0
            
        # Calculate average neighbor properties
        neighbor_personality = np.zeros(self.personality_dim)
        neighbor_content_prefs = np.zeros(self.content_topics)
        neighbor_influence = 0.0
        neighbors = list(network.successors(agent_id))
        
        if neighbors:
            for neighbor in neighbors:
                neighbor_personality += agent_states[neighbor]['personality']
                neighbor_content_prefs += agent_states[neighbor]['content_preferences']
                neighbor_influence += agent_states[neighbor]['influence_score']
            
            neighbor_personality /= len(neighbors)
            neighbor_content_prefs /= len(neighbors)
            neighbor_influence /= len(neighbors)
        
        # Combine all features into state vector
        state = np.concatenate([
            personality,                    # Agent's personality
            content_prefs,                 # Content preferences
            [influence],                   # Influence score
            [degree, in_degree, out_degree],  # Network metrics
            [clustering],                  # Clustering coefficient
            neighbor_personality,          # Average neighbor personality
            neighbor_content_prefs,        # Average neighbor content preferences
            [neighbor_influence]           # Average neighbor influence
        ])
        
        return state
        
    def get_state_dim(self) -> int:
        """Get the dimension of the state vector."""
        return (
            self.personality_dim +        # Agent personality
            self.content_topics +         # Content preferences
            1 +                          # Influence score
            3 +                          # Degree metrics
            1 +                          # Clustering coefficient
            self.personality_dim +        # Neighbor personality
            self.content_topics +         # Neighbor content preferences
            1                            # Neighbor influence
        )
