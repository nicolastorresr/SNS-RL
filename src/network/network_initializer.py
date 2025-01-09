import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple

class NetworkInitializer:
    def __init__(
        self,
        num_agents: int,
        personality_dim: int,
        content_topics: int,
        initial_connectivity: float = 0.1,
        community_structure: bool = True,
        num_communities: int = 3
    ):
        """Initialize network generator.
        
        Args:
            num_agents: Number of agents in the network
            personality_dim: Dimension of personality vectors
            content_topics: Number of content topics
            initial_connectivity: Initial network density
            community_structure: Whether to create community structure
            num_communities: Number of communities if using community structure
        """
        self.num_agents = num_agents
        self.personality_dim = personality_dim
        self.content_topics = content_topics
        self.initial_connectivity = initial_connectivity
        self.community_structure = community_structure
        self.num_communities = num_communities
        
    def create_network(self) -> Tuple[nx.DiGraph, Dict]:
        """Create and initialize network with agent states.
        
        Returns:
            Tuple of (network, agent_states)
        """
        if self.community_structure:
            network = self._create_community_network()
        else:
            network = self._create_random_network()
            
        agent_states = self._initialize_agent_states(network)
        return network, agent_states
    
    def _create_random_network(self) -> nx.DiGraph:
        """Create random network structure."""
        # Create Erdős-Rényi random graph
        network = nx.erdos_renyi_graph(
            self.num_agents,
            self.initial_connectivity,
            directed=True
        )
        
        # Ensure network is weakly connected
        while not nx.is_weakly_connected(network):
            # Add random edges until connected
            components = list(nx.weakly_connected_components(network))
            if len(components) > 1:
                comp1 = np.random.choice(list(components[0]))
                comp2 = np.random.choice(list(components[1]))
                network.add_edge(comp1, comp2)
                
        return network
    
    def _create_community_network(self) -> nx.DiGraph:
        """Create network with community structure."""
        # Calculate agents per community
        agents_per_community = self.num_agents // self.num_communities
        remainder = self.num_agents % self.num_communities
        
        # Create empty directed graph
        network = nx.DiGraph()
        network.add_nodes_from(range(self.num_agents))
        
        # Assign agents to communities
        communities = []
        start_idx = 0
        for i in range(self.num_communities):
            size = agents_per_community + (1 if i < remainder else 0)
            community = list(range(start_idx, start_idx + size))
            communities.append(community)
            start_idx += size
            
        # Add intra-community edges with high probability
        intra_community_prob = self.initial_connectivity * 3
        for community in communities:
            for i in community:
                for j in community:
                    if i != j and np.random.random() < intra_community_prob:
                        network.add_edge(i, j)
                        
        # Add inter-community edges with low probability
        inter_community_prob = self.initial_connectivity / 3
        for i in range(len(communities)):
            for j in range(i + 1, len(communities)):
                for node1 in communities[i]:
                    for node2 in communities[j]:
                        if np.random.random() < inter_community_prob:
                            # Add bidirectional edges with some probability
                            network.add_edge(node1, node2)
                            if np.random.random() < 0.5:
                                network.add_edge(node2, node1)
                                
        # Ensure network is weakly connected
        while not nx.is_weakly_connected(network):
            components = list(nx.weakly_connected_components(network))
            if len(components) > 1:
                comp1 = np.random.choice(list(components[0]))
                comp2 = np.random.choice(list(components[1]))
                network.add_edge(comp1, comp2)
                
        return network
    
    def _initialize_agent_states(self, network: nx.DiGraph) -> Dict:
        """Initialize agent states with attributes."""
        agent_states = {}
        
        # If using community structure, make personalities similar within communities
        communities = (
            list(nx.community.greedy_modularity_communities(network.to_undirected()))
            if self.community_structure
            else [list(network.nodes())]
        )
        
        community_personalities = [
            np.random.normal(0, 1, self.personality_dim)
            for _ in range(len(communities))
        ]
        
        # Initialize states for each agent
        for community_idx, community in enumerate(communities):
            base_personality = community_personalities[community_idx]
            
            for agent_id in community:
                # Add random variation to community personality
                personality = base_personality + np.random.normal(0, 0.3, self.personality_dim)
                personality = personality / np.linalg.norm(personality)  # Normalize
                
                # Initialize content preferences with Dirichlet distribution
                content_preferences = np.random.dirichlet(
                    np.ones(self.content_topics) * 2
                )
                
                agent_states[agent_id] = {
                    'personality': personality,
                    'content_preferences': content_preferences,
                    'influence_score': 1.0,
                    'community_id': community_idx if self.community_structure else 0,
                    'recent_interactions': []
                }
                
        return agent_states
