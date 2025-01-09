import numpy as np
import networkx as nx
from typing import Dict

class RewardFunction:
    def __init__(
        self,
        influence_weight: float = 0.4,
        diversity_weight: float = 0.3,
        efficiency_weight: float = 0.3
    ):
        """Initialize reward function parameters.
        
        Args:
            influence_weight: Weight for influence-based reward
            diversity_weight: Weight for diversity-based reward
            efficiency_weight: Weight for network efficiency reward
        """
        self.influence_weight = influence_weight
        self.diversity_weight = diversity_weight
        self.efficiency_weight = efficiency_weight
    
    def calculate_reward(
        self,
        prev_state: np.ndarray,
        next_state: np.ndarray,
        network: nx.DiGraph,
        agent_id: int
    ) -> float:
        """Calculate the reward for a state transition.
        
        Args:
            prev_state: Previous state vector
            next_state: Current state vector
            network: Current network structure
            agent_id: ID of the agent
            
        Returns:
            Combined reward value
        """
        # Calculate influence-based reward
        influence_reward = self._calculate_influence_reward(
            prev_state,
            next_state
        )
        
        # Calculate diversity-based reward
        diversity_reward = self._calculate_diversity_reward(
            network,
            agent_id
        )
        
        # Calculate network efficiency reward
        efficiency_reward = self._calculate_efficiency_reward(
            network,
            agent_id
        )
        
        # Combine rewards using weights
        total_reward = (
            self.influence_weight * influence_reward +
            self.diversity_weight * diversity_reward +
            self.efficiency_weight * efficiency_reward
        )
        
        return total_reward
    
    def _calculate_influence_reward(
        self,
        prev_state: np.ndarray,
        next_state: np.ndarray
    ) -> float:
        """Calculate reward based on change in influence score."""
        # Extract influence scores from states
        prev_influence = prev_state[prev_state.shape[0] - 1]  # Influence is last element
        next_influence = next_state[next_state.shape[0] - 1]
        
        # Calculate relative change in influence
        influence_change = (next_influence - prev_influence) / prev_influence
        
        # Apply non-linear scaling to reward
        return np.tanh(influence_change * 5)  # Scale factor of 5 for reasonable values
    
    def _calculate_diversity_reward(
        self,
        network: nx.DiGraph,
        agent_id: int
    ) -> float:
        """Calculate reward based on network diversity."""
        # Get neighbors
        neighbors = list(network.successors(agent_id))
        if not neighbors:
            return 0.0
            
        # Calculate network metrics
        try:
            # Local clustering coefficient (negative reward for too much clustering)
            clustering = nx.clustering(network.to_undirected(), agent_id)
            clustering_reward = -np.tanh(clustering * 2)  # Penalize high clustering
            
            # Path diversity (reward for being connected to different communities)
            paths = nx.single_source_shortest_path_length(network, agent_id)
            path_lengths = list(paths.values())
            path_diversity = np.std(path_lengths) if path_lengths else 0
            path_reward = np.tanh(path_diversity)
            
            return (clustering_reward + path_reward) / 2
            
        except:
            return 0.0
          
    def _calculate_efficiency_reward(
            self,
            network: nx.DiGraph,
            agent_id: int
        ) -> float:
            """Calculate reward based on network efficiency.
            
            Evaluates:
            1. Local efficiency (ratio of actual to potential connections)
            2. Betweenness centrality (importance in information flow)
            3. Reach efficiency (ability to reach other nodes quickly)
            
            Returns:
                Normalized efficiency reward between -1 and 1
            """
            try:
                # Calculate out-degree and local density
                out_degree = network.out_degree(agent_id)
                neighbors = set(network.successors(agent_id))
                if not neighbors:
                    return 0.0
                    
                # Calculate local efficiency (connections between neighbors)
                local_edges = 0
                potential_edges = len(neighbors) * (len(neighbors) - 1)
                
                for n1 in neighbors:
                    for n2 in neighbors:
                        if n1 != n2 and network.has_edge(n1, n2):
                            local_edges += 1
                            
                local_efficiency = (
                    local_edges / potential_edges if potential_edges > 0 else 0
                )
                
                # Calculate betweenness centrality (normalized)
                betweenness = nx.betweenness_centrality(network, k=min(100, len(network)))[agent_id]
                
                # Calculate reach efficiency (average shortest path length to others)
                paths = nx.single_source_shortest_path_length(network, agent_id)
                avg_path_length = np.mean(list(paths.values())) if paths else float('inf')
                reach_efficiency = 1.0 / (1.0 + avg_path_length)  # Normalize to [0,1]
                
                # Combine metrics with weights
                efficiency_score = (
                    0.3 * (1 - local_efficiency) +  # Prefer less clustering for efficiency
                    0.4 * betweenness +            # Reward being a central node
                    0.3 * reach_efficiency         # Reward shorter paths to others
                )
                
                # Scale to [-1, 1] range and apply non-linear scaling
                return np.tanh(efficiency_score * 2)
                
            except (nx.NetworkXError, ZeroDivisionError):
                return 0.0
