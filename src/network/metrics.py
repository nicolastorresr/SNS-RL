import networkx as nx
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

class NetworkMetrics:
    def __init__(self, window_size: int = 100):
        """Initialize network metrics calculator.
        
        Args:
            window_size: Window size for temporal metrics
        """
        self.window_size = window_size
        self.temporal_metrics = defaultdict(list)
        
    def calculate_structural_metrics(
        self,
        network: nx.DiGraph,
        agent_states: Dict
    ) -> Dict:
        """Calculate structural network metrics.
        
        Args:
            network: Network structure
            agent_states: Agent state information
            
        Returns:
            Dictionary of metric values
        """
        metrics = {}
        
        # Basic network metrics
        metrics['num_nodes'] = network.number_of_nodes()
        metrics['num_edges'] = network.number_of_edges()
        metrics['density'] = nx.density(network)
        metrics['reciprocity'] = nx.reciprocity(network)
        
        # Centrality metrics
        try:
            metrics['degree_centralization'] = self._calculate_centralization(
                nx.degree_centrality(network)
            )
            metrics['betweenness_centralization'] = self._calculate_centralization(
                nx.betweenness_centrality(network)
            )
            metrics['eigenvector_centralization'] = self._calculate_centralization(
                nx.eigenvector_centrality(network, max_iter=1000)
            )
        except:
            metrics['degree_centralization'] = 0
            metrics['betweenness_centralization'] = 0
            metrics['eigenvector_centralization'] = 0
        
        # Clustering and community metrics
        undirected = network.to_undirected()
        metrics['avg_clustering'] = nx.average_clustering(undirected)
        metrics['transitivity'] = nx.transitivity(undirected)
        
        # Community detection
        communities = list(nx.community.greedy_modularity_communities(undirected))
        metrics['num_communities'] = len(communities)
        metrics['modularity'] = self._calculate_modularity(network, communities)
        
        # Path-based metrics
        if nx.is_weakly_connected(network):
            metrics['avg_path_length'] = nx.average_shortest_path_length(network)
            metrics['diameter'] = nx.diameter(network)
        else:
            metrics['avg_path_length'] = float('inf')
            metrics['diameter'] = float('inf')
            
        return metrics
    
    def calculate_agent_metrics(
        self,
        network: nx.DiGraph,
        agent_states: Dict,
        agent_id: int
    ) -> Dict:
        """Calculate metrics for a specific agent.
        
        Args:
            network: Network structure
            agent_states: Agent state information
            agent_id: ID of agent to analyze
            
        Returns:
            Dictionary of metric values
        """
        metrics = {}
        
        # Degree metrics
        metrics['in_degree'] = network.in_degree(agent_id)
        metrics['out_degree'] = network.out_degree(agent_id)
        metrics['degree_ratio'] = (
            metrics['in_degree'] / max(metrics['out_degree'], 1)
        )
        
        # Centrality metrics
        metrics['degree_centrality'] = nx.degree_centrality(network)[agent_id]
        try:
            metrics['betweenness_centrality'] = nx.betweenness_centrality(network)[agent_id]
            metrics['eigenvector_centrality'] = nx.eigenvector_centrality(network)[agent_id]
        except:
            metrics['betweenness_centrality'] = 0
            metrics['eigenvector_centrality'] = 0
            
        # Local clustering
        metrics['clustering_coefficient'] = nx.clustering(
            network.to_undirected(),
            agent_id
        )
        
        # Influence metrics
        metrics['influence_score'] = agent_states[agent_id]['influence_score']
        
        # Neighbor similarity
        metrics['neighbor_similarity'] = self._calculate_neighbor_similarity(
            network,
            agent_states,
            agent_id
        )
        
        return metrics
    
    def update_temporal_metrics(
        self,
        network: nx.DiGraph,
        agent_states: Dict,
        message_history: List
    ):
        """Update temporal network metrics.
        
        Args:
            network: Network structure
            agent_states: Agent state information
            message_history: List of message events
        """
        # Calculate current metrics
        current_metrics = self.calculate_structural_metrics(network, agent_states)
        
        # Update temporal metrics
        for metric, value in current_metrics.items():
            self.temporal_metrics[metric].append(value)
            
            # Keep only recent values
            if len(self.temporal_metrics[metric]) > self.window_size:
                self.temporal_metrics[metric].pop(0)
                
        # Calculate message-based metrics
        if message_history:
            recent_messages = message_history[-self.window_size:]
            self.temporal_metrics['message_rate'].append(len(recent_messages))
            
            # Calculate content diversity
            topics = [msg['topic'] for msg in recent_messages]
            topic_diversity = len(set(topics)) / len(topics) if topics else 0
            self.temporal_metrics['topic_diversity'].append(topic_diversity)
            
            # Calculate influence distribution
            authors = [msg['author'] for msg in recent_messages]
            unique_authors = len(set(authors))
            self.temporal_metrics['author_diversity'].append(
                unique_authors / len(authors) if authors else 0
            )
    
    def get_temporal_metrics(self) -> Dict:
        """Get temporal metric statistics.
        
        Returns:
            Dictionary of metric statistics
        """
        stats = {}
        
        for metric, values in self.temporal_metrics.items():
            if values:
                stats[f'{metric}_mean'] = np.mean(values)
                stats[f'{metric}_std'] = np.std(values)
                stats[f'{metric}_trend'] = self._calculate_trend(values)
                
        return stats
    
    def _calculate_centralization(self, centrality_dict: Dict) -> float:
        """Calculate network centralization from centrality values."""
        values = list(centrality_dict.values())
        max_val = max(values)
        n = len(values)
        
        if n <= 1 or max_val == 0:
            return 0.0
            
        # Calculate theoretical maximum centralization
        theoretical_max = (n - 1) * (n - 2)
        
        # Calculate actual centralization
        centralization = sum(max_val - val for val in values) / theoretical_max
        
        return centralization
    
    def _calculate_modularity(
        self,
        network: nx.DiGraph,
        communities: List[List[int]]
    ) -> float:
        """Calculate network modularity."""
        if not communities:
            return 0.0
            
        modularity = 0
        m = network.number_of_edges()
        if m == 0:
            return 0.0
            
        # Create community membership dictionary
        community_dict = {}
        for idx, community in enumerate(communities):
            for node in community:
                community_dict[node] = idx
                
        # Calculate modularity
        for i, j in network.edges():
            if community_dict[i] == community_dict[j]:
                modularity += 1
                
        return modularity / m - sum(len(c) * len(c) for c in communities) / (4 * m * m)
    
    def _calculate_neighbor_similarity(
        self,
        network: nx.DiGraph,
        agent_states: Dict,
        agent_id: int
    ) -> float:
        """Calculate average similarity with neighbors."""
        neighbors = list(network.successors(agent_id))
        if not neighbors:
            return 0.0
            
        similarities = []
        agent_personality = agent_states[agent_id]['personality']
        agent_preferences = agent_states[agent_id]['content_preferences']
        
        for neighbor in neighbors:
            neighbor_personality = agent_states[neighbor]['personality']
            neighbor_preferences = agent_states[neighbor]['content_preferences']
            
            # Calculate cosine similarity for both personality and preferences
            personality_sim = np.dot(agent_personality, neighbor_personality)
            preference_sim = np.dot(agent_preferences, neighbor_preferences)
            
            # Average the similarities
            similarities.append((personality_sim + preference_sim) / 2)
            
        return np.mean(similarities)
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in temporal metrics using linear regression."""
        if len(values) < 2:
            return 0.0
            
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate linear regression slope
        slope, _ = np.polyfit(x, y, 1)
        return slope
