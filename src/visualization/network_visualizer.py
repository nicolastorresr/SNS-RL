import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import seaborn as sns

class NetworkVisualizer:
    """
    Handles visualization of social network structure and metrics.
    """
    
    def __init__(self, output_dir: str = "data/output/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.color_palette = sns.color_palette("husl", 8)
    
    def visualize_network(
        self,
        G: nx.Graph,
        node_attributes: Optional[Dict] = None,
        edge_weights: Optional[Dict] = None,
        title: str = "Social Network Structure",
        filename: Optional[str] = None
    ) -> None:
        """
        Visualize the network structure with node attributes and edge weights.
        
        Args:
            G: NetworkX graph object
            node_attributes: Dictionary of node attributes for coloring
            edge_weights: Dictionary of edge weights
            title: Plot title
            filename: Output filename (optional)
        """
        plt.figure(figsize=(12, 8))
        
        # Calculate layout
        pos = nx.spring_layout(G, k=1/np.sqrt(G.number_of_nodes()), iterations=50)
        
        # Draw nodes
        if node_attributes:
            # Color nodes based on attributes
            node_colors = [node_attributes[node] for node in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                 node_size=500, alpha=0.6, 
                                 cmap=plt.cm.viridis)
        else:
            nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                                 node_size=500, alpha=0.6)
        
        # Draw edges
        if edge_weights:
            edges = [(u, v) for (u, v) in G.edges()]
            weights = [edge_weights[(u, v)] for (u, v) in edges]
            nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights,
                                 alpha=0.5, edge_color='gray')
        else:
            nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray')
        
        # Add labels
        nx.draw_networkx_labels(G, pos)
        
        plt.title(title)
        plt.axis('off')
        
        if filename:
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_community_structure(
        self,
        G: nx.Graph,
        communities: List[List[int]],
        title: str = "Community Structure",
        filename: Optional[str] = None
    ) -> None:
        """
        Visualize community structure in the network.
        
        Args:
            G: NetworkX graph object
            communities: List of lists containing node indices for each community
            title: Plot title
            filename: Output filename (optional)
        """
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        
        # Create color map for communities
        color_map = {}
        for idx, community in enumerate(communities):
            for node in community:
                color_map[node] = self.color_palette[idx % len(self.color_palette)]
        
        # Draw nodes colored by community
        node_colors = [color_map[node] for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                             node_size=500, alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.2)
        
        # Add labels
        nx.draw_networkx_labels(G, pos)
        
        plt.title(title)
        plt.axis('off')
        
        if filename:
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_influence_distribution(
        self,
        influence_scores: Dict[int, float],
        title: str = "Influence Distribution",
        filename: Optional[str] = None
    ) -> None:
        """
        Plot the distribution of influence scores across nodes.
        
        Args:
            influence_scores: Dictionary mapping node IDs to influence scores
            title: Plot title
            filename: Output filename (optional)
        """
        plt.figure(figsize=(10, 6))
        
        scores = list(influence_scores.values())
        plt.hist(scores, bins=30, alpha=0.7, color='blue')
        plt.xlabel('Influence Score')
        plt.ylabel('Number of Nodes')
        plt.title(title)
        
        if filename:
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
