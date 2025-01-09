import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import seaborn as sns
from matplotlib.animation import FuncAnimation

class PropagationVisualizer:
    """
    Handles visualization of message propagation and content spread in the network.
    """
    
    def __init__(self, output_dir: str = "data/output/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_message_spread(
        self,
        propagation_data: List[Dict],
        time_steps: List[int],
        title: str = "Message Propagation Over Time",
        filename: Optional[str] = None
    ) -> None:
        """
        Plot the spread of messages through the network over time.
        
        Args:
            propagation_data: List of dictionaries containing spread metrics
            time_steps: List of time steps
            title: Plot title
            filename: Output filename (optional)
        """
        plt.figure(figsize=(10, 6))
        
        # Extract metrics
        reach = [d['reach'] for d in propagation_data]
        engagement = [d['engagement'] for d in propagation_data]
        
        # Plot metrics
        plt.plot(time_steps, reach, 'b-', label='Message Reach')
        plt.plot(time_steps, engagement, 'r--', label='User Engagement')
        
        plt.xlabel('Time Step')
        plt.ylabel('Count')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if filename:
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def create_propagation_animation(
        self,
        G: nx.Graph,
        propagation_states: List[Dict],
        filename: str = "propagation_animation.gif",
        interval: int = 500
    ) -> None:
        """
        Create an animation of message propagation through the network.
        
        Args:
            G: NetworkX graph object
            propagation_states: List of dictionaries containing node states at each time step
            filename: Output filename
            interval: Time between frames in milliseconds
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        pos = nx.spring_layout(G)
        
        def update(frame):
            ax.clear()
            
            # Get node states for current frame
            node_colors = [propagation_states[frame][node] for node in G.nodes()]
            
            # Draw network
            nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                                 node_size=500, alpha=0.7,
                                 cmap=plt.cm.RdYlBu)
            nx.draw_networkx_edges(G, pos, alpha=0.2)
            nx.draw_networkx_labels(G, pos)
            
            ax.set_title(f'Propagation Step {frame}')
            ax.axis('off')
        
        # Create animation
        anim = FuncAnimation(fig, update,
                           frames=len(propagation_states),
                           interval=interval, blit=False)
        
        # Save animation
        anim.save(self.output_dir / filename, writer='pillow')
        plt.close()
    
    def plot_content_type_distribution(
        self,
        content_data: Dict[str, List[float]],
        time_steps: List[int],
        title: str = "Content Type Distribution Over Time",
        filename: Optional[str] = None
    ) -> None:
        """
        Plot the distribution of different content types over time.
        
        Args:
            content_data: Dictionary mapping content types to lists of proportions
            time_steps: List of time steps
            title: Plot title
            filename: Output filename (optional)
        """
        plt.figure(figsize=(12, 6))
        
        # Create stacked area plot
        plt.stackplot(time_steps,
                     content_data.values(),
                     labels=content_data.keys(),
                     alpha=0.7)
        
        plt.xlabel('Time Step')
        plt.ylabel('Proportion')
        plt.title(title)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True, alpha=0.3)
        
        if filename:
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_influence_evolution(
        self,
        influence_history: Dict[int, List[float]],
        time_steps: List[int],
        top_k: int = 5,
        title: str = "Evolution of Top Influencers",
        filename: Optional[str] = None
    ) -> None:
        """
        Plot the evolution of influence scores for top influencers.
        
        Args:
            influence_history: Dictionary mapping node IDs to lists of influence scores
            time_steps: List of time steps
            top_k: Number of top influencers to show
            title: Plot title
            filename: Output filename (optional)
        """
        plt.figure(figsize=(12, 6))
        
        # Find top-k influencers based on final influence scores
        final_scores = {node: scores[-1] 
                       for node, scores in influence_history.items()}
        top_nodes = sorted(final_scores.items(),
                          key=lambda x: x[1],
                          reverse=True)[:top_k]
        
        # Plot influence evolution for top-k nodes
        for node, _ in top_nodes:
            plt.plot(time_steps, influence_history[node],
                    label=f'Node {node}', alpha=0.7)
        
        plt.xlabel('Time Step')
        plt.ylabel('Influence Score')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if filename:
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
