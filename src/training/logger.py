import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt

class TrainingLogger:
    """Handles logging and visualization of training progress."""
    
    def __init__(self, log_dir: str = "data/output/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"training_{timestamp}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # Initialize metrics storage
        self.metrics_history = {
            'episodes': [],
            'rewards': [],
            'losses': [],
            'network_metrics': []
        }
    
    def log_episode(self, episode: int, metrics: Dict[str, Any]):
        """Log metrics for a training episode."""
        self.metrics_history['episodes'].append(episode)
        self.metrics_history['rewards'].append(metrics['reward'])
        self.metrics_history['losses'].append(metrics.get('loss'))
        
        if 'eval_metrics' in metrics:
            self.metrics_history['network_metrics'].append(metrics['eval_metrics'])
        
        # Log to file
        log_msg = (f"Episode {episode} - Reward: {metrics['reward']:.2f}"
                  f" - Loss: {metrics.get('loss', 'N/A')}")
        self.logger.info(log_msg)
        
        # Save metrics to JSON
        self._save_metrics()
    
    def _save_metrics(self):
        """Save metrics history to JSON file."""
        metrics_file = self.log_dir / "metrics_history.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)
    
    def plot_training_progress(self):
        """Plot training metrics."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot rewards
        ax1.plot(self.metrics_history['episodes'],
                self.metrics_history['rewards'])
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Rewards')
        
        # Plot losses
        losses = [l for l in self.metrics_history['losses'] if l is not None]
        if losses:
            ax2.plot(self.metrics_history['episodes'][:len(losses)], losses)
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Loss')
            ax2.set_title('Training Loss')
        
        plt.tight_layout()
        plt.savefig(self.log_dir / "training_progress.png")
        plt.close()
