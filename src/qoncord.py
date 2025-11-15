"""
Qoncord-inspired restart + promotion scheduling implementation.
Adaptive learning rate with periodic restarts and promotion on improvement.
"""

import numpy as np


class QoncordScheduler:
    """
    Qoncord-inspired restart + promotion scheduling for optimization.
    Implements adaptive learning rate scheduling with periodic restarts.
    """
    
    def __init__(self, base_lr: float = 0.01, restart_period: int = 10):
        self.base_lr = base_lr
        self.restart_period = restart_period
        self.iteration = 0
        self.best_loss = float('inf')
        self.promoted_lr = base_lr
    
    def get_learning_rate(self, current_loss: float) -> float:
        """
        Get learning rate with restart and promotion scheduling.
        """
        self.iteration += 1
        
        # Check for promotion (improvement)
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            # Promote: slightly increase learning rate
            self.promoted_lr = min(self.base_lr * 1.2, 0.1)
        
        # Periodic restart
        if self.iteration % self.restart_period == 0:
            # Restart with cosine annealing
            cycle_position = (self.iteration % (self.restart_period * 2)) / (self.restart_period * 2)
            lr = self.promoted_lr * (1 + np.cos(np.pi * cycle_position)) / 2
            return max(lr, self.base_lr * 0.1)
        
        return self.promoted_lr
    
    def reset(self):
        """Reset scheduler state."""
        self.iteration = 0
        self.best_loss = float('inf')
        self.promoted_lr = self.base_lr
