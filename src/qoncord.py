"""
Qoncord-inspired restart + promotion scheduling implementation.
Adaptive learning rate with periodic restarts and promotion on improvement.
Also implements multi-restart training with promotion of best candidates.
"""

import numpy as np
import time
from typing import Callable, List, Dict, Any, Optional
import copy


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


class RestartResult:
    """Container for restart training results."""
    
    def __init__(self, seed: int, params: np.ndarray, val_loss: float, 
                 val_accuracy: float, training_time: float, 
                 training_history: Dict[str, List[float]]):
        self.seed = seed
        self.params = params
        self.val_loss = val_loss
        self.val_accuracy = val_accuracy
        self.training_time = training_time
        self.training_history = training_history
    
    def __repr__(self):
        return (f"RestartResult(seed={self.seed}, val_loss={self.val_loss:.4f}, "
                f"val_acc={self.val_accuracy:.4f}, time={self.training_time:.2f}s)")


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)


def multi_restart_train(train_func: Callable,
                       N_restarts: int = 5,
                       short_epochs: int = 30,
                       K_promote: int = 2,
                       long_epochs: int = 200,
                       verbose: bool = True) -> tuple:
    """
    Qoncord-inspired multi-restart training with promotion.
    
    Explores multiple training trajectories from different initializations,
    then promotes the most promising candidates for extended training.
    
    Args:
        train_func: Training function that accepts 'epochs', 'seed', and optionally 'initial_params'
        N_restarts: Number of short exploratory runs (default: 5)
        short_epochs: Epochs for initial exploration phase (default: 30)
        K_promote: Number of top candidates to promote (default: 2)
        long_epochs: Epochs for promoted long training (default: 200)
        verbose: Whether to print progress information
    
    Returns:
        tuple: (all_short_results, promoted_final_results)
            - all_short_results: List of RestartResult from short training phase
            - promoted_final_results: List of RestartResult from promoted long training
    
    Example:
        >>> def my_train(epochs, seed, initial_params=None):
        ...     # Your training logic here
        ...     return RestartResult(...)
        >>> short_results, final_results = multi_restart_train(my_train, N_restarts=5)
    """
    
    if verbose:
        print("=" * 70)
        print("Qoncord Multi-Restart Training with Promotion")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  • N_restarts: {N_restarts} (exploration phase)")
        print(f"  • Short epochs: {short_epochs}")
        print(f"  • K_promote: {K_promote} (top candidates)")
        print(f"  • Long epochs: {long_epochs}")
        print("=" * 70)
    
    # Phase 1: Exploratory short training with multiple restarts
    if verbose:
        print(f"\n📊 Phase 1: Exploratory Training ({N_restarts} restarts)")
        print("-" * 70)
    
    short_results = []
    for i in range(N_restarts):
        if verbose:
            print(f"\nRestart {i+1}/{N_restarts} (seed={i})...")
        
        # Set seed for reproducibility
        set_seed(i)
        
        # Run short training
        start_time = time.time()
        result = train_func(epochs=short_epochs, seed=i)
        elapsed = time.time() - start_time
        
        # Update timing if not set by train_func
        if hasattr(result, 'training_time') and result.training_time == 0:
            result.training_time = elapsed
        
        short_results.append(result)
        
        if verbose:
            print(f"  ✓ Val Loss: {result.val_loss:.4f}, "
                  f"Val Acc: {result.val_accuracy:.4f}, "
                  f"Time: {result.training_time:.2f}s")
    
    # Rank results by validation loss (lower is better)
    short_results.sort(key=lambda r: r.val_loss)
    
    if verbose:
        print("\n" + "-" * 70)
        print("📈 Short Training Results (ranked by validation loss):")
        for i, result in enumerate(short_results):
            marker = "⭐" if i < K_promote else "  "
            print(f"  {marker} Rank {i+1}: {result}")
    
    # Phase 2: Promote top K candidates for long training
    promoted = short_results[:K_promote]
    
    if verbose:
        print("\n" + "=" * 70)
        print(f"🚀 Phase 2: Promoted Training (top {K_promote} candidates)")
        print("-" * 70)
    
    final_results = []
    for i, restart_result in enumerate(promoted):
        if verbose:
            print(f"\nPromoted Candidate {i+1}/{K_promote} (from seed={restart_result.seed})...")
            print(f"  Initial val loss: {restart_result.val_loss:.4f}")
        
        # Continue training from best checkpoint
        start_time = time.time()
        result = train_func(epochs=long_epochs, 
                          seed=restart_result.seed,
                          initial_params=restart_result.params)
        elapsed = time.time() - start_time
        
        # Update timing
        if hasattr(result, 'training_time') and result.training_time == 0:
            result.training_time = elapsed
        
        final_results.append(result)
        
        if verbose:
            improvement = restart_result.val_loss - result.val_loss
            print(f"  ✓ Final val loss: {result.val_loss:.4f} "
                  f"(improved by {improvement:.4f})")
            print(f"  ✓ Final val acc: {result.val_accuracy:.4f}")
            print(f"  ✓ Long training time: {result.training_time:.2f}s")
    
    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print("📊 Final Summary")
        print("=" * 70)
        
        best_short = short_results[0]
        best_final = min(final_results, key=lambda r: r.val_loss)
        
        print(f"\nBest from short training:")
        print(f"  • Val Loss: {best_short.val_loss:.4f}")
        print(f"  • Val Accuracy: {best_short.val_accuracy:.4f}")
        print(f"  • Seed: {best_short.seed}")
        
        print(f"\nBest after promotion:")
        print(f"  • Val Loss: {best_final.val_loss:.4f}")
        print(f"  • Val Accuracy: {best_final.val_accuracy:.4f}")
        print(f"  • Seed: {best_final.seed}")
        print(f"  • Total improvement: {best_short.val_loss - best_final.val_loss:.4f}")
        
        total_time = sum(r.training_time for r in short_results) + sum(r.training_time for r in final_results)
        print(f"\nTotal training time: {total_time:.2f}s")
        print("=" * 70)
    
    return short_results, final_results
