"""
Training loop with latency simulation and restarts.
Implements the main training procedure with all optimizations.
"""

import numpy as np
import time
from typing import Tuple, Callable, Any
from .model import RLGS_QNN
from .datasets import prepare_dataset, prepare_toy_dataset


def qnode_call_with_latency(qnode_func: Callable, params: np.ndarray, x: np.ndarray, latency: float) -> Any:
    """
    Call QNode with simulated classical↔quantum latency.
    
    Qtenon-inspired latency simulation: inserts time.sleep at each QNode call
    to simulate communication delay between classical and quantum processors.
    
    Args:
        qnode_func: The quantum node function to call
        params: Circuit parameters
        x: Input data
        latency: Latency in seconds (e.g., 0.001 for 1ms, 0.0001 for 0.1ms)
    
    Returns:
        Output from the QNode function
    """
    # Execute quantum circuit
    out = qnode_func(params, x)
    
    # Simulate classical↔quantum communication latency
    time.sleep(latency)
    
    return out


def train_qnn(qnn: RLGS_QNN, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: np.ndarray, 
              y_val: np.ndarray,
              epochs: int = 50, 
              batch_size: int = 5,
              latency_mode: str = 'low'):
    """
    Train the QNN with all RLGS-inspired features.
    Includes Qtenon-inspired latency simulation, restart scheduling, and constraint checking.
    
    Args:
        qnn: RLGS_QNN instance
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Number of training epochs
        batch_size: Batch size for training
        latency_mode: 'low' (0.1ms, Qtenon-style) or 'high' (1ms, standard)
    """
    # Configure Qtenon-inspired latency simulation
    latency_config = {
        'low': 0.0001,   # 0.1 ms - Qtenon-style low latency
        'high': 0.001    # 1 ms - standard latency
    }
    latency = latency_config.get(latency_mode, 0.0001)
    qnn.low_latency_loop.latency_ms = latency * 1000  # Convert to ms for internal use
    
    print(f"Training RLGS-QNN with {qnn.n_qubits} qubits, {qnn.n_layers} layers")
    print(f"RLGS graph simplification: {len(qnn.graph_simplifier.analyze_connectivity([(i, i+1) for i in range(qnn.n_qubits - 1)]))} CZ gates")
    print(f"Qtenon latency mode: {latency_mode} ({latency*1000:.2f} ms)")
    
    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        epoch_loss = 0
        n_batches = len(X_train) // batch_size
        
        for batch in range(n_batches):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            # Compute loss and gradients
            loss = qnn.compute_loss(X_batch, y_batch)
            epoch_loss += loss
            
            # Get adaptive learning rate from Qoncord scheduler
            lr = qnn.scheduler.get_learning_rate(loss)
            
            # Simple gradient descent update (numerical gradients)
            grad = qnn._compute_gradients(X_batch, y_batch)
            qnn.params -= lr * grad
        
        # Validation
        val_loss = qnn.compute_loss(X_val, y_val)
        val_predictions = qnn.predict(X_val)
        val_accuracy = np.mean((val_predictions > 0.5) == (y_val > 0.5))
        
        # Track metrics
        qnn.training_history['loss'].append(epoch_loss / n_batches)
        qnn.training_history['accuracy'].append(val_accuracy)
        qnn.training_history['learning_rate'].append(lr)
        qnn.training_history['gate_count'].append(
            qnn.n_qubits * qnn.n_layers * 3 + len(
                qnn.graph_simplifier.analyze_connectivity(
                    [(i, i+1) for i in range(qnn.n_qubits - 1)]
                )
            ) * (qnn.n_layers - 1)
        )
        qnn.training_history['latency'].append(
            qnn.low_latency_loop.get_average_latency()
        )
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss/n_batches:.4f}, "
                  f"Val Acc: {val_accuracy:.4f}, LR: {lr:.6f}")
    
    print("\nTraining completed!")
    print(f"Average latency: {qnn.low_latency_loop.get_average_latency():.2f} ms")
    print(f"Constraint satisfaction: {qnn.edge_constraints.get_constraint_satisfaction()}")
    
    return qnn


def qnn_train_for_restart(X_train, X_val, y_train, y_val, 
                         n_qubits: int = 4, 
                         n_layers: int = 2,
                         batch_size: int = 5,
                         latency_mode: str = 'low'):
    """
    Create a training function compatible with multi_restart_train.
    
    Returns a function that can be called with epochs, seed, and initial_params.
    """
    from .qoncord import RestartResult, set_seed
    
    def train_func(epochs: int, seed: int, initial_params=None):
        """Training function for a single restart."""
        # Set seed
        set_seed(seed)
        
        # Initialize QNN
        qnn = RLGS_QNN(n_qubits=n_qubits, n_layers=n_layers)
        
        # Use initial params if provided (for promoted training)
        if initial_params is not None:
            qnn.params = initial_params.copy()
        
        # Train
        start_time = time.time()
        qnn = train_qnn(qnn, X_train, y_train, X_val, y_val,
                       epochs=epochs, batch_size=batch_size, 
                       latency_mode=latency_mode)
        training_time = time.time() - start_time
        
        # Get final metrics
        val_loss = qnn.training_history['loss'][-1] if qnn.training_history['loss'] else float('inf')
        val_accuracy = qnn.training_history['accuracy'][-1] if qnn.training_history['accuracy'] else 0.0
        
        # Return RestartResult
        return RestartResult(
            seed=seed,
            params=qnn.params.copy(),
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            training_time=training_time,
            training_history=qnn.training_history
        )
    
    return train_func


def compare_latency_modes(X_train, X_val, y_train, y_val, epochs: int = 10):
    """
    Compare training with high vs low latency modes (Qtenon-inspired).
    
    Demonstrates the impact of classical↔quantum communication latency
    on training time and performance.
    """
    print("\n" + "=" * 70)
    print("Qtenon Latency Comparison: High (1ms) vs Low (0.1ms)")
    print("=" * 70)
    
    results = {}
    
    for mode in ['high', 'low']:
        print(f"\n{'='*70}")
        print(f"Training with {mode.upper()} latency mode")
        print(f"{'='*70}")
        
        # Initialize fresh QNN
        qnn = RLGS_QNN(n_qubits=4, n_layers=2)
        
        # Train with specified latency mode
        start_time = time.time()
        qnn = train_qnn(qnn, X_train, y_train, X_val, y_val, 
                       epochs=epochs, batch_size=5, latency_mode=mode)
        training_time = time.time() - start_time
        
        # Store results
        final_acc = qnn.training_history['accuracy'][-1]
        avg_latency = qnn.low_latency_loop.get_average_latency()
        
        results[mode] = {
            'training_time': training_time,
            'final_accuracy': final_acc,
            'avg_latency': avg_latency
        }
    
    # Print comparison
    print("\n" + "=" * 70)
    print("Latency Mode Comparison Results")
    print("=" * 70)
    print(f"\nHigh Latency (1ms):")
    print(f"  Training time: {results['high']['training_time']:.2f}s")
    print(f"  Final accuracy: {results['high']['final_accuracy']:.2%}")
    print(f"  Avg latency: {results['high']['avg_latency']:.4f} ms")
    
    print(f"\nLow Latency (0.1ms - Qtenon-style):")
    print(f"  Training time: {results['low']['training_time']:.2f}s")
    print(f"  Final accuracy: {results['low']['final_accuracy']:.2%}")
    print(f"  Avg latency: {results['low']['avg_latency']:.4f} ms")
    
    speedup = results['high']['training_time'] / results['low']['training_time']
    print(f"\n🚀 Speedup with Qtenon low-latency: {speedup:.2f}x faster")
    print("=" * 70)
    
    return results


def main():
    """
    Main training loop demonstrating all RLGS-inspired features.
    """
    print("=" * 60)
    print("QNN with RLGS-Inspired Features")
    print("=" * 60)
    print("\nFeatures:")
    print("1. RLGS-inspired graph-state simplification (fewer CZ gates)")
    print("2. Qtenon-inspired simulated low-latency classical↔quantum loop")
    print("3. Qoncord-inspired restart + promotion scheduling")
    print("4. Q-Edge-inspired edge-mode constraints")
    print("=" * 60)
    
    # Prepare dataset
    print("\nPreparing toy classification dataset (moons)...")
    X_train, X_val, y_train, y_val = prepare_dataset('moons', n_samples=200)
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Initialize QNN
    print("\nInitializing RLGS-QNN...")
    qnn = RLGS_QNN(n_qubits=4, n_layers=2)
    
    # Train with low latency (Qtenon-style)
    print("\nStarting training with Qtenon low-latency mode...")
    qnn = train_qnn(qnn, X_train, y_train, X_val, y_val, epochs=30, batch_size=5, latency_mode='low')
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Results:")
    print("=" * 60)
    
    final_predictions = qnn.predict(X_val)
    final_accuracy = np.mean((final_predictions > 0.5) == (y_val > 0.5))
    print(f"Final Validation Accuracy: {final_accuracy:.4f}")
    
    # Feature summaries
    print(f"\n1. RLGS Graph Simplification:")
    print(f"   - Reduced to {len(qnn.graph_simplifier.analyze_connectivity([(i, i+1) for i in range(qnn.n_qubits - 1)]))} CZ gates")
    print(f"   - Circuit depth: {qnn.graph_simplifier.get_simplified_circuit_depth()}")
    
    print(f"\n2. Qtenon Low-Latency Loop:")
    print(f"   - Average latency: {qnn.low_latency_loop.get_average_latency():.2f} ms")
    print(f"   - Total loop executions: {len(qnn.low_latency_loop.loop_times)}")
    
    print(f"\n3. Qoncord Scheduler:")
    print(f"   - Best loss achieved: {qnn.scheduler.best_loss:.4f}")
    print(f"   - Final learning rate: {qnn.training_history['learning_rate'][-1]:.6f}")
    
    print(f"\n4. Q-Edge Constraints:")
    constraints = qnn.edge_constraints.get_constraint_satisfaction()
    print(f"   - Gate utilization: {constraints['gate_utilization']:.2%}")
    print(f"   - Depth utilization: {constraints['depth_utilization']:.2%}")
    print(f"   - Within constraints: {constraints['within_constraints']}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
