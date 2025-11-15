"""
Training loop with latency simulation and restarts.
Implements the main training procedure with all optimizations.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple
from .model import RLGS_QNN


def prepare_toy_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare toy classification dataset (Iris binary classification).
    """
    # Load iris dataset
    iris = load_iris()
    X = iris.data[:100, :4]  # First 100 samples (2 classes)
    y = iris.target[:100]    # Binary labels (0 and 1)
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    return X_train, X_val, y_train, y_val


def train_qnn(qnn: RLGS_QNN, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: np.ndarray, 
              y_val: np.ndarray,
              epochs: int = 50, 
              batch_size: int = 5):
    """
    Train the QNN with all RLGS-inspired features.
    Includes latency simulation, restart scheduling, and constraint checking.
    """
    print(f"Training RLGS-QNN with {qnn.n_qubits} qubits, {qnn.n_layers} layers")
    print(f"RLGS graph simplification: {len(qnn.graph_simplifier.analyze_connectivity([(i, i+1) for i in range(qnn.n_qubits - 1)]))} CZ gates")
    
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
    print("\nPreparing toy classification dataset (Iris binary)...")
    X_train, X_val, y_train, y_val = prepare_toy_dataset()
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Initialize QNN
    print("\nInitializing RLGS-QNN...")
    qnn = RLGS_QNN(n_qubits=4, n_layers=2)
    
    # Train
    print("\nStarting training...")
    qnn = train_qnn(qnn, X_train, y_train, X_val, y_val, epochs=30, batch_size=5)
    
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
