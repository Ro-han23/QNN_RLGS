"""
Experimental framework for comparing QNN optimization strategies.
Implements the complete experiment grid with 5 configurations.
"""

import time
import numpy as np
from typing import Dict, List, Tuple
import copy

from .model import RLGS_QNN
from .train import train_qnn
from .datasets import prepare_dataset
from .qoncord import multi_restart_train, RestartResult, set_seed
from .qedge_sim import QEdgeSimulator


class ExperimentConfig:
    """Configuration for a single experiment."""
    
    def __init__(self, name: str, use_rlgs: bool, use_qtenon: bool, 
                 use_qoncord: bool, edge_mode: bool):
        self.name = name
        self.use_rlgs = use_rlgs  # Use reduced graph
        self.use_qtenon = use_qtenon  # Use low latency
        self.use_qoncord = use_qoncord  # Use multi-restart
        self.edge_mode = edge_mode  # Edge vs cloud mode
    
    def __repr__(self):
        return f"ExperimentConfig({self.name})"


class ExperimentResult:
    """Results from a single experiment."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.final_accuracy = 0.0
        self.final_loss = 0.0
        self.train_loss_curve = []
        self.val_loss_curve = []
        self.val_acc_curve = []
        self.iterations_per_sec = 0.0
        self.total_wall_clock = 0.0
        self.num_cz_gates = 0
        self.inference_time_ms = 0.0
        self.qnn = None  # Store trained QNN
    
    def __repr__(self):
        return (f"ExperimentResult({self.config.name}: "
                f"acc={self.final_accuracy:.4f}, "
                f"loss={self.final_loss:.4f}, "
                f"time={self.total_wall_clock:.2f}s)")


def create_experiment_grid() -> List[ExperimentConfig]:
    """
    Create the 5 experiments from the specification:
    1. Baseline: naive_graph + high_latency + no restarts + cloud_mode
    2. RLGS effect: reduced_graph + high_latency + no restarts + cloud_mode
    3. Qtenon effect: naive_graph + low_latency + no restarts + cloud_mode
    4. Qoncord effect: naive_graph + high_latency + restarts+promotion + cloud_mode
    5. Combined: reduced_graph + low_latency + restarts+promotion + edge_mode
    """
    return [
        ExperimentConfig("Baseline", use_rlgs=False, use_qtenon=False, 
                        use_qoncord=False, edge_mode=False),
        ExperimentConfig("RLGS", use_rlgs=True, use_qtenon=False, 
                        use_qoncord=False, edge_mode=False),
        ExperimentConfig("Qtenon", use_rlgs=False, use_qtenon=True, 
                        use_qoncord=False, edge_mode=False),
        ExperimentConfig("Qoncord", use_rlgs=False, use_qtenon=False, 
                        use_qoncord=True, edge_mode=False),
        ExperimentConfig("Combined", use_rlgs=True, use_qtenon=True, 
                        use_qoncord=True, edge_mode=True),
    ]


def run_single_experiment(config: ExperimentConfig,
                         X_train: np.ndarray,
                         y_train: np.ndarray,
                         X_val: np.ndarray,
                         y_val: np.ndarray,
                         n_qubits: int = 4,
                         n_layers: int = 2,
                         epochs: int = 30,
                         verbose: bool = True) -> ExperimentResult:
    """
    Run a single experiment with specified configuration.
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Running Experiment: {config.name}")
        print(f"{'='*70}")
        print(f"  RLGS (reduced graph): {config.use_rlgs}")
        print(f"  Qtenon (low latency): {config.use_qtenon}")
        print(f"  Qoncord (restarts): {config.use_qoncord}")
        print(f"  Edge mode: {config.edge_mode}")
    
    result = ExperimentResult(config)
    start_time = time.time()
    
    # Setup edge simulator
    edge_sim = QEdgeSimulator(mode='edge' if config.edge_mode else 'cloud')
    batch_size = edge_sim.get_batch_size()
    
    # Determine latency mode
    latency_mode = 'low' if config.use_qtenon else 'high'
    
    # Initialize QNN
    qnn = RLGS_QNN(n_qubits=n_qubits, n_layers=n_layers)
    
    # Force RLGS usage if configured
    if config.use_rlgs:
        # Update threshold to ensure reduced graph
        qnn.graph_simplifier.threshold = 0.1
    else:
        # Use full graph
        qnn.graph_simplifier.threshold = 0.0  # Keep all edges
    
    if config.use_qoncord:
        # Use multi-restart training
        from .train import qnn_train_for_restart
        
        train_func = qnn_train_for_restart(
            X_train, X_val, y_train, y_val,
            n_qubits=n_qubits, n_layers=n_layers,
            batch_size=batch_size, latency_mode=latency_mode
        )
        
        short_results, final_results = multi_restart_train(
            train_func,
            N_restarts=5,
            short_epochs=30,
            K_promote=2,
            long_epochs=epochs,
            verbose=verbose
        )
        
        # Get best result
        best = min(final_results, key=lambda r: r.val_loss)
        result.final_accuracy = best.val_accuracy
        result.final_loss = best.val_loss
        result.train_loss_curve = best.training_history.get('loss', [])
        result.val_loss_curve = [best.val_loss]  # Simplified
        result.val_acc_curve = best.training_history.get('accuracy', [])
        
        # Create QNN with best params
        qnn.params = best.params
        result.qnn = qnn
        
    else:
        # Standard training without restarts
        set_seed(42)
        qnn = train_qnn(qnn, X_train, y_train, X_val, y_val,
                       epochs=epochs, batch_size=batch_size,
                       latency_mode=latency_mode)
        
        # Extract results
        result.final_accuracy = qnn.training_history['accuracy'][-1] if qnn.training_history['accuracy'] else 0.0
        result.final_loss = qnn.training_history['loss'][-1] if qnn.training_history['loss'] else 0.0
        result.train_loss_curve = qnn.training_history['loss']
        result.val_acc_curve = qnn.training_history['accuracy']
        result.qnn = qnn
    
    # Calculate metrics
    result.total_wall_clock = time.time() - start_time
    
    # Count CZ gates
    if config.use_rlgs:
        reduced = qnn.graph_simplifier.get_reduced_graph(qnn.params, n_layers)
        result.num_cz_gates = len(reduced) * (n_layers - 1) if n_layers > 1 else 0
    else:
        naive = qnn.graph_simplifier.get_naive_graph()
        result.num_cz_gates = len(naive) * (n_layers - 1) if n_layers > 1 else 0
    
    # Measure inference time
    if len(X_val) > 0:
        inference_start = time.time()
        _ = qnn.predict(X_val[:5])  # Test on 5 samples
        inference_time = (time.time() - inference_start) / 5 * 1000  # ms per sample
        result.inference_time_ms = inference_time
    
    # Iterations per second
    total_iterations = epochs * (len(X_train) // batch_size)
    result.iterations_per_sec = total_iterations / result.total_wall_clock if result.total_wall_clock > 0 else 0
    
    if verbose:
        print(f"\n Results:")
        print(f"  Final accuracy: {result.final_accuracy:.4f}")
        print(f"  Final loss: {result.final_loss:.4f}")
        print(f"  Wall clock: {result.total_wall_clock:.2f}s")
        print(f"  Iterations/sec: {result.iterations_per_sec:.2f}")
        print(f"  CZ gates: {result.num_cz_gates}")
        print(f"  Inference time: {result.inference_time_ms:.2f}ms/sample")
    
    return result


def run_all_experiments(dataset: str = 'moons',
                       n_samples: int = 200,
                       n_qubits: int = 4,
                       n_layers: int = 2,
                       epochs: int = 30,
                       verbose: bool = True) -> Dict[str, ExperimentResult]:
    """
    Run all 5 experiments in the grid.
    
    Returns:
        Dictionary mapping experiment name to ExperimentResult
    """
    print("="*70)
    print("QNN Optimization Experiment Grid")
    print("="*70)
    print(f"Dataset: {dataset} (n_samples={n_samples})")
    print(f"Architecture: {n_qubits} qubits, {n_layers} layers")
    print(f"Epochs: {epochs}")
    print("="*70)
    
    # Prepare dataset
    X_train, X_val, y_train, y_val = prepare_dataset(dataset, n_samples=n_samples)
    print(f"\nDataset prepared: {len(X_train)} train, {len(X_val)} val samples\n")
    
    # Create experiment grid
    experiments = create_experiment_grid()
    
    # Run all experiments
    results = {}
    for exp_config in experiments:
        result = run_single_experiment(
            exp_config, X_train, y_train, X_val, y_val,
            n_qubits=n_qubits, n_layers=n_layers,
            epochs=epochs, verbose=verbose
        )
        results[exp_config.name] = result
    
    # Summary
    if verbose:
        print("\n" + "="*70)
        print("Experiment Summary")
        print("="*70)
        for name, result in results.items():
            print(f"{name:12s}: acc={result.final_accuracy:.4f}, "
                  f"loss={result.final_loss:.4f}, "
                  f"time={result.total_wall_clock:.2f}s, "
                  f"CZ={result.num_cz_gates}")
        print("="*70)
    
    return results
