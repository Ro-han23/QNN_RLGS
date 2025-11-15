"""
Main script to run all experiments and generate figures.
This runs the 5-experiment grid and produces all required plots.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from src.experiments import run_all_experiments
from src.plotting import (
    plot_fidelity_vs_cz, plot_latency_iterations,
    plot_restarts_promotion, plot_edge_vs_cloud_runtime,
    plot_loss_curves
)
from src.qoncord import multi_restart_train
from src.train import qnn_train_for_restart
from src.datasets import prepare_dataset

# Create results directory if it doesn't exist
os.makedirs('results/figures', exist_ok=True)

def main():
    print("="*70)
    print("QNN RLGS Project - Complete Experiment Suite")
    print("="*70)
    
    # Run all 5 experiments
    print("\n[1/6] Running experiment grid...")
    results = run_all_experiments(
        dataset='moons',
        n_samples=200,
        n_qubits=4,
        n_layers=2,
        epochs=30,
        verbose=True
    )
    
    # Plot 1: Fidelity vs CZ gates
    print("\n[2/6] Generating fidelity vs CZ plot...")
    baseline_cz = results['Baseline'].num_cz_gates
    rlgs_cz = results['RLGS'].num_cz_gates
    plot_fidelity_vs_cz(baseline_cz, rlgs_cz)
    
    # Plot 2: Latency vs iterations
    print("\n[3/6] Generating latency-iterations plot...")
    latency_configs = {
        'Baseline\n(High Latency)': {
            'iterations_per_sec': results['Baseline'].iterations_per_sec,
            'latency_ms': 1.0
        },
        'Qtenon\n(Low Latency)': {
            'iterations_per_sec': results['Qtenon'].iterations_per_sec,
            'latency_ms': 0.1
        }
    }
    plot_latency_iterations(latency_configs)
    
    # Plot 3: Restarts promotion (need to run Qoncord experiment separately for detailed data)
    print("\n[4/6] Running detailed Qoncord experiment for restarts plot...")
    X_train, X_val, y_train, y_val = prepare_dataset('moons', n_samples=200)
    train_func = qnn_train_for_restart(X_train, X_val, y_train, y_val, 
                                       n_qubits=4, n_layers=2)
    short_results, final_results = multi_restart_train(
        train_func, N_restarts=5, short_epochs=30, 
        K_promote=2, long_epochs=30, verbose=False
    )
    plot_restarts_promotion(short_results, final_results)
    
    # Plot 4: Edge vs cloud runtime
    print("\n[5/6] Generating edge vs cloud runtime plot...")
    cloud_time = results['Baseline'].total_wall_clock
    edge_time = results['Combined'].total_wall_clock
    cloud_inference = results['Baseline'].inference_time_ms
    edge_inference = results['Combined'].inference_time_ms
    plot_edge_vs_cloud_runtime(edge_time, cloud_time, 
                               edge_inference, cloud_inference)
    
    # Plot 5: Loss curves
    print("\n[6/6] Generating loss curves plot...")
    baseline_losses = results['Baseline'].train_loss_curve
    combined_losses = results['Combined'].train_loss_curve
    plot_loss_curves(baseline_losses, combined_losses)
    
    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print("\nResults by Experiment:")
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Accuracy: {result.final_accuracy:.4f}")
        print(f"  Loss: {result.final_loss:.4f}")
        print(f"  Wall clock: {result.total_wall_clock:.2f}s")
        print(f"  CZ gates: {result.num_cz_gates}")
        print(f"  Inference: {result.inference_time_ms:.2f}ms/sample")
        print(f"  Iterations/sec: {result.iterations_per_sec:.2f}")
    
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    # RLGS effect
    cz_reduction = (1 - rlgs_cz / baseline_cz) * 100
    print(f"\n1. RLGS Graph Simplification:")
    print(f"   CZ gate reduction: {cz_reduction:.1f}%")
    print(f"   ({baseline_cz} → {rlgs_cz} gates)")
    
    # Qtenon effect
    speedup = results['Qtenon'].iterations_per_sec / results['Baseline'].iterations_per_sec
    print(f"\n2. Qtenon Low-Latency:")
    print(f"   Training speedup: {speedup:.2f}x")
    print(f"   ({results['Baseline'].iterations_per_sec:.1f} → {results['Qtenon'].iterations_per_sec:.1f} iter/s)")
    
    # Qoncord effect
    print(f"\n3. Qoncord Multi-Restart:")
    print(f"   Short training best: {min(r.val_loss for r in short_results):.4f}")
    print(f"   Promoted final best: {min(r.val_loss for r in final_results):.4f}")
    print(f"   Improvement: {min(r.val_loss for r in short_results) - min(r.val_loss for r in final_results):.4f}")
    
    # Q-Edge effect
    print(f"\n4. Q-Edge Constraints:")
    print(f"   Edge mode inference: {edge_inference:.2f}ms/sample")
    print(f"   Cloud mode inference: {cloud_inference:.2f}ms/sample")
    
    # Combined effect
    baseline_acc = results['Baseline'].final_accuracy
    combined_acc = results['Combined'].final_accuracy
    acc_improvement = (combined_acc - baseline_acc) * 100
    print(f"\n5. Combined Optimizations:")
    print(f"   Accuracy improvement: {acc_improvement:+.2f}%")
    print(f"   ({baseline_acc:.4f} → {combined_acc:.4f})")
    
    print("\n" + "="*70)
    print("All experiments completed successfully!")
    print("Plots saved to results/figures/")
    print("="*70)


if __name__ == "__main__":
    main()
