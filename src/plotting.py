"""
Additional plotting functions for experiment results.
Produces the 5 required plots for the final report.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List


def plot_fidelity_vs_cz(baseline_cz: int, reduced_cz: int,
                       save_path: str = 'results/figures/fidelity_vs_cz.png',
                       alpha: float = 0.01):
    """
    Plot fidelity proxy vs number of CZ gates.
    Fidelity proxy = exp(-alpha * num_cz) to simulate noise impact.
    
    Args:
        baseline_cz: Number of CZ gates in baseline (naive graph)
        reduced_cz: Number of CZ gates in RLGS-reduced graph
        save_path: Path to save the figure
        alpha: Noise parameter for fidelity proxy
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate fidelity proxies
    fidelity_baseline = np.exp(-alpha * baseline_cz)
    fidelity_reduced = np.exp(-alpha * reduced_cz)
    
    # Bar plot
    categories = ['Baseline\n(Naive Graph)', 'RLGS\n(Reduced Graph)']
    cz_counts = [baseline_cz, reduced_cz]
    fidelities = [fidelity_baseline, fidelity_reduced]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax2 = ax.twinx()
    bars1 = ax.bar(x - width/2, cz_counts, width, label='CZ Gates', color='#4c72b0', alpha=0.8)
    bars2 = ax2.bar(x + width/2, fidelities, width, label='Fidelity Proxy', color='#55a868', alpha=0.8)
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Number of CZ Gates', color='#4c72b0', fontsize=12)
    ax2.set_ylabel('Fidelity Proxy', color='#55a868', fontsize=12)
    ax.set_title('Fidelity Proxy vs Number of CZ Gates\n(RLGS Graph Simplification Impact)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.tick_params(axis='y', labelcolor='#4c72b0')
    ax2.tick_params(axis='y', labelcolor='#55a868')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Fidelity vs CZ plot saved to {save_path}")
    plt.close()


def plot_latency_iterations(latency_configs: Dict[str, Dict],
                           save_path: str = 'results/figures/latency_iterations.png'):
    """
    Plot training iterations/sec vs simulated latency.
    
    Args:
        latency_configs: Dict with keys like 'High Latency', 'Low Latency'
                        Each value is dict with 'iterations_per_sec' and 'latency_ms'
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(latency_configs.keys())
    iterations = [latency_configs[k]['iterations_per_sec'] for k in names]
    latencies = [latency_configs[k]['latency_ms'] for k in names]
    
    # Color code by latency
    colors = ['#e74c3c' if lat >= 1.0 else '#2ecc71' for lat in latencies]
    
    bars = ax.bar(names, iterations, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Training Iterations per Second', fontsize=12)
    ax.set_xlabel('Latency Configuration', fontsize=12)
    ax.set_title('Training Speed vs Quantum-Classical Latency\n(Qtenon Low-Latency Impact)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, lat in zip(bars, latencies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f} iter/s\n({lat:.1f}ms)',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Latency-iterations plot saved to {save_path}")
    plt.close()


def plot_restarts_promotion(short_results: list, final_results: list,
                           save_path: str = 'results/figures/restarts_promotion.png'):
    """
    Plot restart accuracy showing short runs and promoted final accuracy.
    
    Args:
        short_results: List of RestartResult from short training phase
        final_results: List of RestartResult from promoted long training
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract data
    n_restarts = len(short_results)
    short_accs = [r.val_accuracy for r in short_results]
    
    # Create x positions
    x_short = np.arange(n_restarts)
    x_final = np.arange(len(final_results))
    
    # Plot short training results
    bars1 = ax.bar(x_short, short_accs, width=0.6, label='Short Training (30 epochs)',
                   color='#3498db', alpha=0.7, edgecolor='black')
    
    # Highlight promoted restarts
    promoted_indices = [short_results.index(r) for r in sorted(short_results, key=lambda r: r.val_loss)[:len(final_results)]]
    for idx in promoted_indices:
        bars1[idx].set_color('#f39c12')
        bars1[idx].set_alpha(0.9)
        bars1[idx].set_edgecolor('black')
        bars1[idx].set_linewidth(2)
    
    # Plot promoted final results (offset to the right)
    offset = n_restarts + 1
    final_accs = [r.val_accuracy for r in final_results]
    bars2 = ax.bar(x_final + offset, final_accs, width=0.6, 
                   label='Promoted Long Training (200 epochs)',
                   color='#2ecc71', alpha=0.9, edgecolor='black', linewidth=2)
    
    # Styling
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_xlabel('Restart / Promoted Run', fontsize=12)
    ax.set_title('Qoncord Multi-Restart with Promotion\n(Exploration → Exploitation Strategy)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(list(x_short) + list(x_final + offset))
    ax.set_xticklabels([f'R{i}' for i in range(n_restarts)] + [f'P{i}' for i in range(len(final_results))])
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Restarts-promotion plot saved to {save_path}")
    plt.close()


def plot_edge_vs_cloud_runtime(edge_time: float, cloud_time: float,
                               edge_inference: float, cloud_inference: float,
                               save_path: str = 'results/figures/edge_vs_cloud_runtime.png'):
    """
    Plot edge vs cloud inference time comparison.
    
    Args:
        edge_time: Total training time in edge mode (seconds)
        cloud_time: Total training time in cloud mode (seconds)
        edge_inference: Inference time per sample in edge mode (ms)
        cloud_inference: Inference time per sample in cloud mode (ms)
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Q-Edge: Edge vs Cloud Performance Comparison', 
                 fontsize=14, fontweight='bold')
    
    # Training time comparison
    ax = axes[0]
    categories = ['Cloud Mode', 'Edge Mode']
    times = [cloud_time, edge_time]
    colors = ['#3498db', '#e74c3c']
    bars = ax.bar(categories, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Training Time (seconds)', fontsize=12)
    ax.set_title('Training Wall-Clock Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Inference time comparison
    ax = axes[1]
    inference_times = [cloud_inference, edge_inference]
    bars = ax.bar(categories, inference_times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Inference Time (ms/sample)', fontsize=12)
    ax.set_title('Inference Latency per Sample', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}ms', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Edge vs cloud runtime plot saved to {save_path}")
    plt.close()


def plot_loss_curves(baseline_losses: list, combined_losses: list,
                    save_path: str = 'results/figures/loss_curves.png'):
    """
    Plot loss curves comparing baseline vs combined approach.
    
    Args:
        baseline_losses: Training loss curve for baseline
        combined_losses: Training loss curve for combined optimizations
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    epochs_baseline = np.arange(len(baseline_losses))
    epochs_combined = np.arange(len(combined_losses))
    
    ax.plot(epochs_baseline, baseline_losses, 'o-', label='Baseline (No Optimizations)', 
            color='#e74c3c', linewidth=2, markersize=6, alpha=0.8)
    ax.plot(epochs_combined, combined_losses, 's-', label='Combined (RLGS+Qtenon+Qoncord+Q-Edge)', 
            color='#2ecc71', linewidth=2, markersize=6, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Training Loss Curves: Baseline vs Combined Optimizations', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add annotations for final losses
    if len(baseline_losses) > 0:
        final_baseline = baseline_losses[-1]
        ax.annotate(f'Final: {final_baseline:.4f}',
                   xy=(len(baseline_losses)-1, final_baseline),
                   xytext=(len(baseline_losses)-1, final_baseline + 0.05),
                   fontsize=10, ha='right', color='#e74c3c', fontweight='bold')
    
    if len(combined_losses) > 0:
        final_combined = combined_losses[-1]
        ax.annotate(f'Final: {final_combined:.4f}',
                   xy=(len(combined_losses)-1, final_combined),
                   xytext=(len(combined_losses)-1, final_combined - 0.05),
                   fontsize=10, ha='right', color='#2ecc71', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Loss curves plot saved to {save_path}")
    plt.close()
