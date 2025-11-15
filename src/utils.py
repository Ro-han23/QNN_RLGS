"""
Plotting and metrics utilities.
Functions for visualizing training progress and results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


def plot_training_metrics(training_history: Dict, save_path: str = '../results/figures/training_metrics.png'):
    """
    Plot training metrics including loss, accuracy, learning rate, gate count, and latency.
    
    Args:
        training_history: Dictionary containing training metrics
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('RLGS-QNN Training Metrics', fontsize=16)
    
    # Loss
    axes[0, 0].plot(training_history['loss'])
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(training_history['accuracy'])
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True)
    
    # Learning Rate (Qoncord scheduler)
    axes[0, 2].plot(training_history['learning_rate'])
    axes[0, 2].set_title('Learning Rate (Qoncord Scheduling)')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Learning Rate')
    axes[0, 2].grid(True)
    
    # Gate Count (RLGS simplification)
    axes[1, 0].plot(training_history['gate_count'])
    axes[1, 0].set_title('Gate Count (RLGS Simplified)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Number of Gates')
    axes[1, 0].grid(True)
    
    # Latency (Qtenon loop)
    axes[1, 1].plot(training_history['latency'])
    axes[1, 1].set_title('Loop Latency (Qtenon)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Latency (ms)')
    axes[1, 1].grid(True)
    
    # Leave last subplot for constraint info (will be added separately)
    axes[1, 2].text(0.5, 0.5, 'Constraint Utilization\n(See Q-Edge metrics)', 
                    ha='center', va='center', fontsize=12)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nMetrics plot saved to {save_path}")
    plt.close()


def plot_comparison(baseline_gates: int, optimized_gates: int, 
                   baseline_latency: float, optimized_latency: float,
                   save_path: str = '../results/figures/comparison.png'):
    """
    Plot comparison between baseline and optimized QNN.
    
    Args:
        baseline_gates: Number of gates in baseline circuit
        optimized_gates: Number of gates in optimized circuit
        baseline_latency: Baseline latency in ms
        optimized_latency: Optimized latency in ms
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('RLGS-Inspired Optimizations Comparison', fontsize=16, fontweight='bold')
    
    # Gate count comparison
    ax = axes[0]
    categories = ['Baseline\n(all-to-all)', 'RLGS\n(linear)']
    values = [baseline_gates, optimized_gates]
    bars = ax.bar(categories, values, color=['#ff6b6b', '#51cf66'])
    ax.set_ylabel('Number of CZ Gates')
    ax.set_title('Graph-State Simplification')
    ax.set_ylim([0, max(values) * 1.2])
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}', ha='center', va='bottom', fontweight='bold')
    
    # Latency comparison
    ax = axes[1]
    categories = ['Baseline', 'Qtenon\nOptimized']
    values = [baseline_latency, optimized_latency]
    bars = ax.bar(categories, values, color=['#ff6b6b', '#51cf66'])
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Low-Latency Loop Performance')
    ax.set_ylim([0, max(values) * 1.2])
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}")
    plt.close()


def plot_constraint_satisfaction(gate_utilization: float, depth_utilization: float,
                                 save_path: str = '../results/figures/constraints.png'):
    """
    Plot Q-Edge constraint satisfaction metrics.
    
    Args:
        gate_utilization: Gate utilization ratio (0-1)
        depth_utilization: Depth utilization ratio (0-1)
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    resources = ['Gate\nUtilization', 'Depth\nUtilization']
    utilizations = [gate_utilization, depth_utilization]
    bars = ax.bar(resources, utilizations, color=['#51cf66', '#51cf66'])
    ax.axhline(y=1.0, color='red', linestyle='--', label='Constraint Limit', linewidth=2)
    ax.set_ylabel('Utilization', fontsize=12)
    ax.set_title('Q-Edge Constraint Satisfaction', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.2])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    for bar, val in zip(bars, utilizations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Constraint satisfaction plot saved to {save_path}")
    plt.close()


def print_summary(qnn, final_accuracy: float):
    """
    Print summary of training results and optimizations.
    
    Args:
        qnn: Trained RLGS_QNN instance
        final_accuracy: Final validation accuracy
    """
    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)
    
    print(f"\nFinal Validation Accuracy: {final_accuracy:.2%}")
    
    print(f"\n1. RLGS Graph Simplification:")
    simplified_pairs = len(qnn.graph_simplifier.analyze_connectivity(
        [(i, i+1) for i in range(qnn.n_qubits - 1)]
    ))
    print(f"   • CZ gates: {simplified_pairs} (50% reduction from 6)")
    print(f"   • Circuit depth: {qnn.graph_simplifier.get_simplified_circuit_depth()}")
    
    print(f"\n2. Qtenon Low-Latency Loop:")
    print(f"   • Average latency: {qnn.low_latency_loop.get_average_latency():.2f} ms")
    print(f"   • Total executions: {len(qnn.low_latency_loop.loop_times)}")
    
    print(f"\n3. Qoncord Scheduler:")
    print(f"   • Best loss: {qnn.scheduler.best_loss:.4f}")
    print(f"   • Final LR: {qnn.training_history['learning_rate'][-1]:.6f}")
    
    print(f"\n4. Q-Edge Constraints:")
    constraints = qnn.edge_constraints.get_constraint_satisfaction()
    print(f"   • Gate utilization: {constraints['gate_utilization']:.1%}")
    print(f"   • Depth utilization: {constraints['depth_utilization']:.1%}")
    print(f"   • Within constraints: {'✓ Yes' if constraints['within_constraints'] else '✗ No'}")
    
    print("\n" + "=" * 70)
