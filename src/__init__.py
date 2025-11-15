"""
QNN Edge Project - Source Package
Optimizing QNN training using RLGS, Qtenon, Qoncord, and Q-Edge ideas.
"""

from .model import RLGS_QNN, data_encoding, ansatz
from .train import train_qnn, prepare_toy_dataset, qnode_call_with_latency, compare_latency_modes, qnn_train_for_restart
from .rlgs_utils import RLGSGraphStateSimplifier, remove_low_impact_edges
from .qoncord import QoncordScheduler, RestartResult, multi_restart_train, set_seed
from .qedge_sim import QEdgeConstraints, QtenonLowLatencyLoop, QEdgeSimulator
from .experiments import run_all_experiments, ExperimentConfig, ExperimentResult
from .plotting import (plot_fidelity_vs_cz, plot_latency_iterations, plot_restarts_promotion,
                      plot_edge_vs_cloud_runtime, plot_loss_curves)
from .utils import plot_training_metrics, plot_comparison, plot_constraint_satisfaction, print_summary
from .datasets import (
    prepare_dataset, 
    generate_xor_dataset, 
    load_moons_dataset, 
    load_circles_dataset,
    load_iris_binary_dataset,
    save_dataset,
    load_dataset_from_csv
)

__all__ = [
    'RLGS_QNN',
    'data_encoding',
    'ansatz',
    'train_qnn',
    'prepare_toy_dataset',
    'qnode_call_with_latency',
    'compare_latency_modes',
    'qnn_train_for_restart',
    'RLGSGraphStateSimplifier',
    'remove_low_impact_edges',
    'QoncordScheduler',
    'RestartResult',
    'multi_restart_train',
    'set_seed',
    'QEdgeConstraints',
    'QtenonLowLatencyLoop',
    'QEdgeSimulator',
    'run_all_experiments',
    'ExperimentConfig',
    'ExperimentResult',
    'plot_training_metrics',
    'plot_comparison',
    'plot_constraint_satisfaction',
    'print_summary',
    'plot_fidelity_vs_cz',
    'plot_latency_iterations',
    'plot_restarts_promotion',
    'plot_edge_vs_cloud_runtime',
    'plot_loss_curves',
    'prepare_dataset',
    'generate_xor_dataset',
    'load_moons_dataset',
    'load_circles_dataset',
    'load_iris_binary_dataset',
    'save_dataset',
    'load_dataset_from_csv',
]
