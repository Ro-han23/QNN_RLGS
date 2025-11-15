"""
RLGS-inspired graph-state simplification utilities.
Implements heuristics to reduce CZ gates through connectivity analysis.
"""

import numpy as np
from typing import List, Tuple


def remove_low_impact_edges(entangler_list: List[Tuple[int, int]], 
                            param_magnitudes: np.ndarray, 
                            threshold: float = 0.1) -> List[Tuple[int, int]]:
    """
    Remove low-impact entangling edges based on parameter magnitudes.
    
    This is a heuristic simplifier to emulate RLGS effect without full RL:
    - Scores each edge by estimated contribution
    - Removes edges between qubits with low parameter magnitudes
    
    Args:
        entangler_list: List of (i, j) qubit pairs for entanglement
        param_magnitudes: Array of parameter magnitudes per wire/layer
        threshold: Minimum score to keep an edge (default: 0.1)
    
    Returns:
        Filtered list of high-impact entangling edges
    """
    keep = []
    for (i, j) in entangler_list:
        # Score based on mean absolute parameter value as importance proxy
        score = (param_magnitudes[i] + param_magnitudes[j]) / 2.0
        if score >= threshold:
            keep.append((i, j))
    return keep


class RLGSGraphStateSimplifier:
    """
    RLGS-inspired graph-state simplification to reduce CZ gates.
    Uses heuristic parameter-magnitude-based analysis to minimize entangling operations.
    """
    
    def __init__(self, n_qubits: int, threshold: float = 0.1):
        self.n_qubits = n_qubits
        self.threshold = threshold
        self.adjacency = np.zeros((n_qubits, n_qubits), dtype=int)
        self.param_magnitudes = None
    
    def update_param_magnitudes(self, params: np.ndarray, n_layers: int):
        """
        Update parameter magnitudes for edge importance scoring.
        
        Args:
            params: Flattened parameter array
            n_layers: Number of circuit layers
        """
        # Compute mean absolute parameter magnitude per qubit
        params_per_qubit = len(params) // (self.n_qubits * n_layers)
        self.param_magnitudes = np.zeros(self.n_qubits)
        
        for i in range(self.n_qubits):
            # Get parameters for this qubit across all layers
            qubit_params = []
            for layer in range(n_layers):
                idx_start = layer * self.n_qubits * params_per_qubit + i * params_per_qubit
                idx_end = idx_start + params_per_qubit
                if idx_end <= len(params):
                    qubit_params.extend(params[idx_start:idx_end])
            
            if qubit_params:
                self.param_magnitudes[i] = np.mean(np.abs(qubit_params))
    
    def analyze_connectivity(self, qubit_pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Analyze and simplify connectivity graph to reduce redundant CZ gates.
        
        Two versions:
        - naive_graph: All entanglers (input qubit_pairs)
        - reduced_graph: Filtered based on parameter magnitudes
        
        Returns optimized list of qubit pairs for entanglement.
        """
        # Build adjacency matrix
        for i, j in qubit_pairs:
            self.adjacency[i, j] = 1
            self.adjacency[j, i] = 1
        
        # If param magnitudes available, use heuristic simplification
        if self.param_magnitudes is not None:
            optimized_pairs = remove_low_impact_edges(
                qubit_pairs, self.param_magnitudes, self.threshold
            )
            # Ensure at least linear connectivity if nothing passes threshold
            if len(optimized_pairs) == 0:
                optimized_pairs = [(i, i+1) for i in range(self.n_qubits - 1)]
        else:
            # Fallback: linear connectivity for minimal entanglement
            optimized_pairs = [(i, i+1) for i in range(self.n_qubits - 1)]
        
        return optimized_pairs
    
    def get_simplified_circuit_depth(self) -> int:
        """Returns estimated circuit depth after simplification."""
        all_pairs = [(i, j) for i in range(self.n_qubits) for j in range(i+1, self.n_qubits)]
        return len(self.analyze_connectivity(all_pairs))
    
    def get_naive_graph(self) -> List[Tuple[int, int]]:
        """Returns all possible entangling edges (naive/unoptimized)."""
        return [(i, j) for i in range(self.n_qubits) for j in range(i+1, self.n_qubits)]
    
    def get_reduced_graph(self, params: np.ndarray, n_layers: int) -> List[Tuple[int, int]]:
        """
        Returns reduced graph after RLGS-inspired simplification.
        
        Args:
            params: Current circuit parameters
            n_layers: Number of layers in circuit
        
        Returns:
            Reduced list of entangling edges
        """
        self.update_param_magnitudes(params, n_layers)
        naive = self.get_naive_graph()
        return self.analyze_connectivity(naive)
