"""
QNN ansatz and state preparation.
Defines the quantum circuit structure with parameterized gates.
"""

import pennylane as qml
import numpy as np
from typing import List, Tuple
from .rlgs_utils import RLGSGraphStateSimplifier
from .qedge_sim import QEdgeConstraints, QtenonLowLatencyLoop
from .qoncord import QoncordScheduler


def data_encoding(x: np.ndarray, wires: List[int]):
    """
    Amplitude/angle encoding for input data.
    Simple angle encoding: RY rotation on each wire.
    
    Args:
        x: Input feature vector
        wires: List of wire indices to use
    """
    for i, val in enumerate(x):
        if i < len(wires):
            qml.RY(val, wires=wires[i])


def ansatz(params: np.ndarray, wires: List[int], entangling_pairs: List[Tuple[int, int]]):
    """
    Hardware-efficient ansatz with parameterized rotations and entanglers.
    
    Args:
        params: Parameter array (n_layers, n_qubits)
        wires: List of wire indices
        entangling_pairs: List of (i, j) pairs for CNOT/CZ gates
    """
    n_layers = params.shape[0]
    n_wires = len(wires)
    
    for layer in range(n_layers):
        # Single-qubit rotations
        for i in range(n_wires):
            qml.RY(params[layer, i], wires=wires[i])
        
        # Entanglers (CNOT gates) - using RLGS-simplified pairs
        if layer < n_layers - 1:  # No entanglers after last layer
            for (i, j) in entangling_pairs:
                if i < n_wires and j < n_wires:
                    qml.CNOT(wires=[wires[i], wires[j]])


class RLGS_QNN:
    """
    Quantum Neural Network with all RLGS-inspired features integrated.
    """
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Initialize components
        self.graph_simplifier = RLGSGraphStateSimplifier(n_qubits)
        self.low_latency_loop = QtenonLowLatencyLoop(latency_ms=0.5)
        self.scheduler = QoncordScheduler(base_lr=0.01, restart_period=10)
        self.edge_constraints = QEdgeConstraints(max_gates=50, max_depth=10)
        
        # Create quantum device
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
        # Initialize parameters
        self.params = self._initialize_parameters()
        
        # Metrics tracking
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'learning_rate': [],
            'gate_count': [],
            'latency': []
        }
    
    def _initialize_parameters(self) -> np.ndarray:
        """Initialize quantum circuit parameters."""
        # Parameters: one rotation angle per qubit per layer
        # Shape: (n_layers, n_qubits)
        return np.random.randn(self.n_layers, self.n_qubits) * 0.1
    
    def _create_circuit(self, x, params):
        """
        Create quantum circuit with RLGS-inspired simplifications.
        Uses data_encoding + ansatz structure with reduced entanglers.
        """
        @qml.qnode(self.dev, interface='autograd')
        def circuit(inputs, weights):
            # Data encoding (state preparation)
            data_encoding(inputs, wires=range(self.n_qubits))
            
            # Update RLGS simplifier with current parameters
            self.graph_simplifier.update_param_magnitudes(
                weights.flatten(), self.n_layers
            )
            
            # Get RLGS-simplified entangling pattern
            # Start with naive all-to-all, then reduce based on param magnitudes
            naive_pairs = self.graph_simplifier.get_naive_graph()
            entangling_pairs = self.graph_simplifier.analyze_connectivity(naive_pairs)
            
            # Parameterized ansatz with reduced entanglers
            ansatz(weights, wires=range(self.n_qubits), entangling_pairs=entangling_pairs)
            
            # Measurement - single qubit readout for binary classification
            return [qml.expval(qml.PauliZ(0))]
        
        return circuit(x, params)
    
    def _classical_processor(self, quantum_output):
        """Process quantum output classically."""
        # Simple linear combination for classification
        return np.mean(quantum_output)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the QNN with low-latency loop.
        """
        predictions = []
        for x in X:
            # Use low-latency loop for quantum-classical feedback
            pred = self.low_latency_loop.execute_with_feedback(
                lambda data: self._create_circuit(data, self.params),
                self._classical_processor,
                x
            )
            predictions.append(pred)
        
        return np.array(predictions)
    
    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute loss with constraint checking."""
        predictions = self.predict(X)
        # Binary cross-entropy style loss
        loss = np.mean((predictions - y) ** 2)
        
        # Estimate circuit complexity with RLGS simplification
        # Update param magnitudes for accurate entangler count
        self.graph_simplifier.update_param_magnitudes(
            self.params.flatten(), self.n_layers
        )
        naive_pairs = self.graph_simplifier.get_naive_graph()
        reduced_pairs = self.graph_simplifier.analyze_connectivity(naive_pairs)
        
        # Gates: RY encoding + RY per layer + CNOTs
        circuit_gates = self.n_qubits + (self.n_qubits * self.n_layers) + len(reduced_pairs) * (self.n_layers - 1)
        circuit_depth = self.n_layers * 2
        
        # Check constraints
        if not self.edge_constraints.check_constraints(circuit_gates, circuit_depth):
            loss += 10.0  # Penalty for violating constraints
        
        return loss
    
    def _compute_gradients(self, X: np.ndarray, y: np.ndarray, eps: float = 0.01) -> np.ndarray:
        """Compute numerical gradients."""
        grad = np.zeros_like(self.params)
        loss_base = self.compute_loss(X, y)
        
        # Iterate over all parameter indices
        for i in range(self.params.shape[0]):
            for j in range(self.params.shape[1]):
                self.params[i, j] += eps
                loss_plus = self.compute_loss(X, y)
                self.params[i, j] -= eps
                
                grad[i, j] = (loss_plus - loss_base) / eps
        
        return grad
