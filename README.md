# QNN Edge Project

## Optimizing QNN Training using RLGS, Qtenon, Qoncord, and Q-Edge Ideas (Simulated)

A modular Quantum Neural Network (QNN) implementation demonstrating improvements from multiple quantum computing optimization techniques for toy classification tasks.

## Features

This implementation integrates four key optimizations inspired by recent quantum computing research:

### 1. RLGS-Inspired Graph-State Simplification
- **Reduces CZ gate count** through intelligent graph connectivity analysis
- Analyzes qubit entanglement patterns and simplifies to minimal necessary connections
- Implements linear connectivity for reduced circuit complexity
- **Benefit**: Fewer gates → faster execution and reduced noise

### 2. Qtenon-Inspired Low-Latency Classical↔Quantum Loop
- Simulates fast feedback between classical and quantum components
- Tracks loop execution latency metrics
- Enables rapid parameter updates based on quantum measurements
- **Benefit**: Faster training convergence through tight integration

### 3. Qoncord-Inspired Restart + Promotion Scheduling
- Adaptive learning rate scheduling with periodic restarts
- Promotes (increases) learning rate when loss improves
- Uses cosine annealing with restart cycles
- **Benefit**: Escapes local minima and improves optimization

### 4. Q-Edge-Inspired Edge-Mode Constraints
- Enforces circuit efficiency constraints for resource-limited environments
- Limits maximum gate count and circuit depth
- Tracks constraint satisfaction metrics
- **Benefit**: Ensures circuits are deployable on edge/limited quantum devices

## Architecture

The QNN uses:
- **4-6 qubits** for computation
- **2 layers** of parameterized quantum gates
- **Simplified entanglement** pattern (linear connectivity)
- **Binary classification** on Iris dataset (first 2 classes)

### Circuit Structure
```
Input Encoding (RY gates)
    ↓
Layer 1: RX, RY, RZ rotations + CZ entanglement
    ↓
Layer 2: RX, RY, RZ rotations
    ↓
Measurement (PauliZ expectation values)
```

## Project Structure

```
qnn_edge_project/
├── data/                  # Dataset storage
├── src/                   # Source code modules
│   ├── model.py          # QNN ansatz + state preparation
│   ├── train.py          # Training loop, latency simulation, restarts
│   ├── rlgs_utils.py     # RLGS-inspired graph-state simplifier
│   ├── qoncord.py        # Restart + promotion policy implementation
│   ├── qedge_sim.py      # Edge-mode simulator (CPU/bandwidth constraints)
│   └── utils.py          # Plotting, metrics
├── notebooks/            # Jupyter experiments
│   └── experiments.ipynb
├── results/              # Training outputs
│   └── figures/          # Generated plots
├── report/               # Documentation
│   └── final_report.md   # Comprehensive project report
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Installation

```bash
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PennyLane >= 0.32.0
- NumPy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- PyTorch >= 2.0.0

## Usage

### Quick Start - Training Script

```bash
python src/train.py
```

### Interactive Notebook

```bash
jupyter notebook notebooks/experiments.ipynb
```

### Programmatic Usage

```python
from src import RLGS_QNN, train_qnn, prepare_toy_dataset, plot_training_metrics

# Prepare data
X_train, X_val, y_train, y_val = prepare_toy_dataset()

# Initialize and train
qnn = RLGS_QNN(n_qubits=4, n_layers=2)
qnn = train_qnn(qnn, X_train, y_train, X_val, y_val, epochs=30)

# Visualize results
plot_training_metrics(qnn.training_history)
```

This will:
1. Load and prepare the Iris dataset (binary classification)
2. Initialize the RLGS-QNN with all optimization features
3. Train for 30 epochs
4. Generate training metrics visualization
5. Display final results and feature summaries

### Expected Output

```
Training RLGS-QNN with 4 qubits, 2 layers
RLGS graph simplification: 3 CZ gates

Epoch 0/30 - Loss: X.XXXX, Val Acc: X.XXXX, LR: X.XXXXXX
...
Epoch 25/30 - Loss: X.XXXX, Val Acc: X.XXXX, LR: X.XXXXXX

Training completed!
Average latency: XX.XX ms
Constraint satisfaction: {'gate_utilization': X.XX, ...}
```

## Results Visualization

The training process generates `training_metrics.png` with 6 subplots:

1. **Training Loss** - Loss over epochs
2. **Validation Accuracy** - Classification accuracy
3. **Learning Rate** - Adaptive LR from Qoncord scheduling
4. **Gate Count** - Circuit complexity after RLGS simplification
5. **Loop Latency** - Qtenon loop execution time
6. **Constraint Utilization** - Q-Edge resource usage

## Key Modules

### `src/model.py` - RLGS_QNN
Main QNN class integrating all optimization components:
- Parameterized quantum circuits with state preparation
- RLGS-simplified entanglement patterns
- Training with numerical gradient descent
- Comprehensive metrics tracking

### `src/rlgs_utils.py` - RLGSGraphStateSimplifier
Analyzes and simplifies qubit connectivity graphs:
- Builds adjacency matrices from qubit pairs
- Applies heuristics to minimize CZ gates
- Returns optimized linear connectivity pattern

### `src/qedge_sim.py` - QEdgeConstraints & QtenonLowLatencyLoop
Edge-mode simulation and low-latency loop:
- Enforces resource limits (max gates, depth)
- Simulates classical-quantum feedback latency
- Tracks constraint satisfaction metrics

### `src/qoncord.py` - QoncordScheduler
Adaptive learning rate scheduling:
- Promotes LR on loss improvement
- Periodic restarts with cosine annealing
- Helps escape local minima

### `src/train.py` - Training Loop
Complete training procedure:
- Dataset preparation (Iris binary classification)
- Training loop with all optimizations
- Validation and metrics collection

### `src/utils.py` - Plotting & Metrics
Visualization and reporting utilities:
- Training metrics plots (loss, accuracy, LR, etc.)
- Comparison visualizations
- Summary statistics printing

## Customization

### Adjust QNN Architecture

```python
qnn = RLGS_QNN(n_qubits=6, n_layers=3)
```

### Modify Training Parameters

```python
qnn.train(X_train, y_train, X_val, y_val, 
          epochs=50, batch_size=10)
```

### Tune Optimization Features

```python
# Low-latency loop
low_latency_loop = QtenonLowLatencyLoop(latency_ms=1.0)

# Scheduler
scheduler = QoncordScheduler(base_lr=0.02, restart_period=15)

# Constraints
edge_constraints = QEdgeConstraints(max_gates=60, max_depth=12)
```

## Performance Characteristics

### Without Optimizations (Hypothetical)
- All-to-all CZ connectivity: 6 gates for 4 qubits
- Fixed learning rate: slow convergence
- No latency optimization: slower feedback
- No resource constraints: potentially oversized circuits

### With RLGS-Inspired Optimizations
- Linear CZ connectivity: **3 gates** (50% reduction)
- Adaptive learning rate: **faster convergence**
- Simulated low latency: **rapid parameter updates**
- Edge constraints: **guaranteed deployability**

## Toy Classification Task

Uses the Iris dataset for binary classification:
- **Classes**: Setosa vs Versicolor
- **Features**: 4 (sepal/petal length and width)
- **Training samples**: 70
- **Validation samples**: 30
- **Preprocessing**: Standard scaling

## Extension Ideas

1. **Multi-class classification**: Extend to all 3 Iris classes
2. **Real quantum hardware**: Deploy on IBM Quantum or other providers
3. **Advanced simplification**: Implement more sophisticated graph algorithms
4. **Hyperparameter tuning**: Optimize all configuration parameters
5. **Benchmark comparison**: Compare against classical ML and vanilla QNN

## Citation

If you use this code, please cite the original research papers that inspired these techniques:
- RLGS: Random Local Gate Stabilization
- Qtenon: Low-latency quantum-classical integration
- Qoncord: Quantum optimization with restarts
- Q-Edge: Edge computing for quantum systems

## License

MIT License - Feel free to use and modify for your research and applications.

## Contributing

Contributions welcome! Please open issues or pull requests for:
- Bug fixes
- Performance improvements
- Additional optimization techniques
- Documentation enhancements

## Acknowledgments

This implementation demonstrates the integration of multiple quantum optimization techniques for educational and research purposes. The specific implementations are simplified adaptations inspired by the respective research directions.
