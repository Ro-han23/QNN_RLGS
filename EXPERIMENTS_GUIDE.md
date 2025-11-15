# QNN RLGS Experiments Guide

This guide explains how to run the complete experimental suite and interpret the results.

## Quick Start

```bash
# Run all 5 experiments and generate all plots
python run_experiments.py
```

This will:
1. Run 5 different QNN configurations
2. Generate 5 plots in `results/figures/`
3. Print comprehensive summary with key findings

## Experiment Grid

### 1. Baseline
- **Configuration**: Naive graph + high latency + no restarts + cloud mode
- **Purpose**: Establish baseline performance without optimizations

### 2. RLGS
- **Configuration**: Reduced graph + high latency + no restarts + cloud mode
- **Purpose**: Measure impact of RLGS graph simplification
- **Expected**: Fewer CZ gates, similar accuracy

### 3. Qtenon
- **Configuration**: Naive graph + low latency + no restarts + cloud mode
- **Purpose**: Measure impact of low-latency quantum-classical loop
- **Expected**: Faster training (higher iterations/sec)

### 4. Qoncord
- **Configuration**: Naive graph + high latency + restarts+promotion + cloud mode
- **Purpose**: Measure impact of multi-restart exploration strategy
- **Expected**: Better final accuracy through exploration

### 5. Combined
- **Configuration**: Reduced graph + low latency + restarts+promotion + edge mode
- **Purpose**: Show cumulative benefit of all optimizations
- **Expected**: Best overall performance with edge deployment constraints

## Generated Plots

### 1. fidelity_vs_cz.png
- **X-axis**: Configuration (Baseline vs RLGS)
- **Y-axes**: CZ gates count (left), Fidelity proxy (right)
- **Key Insight**: RLGS reduces gates → higher fidelity

### 2. latency_iterations.png
- **X-axis**: Latency configuration
- **Y-axis**: Training iterations per second
- **Key Insight**: Qtenon low-latency enables faster training

### 3. restarts_promotion.png
- **X-axis**: Restart runs (R0-R4) and promoted runs (P0-P1)
- **Y-axis**: Validation accuracy
- **Key Insight**: Top candidates promoted for extended training

### 4. edge_vs_cloud_runtime.png
- **Left plot**: Training wall-clock time
- **Right plot**: Inference latency per sample
- **Key Insight**: Edge mode trade-offs (constraints vs deployment)

### 5. loss_curves.png
- **X-axis**: Training epochs
- **Y-axis**: Training loss
- **Key Insight**: Combined approach converges better

## Metrics Collected

For each experiment:
- `final_accuracy`: Final validation accuracy
- `final_loss`: Final training loss
- `train_loss_curve`: Loss at each epoch
- `val_acc_curve`: Validation accuracy at each epoch
- `iterations_per_sec`: Training speed
- `total_wall_clock`: Total training time (seconds)
- `num_cz_gates`: Number of CZ gates in circuit
- `inference_time_ms`: Inference time per sample (milliseconds)

## Hyperparameters

Default configuration:
- **QNN**: 4 qubits, 2 layers
- **Dataset**: make_moons (n_samples=200)
- **Optimizer**: Gradient descent, lr=0.01
- **Batch size**: 8 (cloud), 4 (edge)
- **Epochs**: 30 (standard), 30 short + 200 long (Qoncord)
- **Latency**: 1ms (high), 0.1ms (low)
- **RLGS threshold**: 0.1

## Customization

To modify experiments, edit `run_experiments.py`:

```python
# Change dataset
results = run_all_experiments(
    dataset='circles',  # or 'xor', 'iris'
    n_samples=300,
    ...
)

# Change architecture
results = run_all_experiments(
    n_qubits=6,
    n_layers=3,
    ...
)

# Change training duration
results = run_all_experiments(
    epochs=50,  # More epochs
    ...
)
```

## Expected Runtime

On typical hardware:
- Each experiment: ~30-60 seconds
- Total for 5 experiments: ~5-10 minutes
- Plot generation: <5 seconds

## Troubleshooting

**If experiments are slow:**
- Reduce `n_samples` (e.g., 100 instead of 200)
- Reduce `epochs` (e.g., 20 instead of 30)
- Use smaller batch size

**If plots look noisy:**
- Increase `n_samples`
- Run more epochs
- Average over multiple seeds

**If accuracy is low:**
- Try different dataset (iris usually converges better)
- Increase learning rate (0.02)
- Add more layers or qubits

## Next Steps

After running experiments:

1. **Report (Option C)**: Use the experimental results and plots to write the final report
   - Results are in the returned dictionary
   - Plots are saved in `results/figures/`
   - Summary statistics are printed to console

2. **Slides (Option B)**: Create presentation using the generated plots
   - All 5 plots are high-resolution (300 DPI)
   - Key findings are summarized in console output
   - Use the experiment grid table for methodology slides

## File Structure

```
results/
└── figures/
    ├── fidelity_vs_cz.png
    ├── latency_iterations.png
    ├── restarts_promotion.png
    ├── edge_vs_cloud_runtime.png
    └── loss_curves.png
```

## Citation

If using this code for academic work, cite the four inspirational papers:
- RLGS: [Graph-state simplification for quantum circuits]
- Qtenon: [Low-latency quantum-classical feedback]
- Qoncord: [Restart and promotion scheduling]
- Q-Edge: [Edge-mode quantum computing constraints]
