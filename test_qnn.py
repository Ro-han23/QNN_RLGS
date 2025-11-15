"""
Test script to validate QNN RLGS features.
"""

import numpy as np
from qnn_rlgs import (
    RLGSGraphStateSimplifier,
    QtenonLowLatencyLoop,
    QoncordScheduler,
    QEdgeConstraints,
    RLGS_QNN,
    prepare_toy_dataset
)


def test_rlgs_simplifier():
    """Test RLGS graph-state simplification."""
    print("Testing RLGS Graph State Simplifier...")
    simplifier = RLGSGraphStateSimplifier(n_qubits=4)
    
    # Test with all-to-all connectivity
    all_pairs = [(i, j) for i in range(4) for j in range(i+1, 4)]
    optimized = simplifier.analyze_connectivity(all_pairs)
    
    print(f"  Original pairs: {len(all_pairs)} (all-to-all)")
    print(f"  Optimized pairs: {len(optimized)} (simplified)")
    print(f"  Reduction: {(1 - len(optimized)/len(all_pairs))*100:.1f}%")
    assert len(optimized) < len(all_pairs), "Simplification should reduce CZ gates"
    print("  ✓ RLGS simplification working correctly\n")


def test_qtenon_loop():
    """Test Qtenon low-latency loop."""
    print("Testing Qtenon Low-Latency Loop...")
    loop = QtenonLowLatencyLoop(latency_ms=0.5)
    
    # Test quantum-classical feedback
    def quantum_func(data):
        return np.sum(data)
    
    def classical_func(result):
        return result * 2
    
    test_data = np.array([1.0, 2.0, 3.0])
    result = loop.execute_with_feedback(quantum_func, classical_func, test_data)
    
    print(f"  Test result: {result}")
    print(f"  Average latency: {loop.get_average_latency():.2f} ms")
    assert loop.get_average_latency() > 0, "Latency should be tracked"
    print("  ✓ Qtenon low-latency loop working correctly\n")


def test_qoncord_scheduler():
    """Test Qoncord scheduler."""
    print("Testing Qoncord Restart + Promotion Scheduler...")
    scheduler = QoncordScheduler(base_lr=0.01, restart_period=5)
    
    # Simulate training with improving loss
    losses = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    lrs = []
    
    for loss in losses:
        lr = scheduler.get_learning_rate(loss)
        lrs.append(lr)
    
    print(f"  Initial LR: {lrs[0]:.6f}")
    print(f"  Final LR: {lrs[-1]:.6f}")
    print(f"  Best loss tracked: {scheduler.best_loss:.2f}")
    assert scheduler.best_loss == min(losses), "Scheduler should track best loss"
    print("  ✓ Qoncord scheduler working correctly\n")


def test_qedge_constraints():
    """Test Q-Edge constraints."""
    print("Testing Q-Edge Constraint Checker...")
    constraints = QEdgeConstraints(max_gates=50, max_depth=10)
    
    # Test within constraints
    within = constraints.check_constraints(circuit_gates=30, circuit_depth=6)
    print(f"  Circuit (30 gates, depth 6): {'✓ Pass' if within else '✗ Fail'}")
    
    # Test exceeding constraints
    exceeds = constraints.check_constraints(circuit_gates=60, circuit_depth=12)
    print(f"  Circuit (60 gates, depth 12): {'✗ Exceeds' if not exceeds else '✓ Pass'}")
    
    metrics = constraints.get_constraint_satisfaction()
    print(f"  Gate utilization: {metrics['gate_utilization']:.1%}")
    print(f"  Depth utilization: {metrics['depth_utilization']:.1%}")
    assert metrics['gate_utilization'] > 1.0 or not metrics['within_constraints'], \
           "Over-utilization should violate constraints"
    print("  ✓ Q-Edge constraints working correctly\n")


def test_qnn_integration():
    """Test full QNN integration."""
    print("Testing Full QNN Integration...")
    
    # Create small dataset
    X_train, X_val, y_train, y_val = prepare_toy_dataset()
    
    # Initialize QNN with all features
    qnn = RLGS_QNN(n_qubits=4, n_layers=2)
    
    print(f"  QNN initialized with {qnn.n_qubits} qubits")
    print(f"  Parameter count: {len(qnn.params)}")
    
    # Test prediction
    predictions = qnn.predict(X_val[:5])
    print(f"  Sample predictions shape: {predictions.shape}")
    assert len(predictions) == 5, "Should predict for all samples"
    
    # Check all components are initialized
    assert qnn.graph_simplifier is not None, "RLGS simplifier should be initialized"
    assert qnn.low_latency_loop is not None, "Qtenon loop should be initialized"
    assert qnn.scheduler is not None, "Qoncord scheduler should be initialized"
    assert qnn.edge_constraints is not None, "Q-Edge constraints should be initialized"
    
    print("  ✓ All QNN components integrated correctly\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("QNN RLGS Features Validation Tests")
    print("=" * 60 + "\n")
    
    test_rlgs_simplifier()
    test_qtenon_loop()
    test_qoncord_scheduler()
    test_qedge_constraints()
    test_qnn_integration()
    
    print("=" * 60)
    print("All Tests Passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
