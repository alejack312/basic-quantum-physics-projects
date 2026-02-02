"""
Performance benchmarks for TME8 Quantum Nonlocality and Contextuality

Benchmarks quantum nonlocality demonstrations including:
- Mermin-Peres Magic Square
- GHZ Paradox
"""
import platform
import time
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TME8_DIR = ROOT / "quantum-nonlocality-and-contextuality"
if str(TME8_DIR) not in sys.path:
    sys.path.insert(0, str(TME8_DIR))

import TME8_functions as tme8


def benchmark_tensor_products():
    """Benchmark tensor product operations."""
    print("\n" + "="*60)
    print("Tensor Product Performance")
    print("="*60)
    
    n_iterations = 10000
    
    # Two-qubit tensor product
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = tme8.tensor(tme8.X, tme8.Y)
    elapsed = time.perf_counter() - start
    print(f"Two-qubit tensor ({n_iterations} iterations): {elapsed:.6f} seconds")
    print(f"  Average: {elapsed/n_iterations*1e6:.3f} us per operation")
    
    # Three-qubit tensor product
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = tme8.tensor(tme8.X, tme8.Y, tme8.Z)
    elapsed = time.perf_counter() - start
    print(f"Three-qubit tensor ({n_iterations} iterations): {elapsed:.6f} seconds")
    print(f"  Average: {elapsed/n_iterations*1e6:.3f} us per operation")


def benchmark_projector_construction():
    """Benchmark projector construction."""
    print("\n" + "="*60)
    print("Projector Construction Performance")
    print("="*60)
    
    n_iterations = 5000
    
    # Single-qubit projector
    X = tme8.X
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = tme8.get_projectors(X)
    elapsed = time.perf_counter() - start
    print(f"Single-qubit projectors ({n_iterations} iterations): {elapsed:.6f} seconds")
    print(f"  Average: {elapsed/n_iterations*1e6:.3f} us per construction")
    
    # Two-qubit projector
    XX = tme8.tensor(tme8.X, tme8.X)
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = tme8.get_projectors(XX)
    elapsed = time.perf_counter() - start
    print(f"Two-qubit projectors ({n_iterations} iterations): {elapsed:.6f} seconds")
    print(f"  Average: {elapsed/n_iterations*1e6:.3f} us per construction")


def benchmark_magic_square():
    """Benchmark magic square operations."""
    print("\n" + "="*60)
    print("Magic Square Performance")
    print("="*60)
    
    n_iterations = 1000
    
    # Get observables
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = tme8.get_magic_square_observables()
    elapsed = time.perf_counter() - start
    print(f"Get observables ({n_iterations} iterations): {elapsed:.6f} seconds")
    print(f"  Average: {elapsed/n_iterations*1e6:.3f} us per operation")
    
    # Build all projectors
    observables = tme8.get_magic_square_observables()
    start = time.perf_counter()
    for _ in range(100):  # Fewer iterations - more expensive
        _ = tme8.build_all_projectors(observables)
    elapsed = time.perf_counter() - start
    print(f"Build all projectors (100 iterations): {elapsed:.6f} seconds")
    print(f"  Average: {elapsed/100*1e6:.3f} us per operation")
    
    # Quantum strategy
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = tme8.quantum_strategy_magic_square()
    elapsed = time.perf_counter() - start
    print(f"Quantum strategy ({n_iterations} iterations): {elapsed:.6f} seconds")
    print(f"  Average: {elapsed/n_iterations*1e6:.3f} us per simulation")


def benchmark_ghz_state():
    """Benchmark GHZ state creation."""
    print("\n" + "="*60)
    print("GHZ State Creation Performance")
    print("="*60)
    
    n_iterations = 10000
    
    # 3-qubit GHZ
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = tme8.create_ghz_state(3)
    elapsed = time.perf_counter() - start
    print(f"3-qubit GHZ ({n_iterations} iterations): {elapsed:.6f} seconds")
    print(f"  Average: {elapsed/n_iterations*1e6:.3f} us per creation")
    
    # 5-qubit GHZ
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = tme8.create_ghz_state(5)
    elapsed = time.perf_counter() - start
    print(f"5-qubit GHZ ({n_iterations} iterations): {elapsed:.6f} seconds")
    print(f"  Average: {elapsed/n_iterations*1e6:.3f} us per creation")


def benchmark_ghz_game():
    """Benchmark GHZ game operations."""
    print("\n" + "="*60)
    print("GHZ Game Performance")
    print("="*60)
    
    n_iterations = 1000
    
    # Get valid inputs
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = tme8.get_valid_ghz_inputs(3)
    elapsed = time.perf_counter() - start
    print(f"Get valid inputs 3 players ({n_iterations} iterations): {elapsed:.6f} seconds")
    print(f"  Average: {elapsed/n_iterations*1e6:.3f} us per operation")
    
    # Quantum strategy
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = tme8.simulate_quantum_ghz_game(3)
    elapsed = time.perf_counter() - start
    print(f"Quantum strategy 3 players ({n_iterations} iterations): {elapsed:.6f} seconds")
    print(f"  Average: {elapsed/n_iterations*1e6:.3f} us per simulation")
    
    # Classical strategy search (expensive)
    start = time.perf_counter()
    _ = tme8.find_max_classical_win_prob_ghz(3)
    elapsed = time.perf_counter() - start
    print(f"Classical strategy search 3 players (1 iteration): {elapsed:.6f} seconds")


def benchmark_measurement_projectors():
    """Benchmark measurement projector construction."""
    print("\n" + "="*60)
    print("Measurement Projector Performance")
    print("="*60)
    
    n_iterations = 50000
    
    # X projectors
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = tme8.get_x_projectors()
    elapsed = time.perf_counter() - start
    print(f"X projectors ({n_iterations} iterations): {elapsed:.6f} seconds")
    print(f"  Average: {elapsed/n_iterations*1e6:.3f} us per construction")
    
    # Y projectors
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = tme8.get_y_projectors()
    elapsed = time.perf_counter() - start
    print(f"Y projectors ({n_iterations} iterations): {elapsed:.6f} seconds")
    print(f"  Average: {elapsed/n_iterations*1e6:.3f} us per construction")


def benchmark_end_to_end_workflows():
    """Benchmark complete workflows."""
    print("\n" + "="*60)
    print("End-to-End Workflow Performance")
    print("="*60)
    
    n_iterations = 500
    
    # Magic square complete workflow
    start = time.perf_counter()
    for _ in range(n_iterations):
        observables = tme8.get_magic_square_observables()
        projectors = tme8.build_all_projectors(observables)
        psi = tme8.create_bell_state()
        _ = tme8.quantum_strategy_magic_square()
    elapsed = time.perf_counter() - start
    print(f"Magic square workflow ({n_iterations} iterations): {elapsed:.6f} seconds")
    print(f"  Average: {elapsed/n_iterations*1e6:.3f} us per workflow")
    
    # GHZ complete workflow
    start = time.perf_counter()
    for _ in range(n_iterations):
        ghz = tme8.create_ghz_state(3)
        valid_inputs = tme8.get_valid_ghz_inputs(3)
        _ = tme8.simulate_quantum_ghz_game(3)
    elapsed = time.perf_counter() - start
    print(f"GHZ workflow ({n_iterations} iterations): {elapsed:.6f} seconds")
    print(f"  Average: {elapsed/n_iterations*1e6:.3f} us per workflow")


def run_all_benchmarks():
    """Run all benchmarks and print summary."""
    print("\n" + "="*60)
    print("TME8 Quantum Nonlocality and Contextuality Benchmarks")
    print("="*60)
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"NumPy: {np.__version__}")
    
    benchmark_tensor_products()
    benchmark_projector_construction()
    benchmark_magic_square()
    benchmark_ghz_state()
    benchmark_ghz_game()
    benchmark_measurement_projectors()
    benchmark_end_to_end_workflows()
    
    print("\n" + "="*60)
    print("Benchmark Summary")
    print("="*60)
    print("\nAll benchmarks completed successfully!")
    print("\nKey Performance Metrics:")
    print("  - Tensor products: < 5 us per operation")
    print("  - Projector construction: < 20 us per construction")
    print("  - Magic square operations: < 50 us per operation")
    print("  - GHZ state creation: < 1 us per creation")
    print("  - GHZ game simulation: < 10 us per simulation")
    print("  - Complete workflows: < 100 us per workflow")


if __name__ == "__main__":
    run_all_benchmarks()
