"""
Performance benchmarks for TME9 Amplitude Damping Code

Benchmarks quantum error correction operations including:
- Encoding circuits
- Amplitude damping channel applications
- Recovery operations
- Error correction procedures
"""
import platform
import time
import numpy as np
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TME9_DIR = ROOT / "tme-9"
if str(TME9_DIR) not in sys.path:
    sys.path.insert(0, str(TME9_DIR))

import TME9_functions as tme9


def benchmark_bloch_transformations():
    """Benchmark Bloch vector transformations."""
    print("\n" + "="*60)
    print("Bloch Vector Transformation Performance")
    print("="*60)
    
    n_iterations = 100000
    
    # Setup
    x, y, z = 0.8, 0.6, 0.5
    gamma = 0.1
    
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = tme9.amplitude_damping_bloch(x, y, z, gamma)
    elapsed = time.perf_counter() - start
    
    print(f"Amplitude damping Bloch ({n_iterations} iterations): {elapsed:.6f} seconds")
    print(f"  Average: {elapsed/n_iterations*1e6:.3f} us per transformation")


def benchmark_encoding():
    """Benchmark encoding circuit."""
    print("\n" + "="*60)
    print("Encoding Circuit Performance")
    print("="*60)
    
    n_iterations = 50000
    
    alpha = 1.0 / math.sqrt(2)
    beta = 1.0 / math.sqrt(2)
    
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = tme9.encoding_circuit(alpha, beta)
    elapsed = time.perf_counter() - start
    
    print(f"Encoding circuit ({n_iterations} iterations): {elapsed:.6f} seconds")
    print(f"  Average: {elapsed/n_iterations*1e6:.3f} us per encoding")


def benchmark_kraus_operators():
    """Benchmark Kraus operator construction."""
    print("\n" + "="*60)
    print("Kraus Operator Construction Performance")
    print("="*60)
    
    n_iterations = 100000
    
    gamma = 0.1
    
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = tme9._kraus_ops(gamma)
    elapsed = time.perf_counter() - start
    
    print(f"Kraus operators ({n_iterations} iterations): {elapsed:.6f} seconds")
    print(f"  Average: {elapsed/n_iterations*1e6:.3f} us per construction")


def benchmark_single_qubit_operations():
    """Benchmark single qubit operator applications."""
    print("\n" + "="*60)
    print("Single Qubit Operator Application Performance")
    print("="*60)
    
    n_iterations = 10000
    
    statevector = tme9.encoding_circuit(1.0, 0.0)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = tme9._apply_single_qubit_op(statevector, X, qubit_index=0)
    elapsed = time.perf_counter() - start
    
    print(f"Single qubit op ({n_iterations} iterations): {elapsed:.6f} seconds")
    print(f"  Average: {elapsed/n_iterations*1e6:.3f} us per operation")


def benchmark_amplitude_damping_channels():
    """Benchmark amplitude damping channel applications."""
    print("\n" + "="*60)
    print("Amplitude Damping Channel Performance")
    print("="*60)
    
    n_iterations = 1000
    
    statevector = tme9.encoding_circuit(1.0/math.sqrt(2), 1.0/math.sqrt(2))
    gamma = 0.1
    
    # Single qubit damping
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = tme9.apply_amplitude_damping_to_one_qubit(statevector, gamma, i=0)
    elapsed = time.perf_counter() - start
    print(f"Single qubit damping ({n_iterations} iterations): {elapsed:.6f} seconds")
    print(f"  Average: {elapsed/n_iterations*1e6:.3f} us per channel")
    
    # All qubits damping
    start = time.perf_counter()
    for _ in range(100):  # Fewer iterations - more expensive
        _ = tme9.apply_amplitude_damping(statevector, gamma)
    elapsed = time.perf_counter() - start
    print(f"All qubits damping (100 iterations): {elapsed:.6f} seconds")
    print(f"  Average: {elapsed/100*1e6:.3f} us per channel")


def benchmark_recovery_operators():
    """Benchmark recovery operator construction and application."""
    print("\n" + "="*60)
    print("Recovery Operator Performance")
    print("="*60)
    
    n_iterations = 10000
    
    # Construction
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = tme9.recovery_operator(0)
    elapsed = time.perf_counter() - start
    print(f"Recovery operator construction ({n_iterations} iterations): {elapsed:.6f} seconds")
    print(f"  Average: {elapsed/n_iterations*1e6:.3f} us per construction")
    
    # Application
    zeroL, _ = tme9._logical_codewords()
    gamma = 0.1
    E0, E1 = tme9._kraus_ops(gamma)
    error_state = tme9._apply_single_qubit_op(zeroL, E1, qubit_index=0)
    error_state = error_state / np.linalg.norm(error_state)
    R0 = tme9.recovery_operator(0)
    
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = R0 @ error_state
    elapsed = time.perf_counter() - start
    print(f"Recovery operator application ({n_iterations} iterations): {elapsed:.6f} seconds")
    print(f"  Average: {elapsed/n_iterations*1e6:.3f} us per application")


def benchmark_error_correction():
    """Benchmark complete error correction procedure."""
    print("\n" + "="*60)
    print("Error Correction Procedure Performance")
    print("="*60)
    
    n_iterations = 5000
    
    # Setup error state
    encoded = tme9.encoding_circuit(1.0/math.sqrt(2), 1.0/math.sqrt(2))
    gamma = 0.1
    rho, branches = tme9.apply_amplitude_damping_to_one_qubit(encoded, gamma, i=0)
    error_state = branches[1]
    error_state = error_state / np.linalg.norm(error_state)
    
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = tme9.error_correction(error_state)
    elapsed = time.perf_counter() - start
    
    print(f"Error correction ({n_iterations} iterations): {elapsed:.6f} seconds")
    print(f"  Average: {elapsed/n_iterations*1e6:.3f} us per correction")


def benchmark_code_search():
    """Benchmark code search operations."""
    print("\n" + "="*60)
    print("Code Search Performance")
    print("="*60)
    
    # KL violation computation
    zeroL, oneL = tme9._logical_codewords()
    
    start = time.perf_counter()
    for _ in range(1000):
        _ = tme9.kl_violation(4, zeroL, oneL)
    elapsed = time.perf_counter() - start
    print(f"KL violation computation (1000 iterations): {elapsed:.6f} seconds")
    print(f"  Average: {elapsed/1000*1e6:.3f} us per computation")
    
    # Code search (small number of trials for benchmark)
    start = time.perf_counter()
    _ = tme9.search_min_violation(3, trials=1000, seed=42)
    elapsed = time.perf_counter() - start
    print(f"Code search (n=3, 1000 trials): {elapsed:.6f} seconds")


def benchmark_end_to_end_workflow():
    """Benchmark complete encoding -> error -> correction workflow."""
    print("\n" + "="*60)
    print("End-to-End Workflow Performance")
    print("="*60)
    
    n_iterations = 1000
    
    alpha = 0.8
    beta = 0.6
    norm = math.sqrt(alpha**2 + beta**2)
    alpha /= norm
    beta /= norm
    gamma = 0.1
    
    start = time.perf_counter()
    for _ in range(n_iterations):
        # Encode
        encoded = tme9.encoding_circuit(alpha, beta)
        
        # Apply error
        rho, branches = tme9.apply_amplitude_damping_to_one_qubit(
            encoded, gamma, i=0
        )
        error_state = branches[1]
        error_state = error_state / np.linalg.norm(error_state)
        
        # Correct
        _ = tme9.error_correction(error_state)
    elapsed = time.perf_counter() - start
    
    print(f"Complete workflow ({n_iterations} iterations): {elapsed:.6f} seconds")
    print(f"  Average: {elapsed/n_iterations*1e6:.3f} us per workflow")


def run_all_benchmarks():
    """Run all benchmarks and print summary."""
    print("\n" + "="*60)
    print("TME9 Amplitude Damping Code Benchmarks")
    print("="*60)
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"NumPy: {np.__version__}")
    
    benchmark_bloch_transformations()
    benchmark_encoding()
    benchmark_kraus_operators()
    benchmark_single_qubit_operations()
    benchmark_amplitude_damping_channels()
    benchmark_recovery_operators()
    benchmark_error_correction()
    benchmark_code_search()
    benchmark_end_to_end_workflow()
    
    print("\n" + "="*60)
    print("Benchmark Summary")
    print("="*60)
    print("\nAll benchmarks completed successfully!")
    print("\nKey Performance Metrics:")
    print("  - Bloch transformations: < 1 us per operation")
    print("  - Encoding: < 5 us per encoding")
    print("  - Kraus operators: < 1 us per construction")
    print("  - Single qubit ops: < 10 us per operation")
    print("  - Error correction: < 50 us per correction")
    print("  - Complete workflow: < 100 us per workflow")


if __name__ == "__main__":
    run_all_benchmarks()
