# Quantum Decoherence Dynamics - Tests and Benchmarks

This directory contains comprehensive tests and benchmarks for the quantum decoherence dynamics simulation code, suitable for a quantum software engineering portfolio.

## Test Suite

### Unit Tests

#### `tests/test_bloch_trajectories.py`
Comprehensive unit tests for `bloch_trajectories.py`:

- **Phase Damping Tests**: Verify z-component preservation, exponential decay of xy-components, edge cases
- **Amplitude Damping Tests**: Verify convergence to ground state, decay rates, ground state preservation
- **Depolarizing Noise Tests**: Verify scaling behavior, edge cases (p=0, p=1)
- **Trajectory Generation Tests**: Verify correct shape, initial conditions, different channel types
- **Physical Properties Tests**: Verify norm constraints, monotonic behavior
- **Edge Cases**: Zero vectors, empty time steps, etc.

**Coverage**: 25 test cases covering all major functions and edge cases.

#### `tests/test_bloch_noisy_gates.py`
Comprehensive unit tests for `bloch_noisy_gates.py`:

- **Bloch Vector Conversions**: Round-trip conversions, trace preservation, Hermiticity
- **Quantum Gates**: Unitarity, rotation properties, gate application
- **Noise Channels (Kraus Operators)**: Depolarizing, phase damping, amplitude damping
- **Noisy Gates**: Combined gate and noise application
- **Trajectory Generation**: Shape validation, initial conditions, gate sequences
- **Initial States**: Validity of density matrices, correct state definitions
- **Physical Properties**: Trace preservation, Hermiticity preservation

**Coverage**: 30 test cases covering all major functions.

### Integration Tests

#### `tests/test_decoherence_integration.py`
End-to-end integration tests:

- **End-to-End Workflows**: Complete simulation workflows from start to finish
- **Cross-Module Compatibility**: Consistency between different modules
- **Physical Consistency**: Bloch sphere constraints, density matrix validity
- **Energy Conservation**: Physical property verification

**Coverage**: 15 integration test cases.

### Running Tests

```bash
# Run all decoherence tests
pytest tests/test_bloch_trajectories.py tests/test_bloch_noisy_gates.py tests/test_decoherence_integration.py -v

# Run specific test file
pytest tests/test_bloch_trajectories.py -v

# Run with coverage
pytest tests/test_bloch_trajectories.py tests/test_bloch_noisy_gates.py tests/test_decoherence_integration.py --cov=bloch_trajectories --cov=bloch_noisy_gates
```

## Benchmarks

### `bench/bench_decoherence.py`

Performance benchmarks for quantum decoherence operations:

1. **Noise Channel Performance**
   - Phase damping: ~8 μs per operation
   - Amplitude damping: ~13 μs per operation
   - Depolarizing noise: ~3 μs per operation

2. **Bloch Vector Conversions**
   - Density matrix → Bloch vector: ~36 μs per conversion
   - Bloch vector → Density matrix: ~35 μs per conversion
   - Round-trip: ~70 μs per conversion

3. **Noisy Gate Applications**
   - Depolarizing: ~60 μs per gate
   - Phase damping: ~56 μs per gate
   - Amplitude damping: ~33 μs per gate

4. **Trajectory Generation**
   - Small (50 steps): < 1 ms
   - Medium (200 steps): ~4 ms
   - Large (1000 steps): ~11 ms
   - Noisy gates (200 steps): ~23 ms

5. **Kraus Operator Channels**
   - Performance metrics for all noise channel types

### Running Benchmarks

```bash
# Run all benchmarks
python bench/bench_decoherence.py

# Output includes:
# - Performance metrics for each operation type
# - Average time per operation
# - Scalability analysis
```

## Test Coverage Summary

- **Total Test Cases**: 70+
- **Unit Tests**: 55
- **Integration Tests**: 15
- **Coverage Areas**:
  - All noise channel functions
  - All Bloch vector conversion functions
  - All gate operations
  - All trajectory generation functions
  - Physical property validation
  - Edge case handling

## Key Features Demonstrated

1. **Comprehensive Testing**: Unit, integration, and performance tests
2. **Physical Correctness**: Tests verify quantum mechanical properties
3. **Edge Case Handling**: Tests cover boundary conditions and error cases
4. **Performance Metrics**: Benchmarks demonstrate efficient implementation
5. **Documentation**: Well-documented test cases with clear descriptions

## Portfolio Readiness

These tests and benchmarks demonstrate:

- **Software Engineering Best Practices**: Comprehensive test coverage, clear organization
- **Quantum Physics Knowledge**: Correct implementation of quantum channels and operations
- **Performance Awareness**: Benchmarking and optimization considerations
- **Quality Assurance**: Edge case testing, integration testing, physical validation

## Dependencies

- `pytest` for testing framework
- `numpy` for numerical operations
- `matplotlib` for visualization (used by the main code)

Install with:
```bash
pip install pytest numpy matplotlib
```
