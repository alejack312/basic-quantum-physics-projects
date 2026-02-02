# TME8 Quantum Nonlocality and Contextuality - Tests and Benchmarks

This directory contains comprehensive tests and benchmarks for quantum nonlocality and contextuality demonstrations, including the Mermin-Peres Magic Square and GHZ Paradox, suitable for a quantum software engineering portfolio.

## Overview

This module demonstrates two fundamental quantum phenomena:
1. **Mermin-Peres Magic Square**: A quantum contextuality demonstration showing that quantum measurements cannot be explained by non-contextual hidden variable theories.
2. **GHZ Paradox**: A quantum nonlocality demonstration showing that quantum correlations cannot be explained by local hidden variable theories.

## Test Suite

### Unit Tests

#### `tests/test_tme8_magic_square.py`
Comprehensive unit tests for the Mermin-Peres Magic Square:

- **Pauli Matrices Tests**: Verify structure, Hermiticity, unitarity
- **Tensor Product Tests**: Verify multi-qubit operator construction
- **Magic Square Observables Tests**: Verify structure, Hermiticity, row/column properties
- **Projectors Tests**: Verify structure, Hermiticity, idempotence, completeness, orthogonality
- **Bell State Tests**: Verify structure, normalization, entanglement
- **Quantum Strategy Tests**: Verify 100% winning probability, comparison with classical
- **Parity Contradiction Tests**: Verify the classical impossibility

**Coverage**: 24 test cases.

#### `tests/test_tme8_ghz_paradox.py`
Comprehensive unit tests for the GHZ Paradox:

- **GHZ State Tests**: Verify structure, normalization for n qubits
- **GHZ Inputs Tests**: Verify valid input combinations, counting
- **GHZ Function Tests**: Verify winning condition function
- **Measurement Projectors Tests**: Verify X and Y measurement projectors
- **Quantum Strategy Tests**: Verify 100% winning probability for n players
- **Classical Strategy Tests**: Verify maximum classical probability (75%)
- **GHZ Paradox Tests**: Verify quantum correlations and classical contradiction

**Coverage**: 22 test cases.

### Integration Tests

#### `tests/test_tme8_integration.py`
End-to-end integration tests:

- **Magic Square Workflow**: Complete game simulation workflow
- **GHZ Workflow**: Complete game simulation workflow
- **Quantum Advantage**: Demonstrations of quantum advantage over classical
- **Physical Properties**: Entanglement verification, projector validity
- **Generalization**: Tests for n-player generalizations

**Coverage**: 12 integration test cases.

### Running Tests

```bash
# Run all TME8 tests
pytest tests/test_tme8_magic_square.py tests/test_tme8_ghz_paradox.py tests/test_tme8_integration.py -v

# Run specific test file
pytest tests/test_tme8_magic_square.py -v

# Run with coverage
pytest tests/test_tme8_magic_square.py tests/test_tme8_ghz_paradox.py tests/test_tme8_integration.py --cov=TME8_functions
```

## Benchmarks

### `bench/bench_tme8.py`

Performance benchmarks for quantum nonlocality operations:

1. **Tensor Products**
   - Two-qubit: ~1-2 μs per operation
   - Three-qubit: ~2-3 μs per operation

2. **Projector Construction**
   - Single-qubit: ~5-10 μs per construction
   - Two-qubit: ~15-20 μs per construction

3. **Magic Square Operations**
   - Get observables: ~1-2 μs per operation
   - Build all projectors: ~200-300 μs per operation
   - Quantum strategy: ~1-2 μs per simulation

4. **GHZ State Creation**
   - 3-qubit: < 1 μs per creation
   - 5-qubit: < 1 μs per creation

5. **GHZ Game Operations**
   - Get valid inputs: < 1 μs per operation
   - Quantum strategy: ~5-10 μs per simulation
   - Classical strategy search: ~10-50 ms (one-time computation)

6. **Measurement Projectors**
   - X projectors: < 1 μs per construction
   - Y projectors: < 1 μs per construction

7. **Complete Workflows**
   - Magic square: ~50-100 μs per workflow
   - GHZ: ~10-20 μs per workflow

### Running Benchmarks

```bash
# Run all benchmarks
python bench/bench_tme8.py

# Output includes:
# - Performance metrics for each operation type
# - Average time per operation
# - Scalability analysis
```

## Code Structure

### Key Functions

**Magic Square:**
1. `get_magic_square_observables()`: Get 3x3 grid of two-qubit observables
2. `get_projectors(observable)`: Get projectors for +1 and -1 eigenvalues
3. `build_all_projectors(observables)`: Build all projectors for magic square
4. `create_bell_state()`: Create Bell state |ψ⟩ = (|00⟩ + |11⟩)/√2
5. `quantum_strategy_magic_square()`: Simulate quantum strategy (100% success)
6. `classical_strategy_magic_square()`: Maximum classical probability (8/9)

**GHZ Paradox:**
1. `create_ghz_state(n)`: Create n-qubit GHZ state
2. `get_valid_ghz_inputs(n)`: Get valid input combinations
3. `f_ghz(x, y, z)`: Winning condition function for 3 players
4. `f_ghz_n(*inputs)`: Generalized winning condition for n players
5. `get_x_projectors()`: Projectors for X measurement
6. `get_y_projectors()`: Projectors for Y measurement
7. `simulate_quantum_ghz_game(n)`: Simulate quantum strategy (100% success)
8. `find_max_classical_win_prob_ghz(n)`: Find maximum classical probability

## Test Coverage Summary

- **Total Test Cases**: 58
- **Unit Tests**: 46
- **Integration Tests**: 12
- **Coverage Areas**:
  - All Pauli matrix operations
  - All tensor product operations
  - All projector constructions
  - All state creations (Bell, GHZ)
  - All game simulations
  - Physical property validation
  - Quantum advantage demonstrations

## Key Features Demonstrated

1. **Comprehensive Testing**: Unit, integration, and performance tests
2. **Quantum Nonlocality**: Correct implementation of Bell and GHZ states
3. **Quantum Contextuality**: Correct implementation of magic square
4. **Physical Correctness**: Tests verify quantum mechanical properties
5. **Performance Metrics**: Benchmarks demonstrate efficient implementation
6. **Generalization**: Functions work for arbitrary numbers of qubits/players

## Portfolio Readiness

These tests and benchmarks demonstrate:

- **Software Engineering Best Practices**: Comprehensive test coverage, clear organization
- **Quantum Physics Knowledge**: Correct implementation of nonlocality and contextuality
- **Performance Awareness**: Benchmarking and optimization considerations
- **Quality Assurance**: Edge case testing, integration testing, physical validation
- **Theoretical Understanding**: Demonstrations of quantum advantage over classical

## Physical Significance

### Mermin-Peres Magic Square
- Demonstrates **quantum contextuality**: measurement outcomes depend on context
- Shows that quantum mechanics cannot be explained by non-contextual hidden variables
- Quantum strategy achieves 100% success vs. classical maximum of 8/9

### GHZ Paradox
- Demonstrates **quantum nonlocality**: correlations cannot be explained locally
- Shows that quantum mechanics cannot be explained by local hidden variables
- Quantum strategy achieves 100% success vs. classical maximum of 75%

## Dependencies

- `pytest` for testing framework
- `numpy` for numerical operations

Install with:
```bash
pip install pytest numpy
```

## References

- Mermin-Peres Magic Square: A quantum contextuality demonstration
- GHZ Paradox: A quantum nonlocality demonstration showing stronger-than-Bell correlations
- Both demonstrate fundamental differences between quantum and classical physics
