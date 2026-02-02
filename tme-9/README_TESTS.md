# TME9 Amplitude Damping Code - Tests and Benchmarks

This directory contains comprehensive tests and benchmarks for the Amplitude Damping (AD) quantum error correction code, suitable for a quantum software engineering portfolio.

## Overview

The Amplitude Damping Code is a quantum error correction code designed to protect quantum information from amplitude damping errors. This code uses 4 physical qubits to encode 1 logical qubit and can correct single-qubit amplitude damping errors.

## Test Suite

### Unit Tests

#### `tests/test_tme9_amplitude_damping_code.py`
Comprehensive unit tests covering all major components:

- **Amplitude Damping Bloch Tests**: Verify Bloch vector transformations, ground state preservation, decay rates
- **Encoding Circuit Tests**: Verify normalization, logical state encoding, superposition encoding
- **Kraus Operators Tests**: Verify completeness relation, structure, edge cases
- **Single Qubit Operator Tests**: Verify operator application, norm preservation
- **Amplitude Damping Channel Tests**: Verify trace preservation, Hermiticity, branch computation
- **Recovery Operator Tests**: Verify shape, error correction capability, orthogonality
- **Error Correction Tests**: Verify normalization, code space preservation, single-jump correction
- **Logical Codewords Tests**: Verify orthogonality, normalization, structure
- **Code Search Tests**: Verify KL violation computation, code search functionality

**Coverage**: 30 test cases covering all major functions.

### Integration Tests

#### `tests/test_tme9_integration.py`
End-to-end integration tests:

- **End-to-End Workflow**: Complete encoding -> error -> correction workflows
- **Multiple Error Syndromes**: Error correction for errors on different qubits
- **Physical Properties**: Density matrix validity, recovery operator completeness
- **Code Properties**: Dimension counting, error state orthogonality, minimum qubit requirements

**Coverage**: 9 integration test cases.

### Running Tests

```bash
# Run all TME9 tests
pytest tests/test_tme9_amplitude_damping_code.py tests/test_tme9_integration.py -v

# Run specific test file
pytest tests/test_tme9_amplitude_damping_code.py -v

# Run with coverage
pytest tests/test_tme9_amplitude_damping_code.py tests/test_tme9_integration.py --cov=TME9_functions
```

## Benchmarks

### `bench/bench_tme9.py`

Performance benchmarks for quantum error correction operations:

1. **Bloch Vector Transformations**
   - ~0.6 μs per transformation

2. **Encoding Circuits**
   - ~3.6 μs per encoding

3. **Kraus Operator Construction**
   - ~11 μs per construction

4. **Single Qubit Operations**
   - ~21 μs per operation

5. **Amplitude Damping Channels**
   - Single qubit: ~94 μs per channel
   - All qubits: ~1.9 ms per channel

6. **Recovery Operators**
   - Construction: ~40 μs per operator
   - Application: ~3 μs per application

7. **Error Correction**
   - ~50-100 μs per correction

8. **Complete Workflow**
   - ~100 μs per encode->error->correct cycle

### Running Benchmarks

```bash
# Run all benchmarks
python bench/bench_tme9.py

# Output includes:
# - Performance metrics for each operation type
# - Average time per operation
# - Scalability analysis
```

## Code Structure

### Key Functions

1. **`amplitude_damping_bloch(x, y, z, gamma)`**: Apply amplitude damping to Bloch vector
2. **`encoding_circuit(alpha, beta)`**: Encode single qubit into 4-qubit logical state
3. **`apply_amplitude_damping(statevector, gamma)`**: Apply AD channel to all qubits
4. **`apply_amplitude_damping_to_one_qubit(statevector, gamma, i)`**: Apply AD to single qubit
5. **`recovery_operator(j)`**: Get recovery operator for qubit j
6. **`error_correction(statevector)`**: Apply error correction based on syndrome

### Code Properties

- **Code Space**: 2 dimensions (logical qubit)
- **Error Spaces**: 4 spaces of 2 dimensions each (one per physical qubit)
- **Total Protected Space**: 10 dimensions out of 16
- **Minimum Qubits**: 4 qubits required for exact single-jump correction

## Test Coverage Summary

- **Total Test Cases**: 39
- **Unit Tests**: 30
- **Integration Tests**: 9
- **Coverage Areas**:
  - All encoding/decoding functions
  - All error channel applications
  - All recovery operations
  - Physical property validation
  - Code structure verification
  - Edge case handling

## Key Features Demonstrated

1. **Comprehensive Testing**: Unit, integration, and performance tests
2. **Quantum Error Correction**: Correct implementation of AD code
3. **Physical Correctness**: Tests verify quantum mechanical properties
4. **Performance Metrics**: Benchmarks demonstrate efficient implementation
5. **Code Search**: Functions to verify minimum qubit requirements

## Portfolio Readiness

These tests and benchmarks demonstrate:

- **Software Engineering Best Practices**: Comprehensive test coverage, clear organization
- **Quantum Error Correction Knowledge**: Correct implementation of AD code
- **Performance Awareness**: Benchmarking and optimization considerations
- **Quality Assurance**: Edge case testing, integration testing, physical validation
- **Theoretical Understanding**: Code search functions verify theoretical bounds

## Dependencies

- `pytest` for testing framework
- `numpy` for numerical operations

Install with:
```bash
pip install pytest numpy
```

## References

The Amplitude Damping Code is a well-known quantum error correction code that protects against amplitude damping errors using 4 physical qubits to encode 1 logical qubit. The code can correct single-qubit amplitude damping errors by detecting which qubit experienced the error and applying the appropriate recovery operation.
