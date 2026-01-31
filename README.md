# basic-quantum-physics-projects

A lightweight statevector simulator and benchmarks packaged into a recruiter-ready artifact for validating core quantum circuit behaviors.

## Features
- Pure-Python statevector simulator (no external deps) built on the existing ndlist linear algebra utilities.
- Gate-level correctness tests (norm preservation, Bell correlations, basis sanity checks, deterministic sampling).
- Reproducible benchmark script with runtime table output and environment metadata.
- Minimal glue module (`quantum_simulator.py`) for clean imports and examples.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Minimal usage example
```python
import math
import quantum_simulator as qs

# Build |00>
zero = qs.ket([1, 0])
state = qs.tensor_product(zero, zero)

# Apply H on qubit 0, then CNOT
h_on_0 = qs.gate_on_qubit(qs.gate_h(), 0, 2)
state = qs.apply_operator(h_on_0, state)
state = qs.apply_operator(qs.gate_cnot(), state)

# Inspect amplitudes for |00>, |01>, |10>, |11>
amps = [state[i, 0] for i in range(state.shape[0])]
print("Bell state amplitudes:", amps)
print("Norm:", math.sqrt(sum(abs(a) ** 2 for a in amps)))
```

## Validation
Run the unit tests:
```bash
pytest
```
The test suite checks norm preservation under gates, Bell-state correlations, basis-gate sanity for X/H/CNOT, and deterministic sampling with a fixed RNG seed.

## Benchmarks
Run the benchmark script:
```bash
python bench/run_bench.py
```

Sample output (Python 3.10.19, Linux 6.12.13):

| n_qubits | depth | seconds |
| --- | --- | ---: |
| 2 | 50 | 0.012726 |
| 3 | 50 | 0.037005 |
| 4 | 50 | 0.125383 |

## Repo structure
- `quantum_simulator.py`: Glue module for a clean import surface around the existing simulator.
- `tme-3/`: Source of the original statevector and measurement utilities.
- `tests/`: Pytest suite validating simulator behavior.
- `bench/`: Runtime benchmarks.

## Roadmap
- Add multi-qubit controlled gate builder for arbitrary control/target layouts.
- Introduce optional NumPy backend for speed comparison.
- Expand benchmarks to include entangling layers and noise models.
- Publish performance baselines across Python versions.
- Add CI workflow for tests and benchmarks.
