import platform
import random
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import quantum_simulator as qs


def build_layer(rng, n_qubits):
    gates = []
    for _ in range(n_qubits):
        gate = qs.gate_h() if rng.random() < 0.5 else qs.gate_x()
        gates.append(gate)
    return qs.kron_all(gates)


def run_bench():
    rng = random.Random(0)
    configs = [
        (2, 50),
        (3, 50),
        (4, 50),
    ]

    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print("\nRuntime (seconds)")
    print("| n_qubits | depth | seconds |")
    print("| --- | --- | ---: |")

    for n_qubits, depth in configs:
        state = qs.basis_state(0, n_qubits)
        start = time.perf_counter()
        for _ in range(depth):
            layer = build_layer(rng, n_qubits)
            state = qs.apply_operator(layer, state)
        elapsed = time.perf_counter() - start
        print(f"| {n_qubits} | {depth} | {elapsed:.6f} |")


if __name__ == "__main__":
    run_bench()
