"""Glue module that exposes the existing statevector simulator utilities."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
TME3_DIR = ROOT / "tme-3"
if TME3_DIR.exists() and str(TME3_DIR) not in sys.path:
    sys.path.insert(0, str(TME3_DIR))

import ndlists as nd  # noqa: E402
import TME1_functions_solution as tme1  # noqa: E402
import TME2_functions_solution as tme2  # noqa: E402


SQRT2_INV = 2 ** -0.5


def ket(values: list[complex]) -> nd.ndlist:
    return tme1._ket(values)


def tensor_product(a: nd.ndlist, b: nd.ndlist) -> nd.ndlist:
    return tme2.tensor_product(a, b)


def apply_operator(operator: nd.ndlist, state: nd.ndlist) -> nd.ndlist:
    return tme1._matmul(operator, state)


def gate_identity() -> nd.ndlist:
    return nd.ndlist([[1, 0], [0, 1]])


def gate_x() -> nd.ndlist:
    return nd.ndlist([[0, 1], [1, 0]])


def gate_h() -> nd.ndlist:
    return nd.ndlist([
        [SQRT2_INV, SQRT2_INV],
        [SQRT2_INV, -SQRT2_INV],
    ])


def gate_cnot() -> nd.ndlist:
    return nd.ndlist([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ])


def basis_state(index: int, n_qubits: int) -> nd.ndlist:
    dimension = 2 ** n_qubits
    if index < 0 or index >= dimension:
        raise ValueError("Index out of range for basis state")
    return tme1._ket([1 if i == index else 0 for i in range(dimension)])


def kron_all(operators: list[nd.ndlist]) -> nd.ndlist:
    if not operators:
        raise ValueError("At least one operator is required")
    result = operators[0]
    for op in operators[1:]:
        result = tensor_product(result, op)
    return result


def gate_on_qubit(gate: nd.ndlist, qubit: int, n_qubits: int) -> nd.ndlist:
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    if qubit < 0 or qubit >= n_qubits:
        raise ValueError("qubit index out of range")
    ops = [gate_identity() for _ in range(n_qubits)]
    ops[qubit] = gate
    return kron_all(ops)


projector = tme2.projector
measurement_probability = tme2.measurement_probability
simulate_measurement = tme2.simulate_measurement

__all__ = [
    "nd",
    "tme1",
    "tme2",
    "ket",
    "tensor_product",
    "apply_operator",
    "gate_identity",
    "gate_x",
    "gate_h",
    "gate_cnot",
    "basis_state",
    "kron_all",
    "gate_on_qubit",
    "projector",
    "measurement_probability",
    "simulate_measurement",
]
