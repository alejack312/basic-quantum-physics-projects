import math
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import quantum_simulator as qs


def _flatten_state(state):
    return [state[i, 0] for i in range(state.shape[0])]


def _norm(state):
    return math.sqrt(sum(abs(amplitude) ** 2 for amplitude in _flatten_state(state)))


def test_norm_preservation_single_qubit_gate():
    state = qs.ket([1, 0])
    hadamard = qs.gate_h()
    new_state = qs.apply_operator(hadamard, state)
    assert math.isclose(_norm(state), _norm(new_state), rel_tol=1e-12, abs_tol=1e-12)


def test_basis_sanity_x_h_cnot():
    zero = qs.ket([1, 0])
    one = qs.ket([0, 1])

    x_gate = qs.gate_x()
    h_gate = qs.gate_h()

    assert _flatten_state(qs.apply_operator(x_gate, zero)) == _flatten_state(one)
    assert _flatten_state(qs.apply_operator(x_gate, one)) == _flatten_state(zero)

    expected_plus = [1 / math.sqrt(2), 1 / math.sqrt(2)]
    plus_state = _flatten_state(qs.apply_operator(h_gate, zero))
    assert all(
        math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
        for a, b in zip(plus_state, expected_plus)
    )

    cnot = qs.gate_cnot()
    state_10 = qs.tensor_product(one, zero)
    result = qs.apply_operator(cnot, state_10)
    assert _flatten_state(result) == [0, 0, 0, 1]


def test_bell_state_correlations():
    zero = qs.ket([1, 0])
    state_00 = qs.tensor_product(zero, zero)

    hadamard = qs.gate_h()
    h_on_qubit0 = qs.gate_on_qubit(hadamard, 0, 2)
    after_h = qs.apply_operator(h_on_qubit0, state_00)

    cnot = qs.gate_cnot()
    bell = qs.apply_operator(cnot, after_h)
    amplitudes = _flatten_state(bell)

    expected = [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)]
    assert all(
        math.isclose(abs(a), abs(b), rel_tol=1e-12, abs_tol=1e-12)
        for a, b in zip(amplitudes, expected)
    )


def test_deterministic_sampling_with_seed():
    random.seed(1234)
    plus_state = qs.ket([1 / math.sqrt(2), 1 / math.sqrt(2)])
    projectors = [qs.projector([1, 0]), qs.projector([0, 1])]
    outcomes = qs.simulate_measurement(plus_state, projectors, 12)
    assert outcomes == [1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1]
