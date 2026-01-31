import ndlists as nd

# import TME1_functions as tme1  # Uncomment and change the module name if you have TME1 functions from your previous work or solutions. YOU NEED TO UNCOMMENT THIS LINE TO USE TME1 FUNCTIONS
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tme-1'))
import TME1_functions as tme1

from math import sqrt, pi, cos, sin, tan, exp
from typing import List

import random



# Exercise 1
def tensor_product(A: nd.ndlist, B: nd.ndlist) -> nd.ndlist:
    """
    Compute the tensor product of two ndlists recursively.
    :param A:
    :param B:
    :return: The tensor product A ⊗ B as a ndlist.
    """
    # Both are matrices - tensor product of matrices
    result_list = []
    for i in range(len(A)):
        for j in range(len(B)):
            row = []
            for k in range(len(A[i])):
                for l in range(len(B[j])):
                    row.append(A[i][k] * B[j][l])
            result_list.append(row)
    result = nd.ndlist(result_list)
    result.shape = (len(A) * len(B), len(A[0]) * len(B[0]))
    return result

# Test the tensor_product function
v0 = tme1._ket([1, 0])  # |0>
v1 = tme1._ket([0, 1])  # |1>

tp = tensor_product(v0, v1)  # |0> ⊗ |1>
print("Tensor product |0> ⊗ |1>:\n", tp, tp.shape)

# Bell states
bell_00 = (1 / sqrt(2)) * (tensor_product(v0, v0) + tensor_product(v1, v1))
bell_01 = (1 / sqrt(2)) * (tensor_product(v0, v1) + tensor_product(v1, v0))
bell_10 = (1 / sqrt(2)) * (tensor_product(v0, v0) - tensor_product(v1, v1))
bell_11 = (1 / sqrt(2)) * (tensor_product(v0, v1) - tensor_product(v1, v0))

print("Bell state |Φ+>:\n", bell_00, bell_00.shape)
print("Bell state |Φ->:\n", bell_10, bell_10.shape)
print("Bell state |Ψ+>:\n", bell_01, bell_01.shape)
print("Bell state |Ψ->:\n", bell_11, bell_11.shape)

# Verify that the tensor product also holds for quantum gates operators.
I = nd.ndlist([[1, 0],
            [0, 1]])
X = nd.ndlist([[0, 1],
            [1, 0]])
Y = nd.ndlist([[0, -1j],
            [1j, 0]])
Z = nd.ndlist([[1, 0],
            [0, -1]])

# Example: X ⊗ I
X_I = tensor_product(X, I)
print("X ⊗ I:\n", X_I)

# Example: Z ⊗ Y
Z_Y = tensor_product(Z, Y)
print("Z ⊗ Y:\n", Z_Y)

# CNOT gate using tensor products
m0 = tme1._matrix([v0, tme1._zeros(2, 1)])  # |0><0|
m1 = tme1._matrix([tme1._zeros(2, 1), v1])  # |1><1|

CNOT = tensor_product(m0, I) + tensor_product(m1, X)
print("CNOT gate:\n", CNOT, CNOT.shape)

# Define |+> state
v_plus = nd.ndlist([[1/sqrt(2)], [1/sqrt(2)]])  # |+> = (|0> + |1>) / sqrt(2)
print(v_plus, v_plus.shape)
# Initial state |+> ⊗ |0>
initial_state = tensor_product(v_plus, v0)
print("Initial state |+> ⊗ |0>:\n", initial_state, initial_state.shape)

# Apply CNOT gate
final_state = tme1._matmul(CNOT, initial_state)
print("Final state after CNOT:\n", final_state, final_state.shape)

# Define |+> state
v_plus = nd.ndlist([[1/sqrt(2)], [1/sqrt(2)]])  # |+> = (|0> + |1>) / sqrt(2)
print(v_plus, v_plus.shape)
# Initial state |+> ⊗ |0>
initial_state = tensor_product(v_plus, v0)
print("Initial state |+> ⊗ |0>:\n", initial_state, initial_state.shape)

# Apply CNOT gate
final_state = tme1._matmul(CNOT, initial_state)
print("Final state after CNOT:\n", final_state, final_state.shape)

# Exercise 2
def projector(state: nd.ndlist) -> nd.ndlist:
    """
    Construct the projector |ψ><ψ| for a normalized state vector ψ.
    """
    return tme1._matmul(tme1._ket(state),tme1.bra(tme1._ket(state)))

# Define single-qubit projectors

P0 = projector(nd.ndlist([1,0]))  # |0><0|
P1 = projector(nd.ndlist([0,1]))  # |1><1|
print("Projector P0:\n", P0)
print("Projector P1:\n", P1)


def measurement_probability(state: nd.ndlist, projector: nd.ndlist) -> float:
    """
    Compute probability of obtaining outcome associated with projector
    when measuring state.
    """
    # State must be a ket meaning it is a column vector
    if state.shape[1] != 1:
        raise ValueError("State must be a ket (column vector)")
    # Projector must be a matrix
    if projector.shape[0] != projector.shape[1]:
        raise ValueError("Projector must be a square matrix")
    # Projector must be a projector
    
    # Return the inner product of the hermitian of the state and the matrix 
    # multiplication product of the projector and the state
    return tme1._inner(state, tme1._matmul(projector, state))

# Bell state |Φ+> was defined earlier as bell_00
# Define two-qubit projectors
P00 = projector(nd.ndlist([1, 0, 0, 0]))  # P0 ⊗ P0
P01 = projector(nd.ndlist([0, 1, 0, 0]))  # P0 ⊗ P1
P10 = projector(nd.ndlist([0, 0, 1, 0]))  # P1 ⊗ P0
P11 = projector(nd.ndlist([0, 0, 0, 1]))  # P1 ⊗ P1
print("Projector P00:\n", P00)
print("Projector P01:\n", P01)
print("Projector P10:\n", P10)
print("Projector P11:\n", P11)

# Compute probabilities for Bell state |Φ+>
prob_00 = measurement_probability(bell_00, P00)
prob_01 = measurement_probability(bell_00, P01)
prob_10 = measurement_probability(bell_00, P10)
prob_11 = measurement_probability(bell_00, P11)
print("Probabilities of measuring |Φ+> in computational basis:")
print(f"P(00) = {prob_00}, P(01) = {prob_01}, P(10) = {prob_10}, P(11) = {prob_11}")

# (Re)define |+> and |-> states
v_plus = tme1._ket([1/sqrt(2), 1/sqrt(2)]) # |+> = (|0> + |1>) / sqrt(2)
v_minus = tme1._ket([1/sqrt(2), -1/sqrt(2)])  # |-> = (|0> - |1>) / sqrt(2)

# Define single-qubit projectors in the Hadamard basis
P_plus = tme1._matmul(tme1._ket([1/sqrt(2),1/sqrt(2)]), tme1.bra(tme1._ket([1/sqrt(2), 1/sqrt(2)])))  # |+><+|
P_minus = tme1._matmul(tme1._ket([(1/sqrt(2)), (1/sqrt(2))]), tme1.bra(tme1._ket([1/sqrt(2), -1/sqrt(2)]))) # |-><-|
print("Projector P_plus:\n", P_plus)
print("Projector P_minus:\n", P_minus)

# Extend to two-qubit projectors using tensor_product
P_pp = tensor_product(P_plus, P_plus)  # P_plus ⊗ P_plus
P_pm = tensor_product(P_plus, P_minus)  # P_plus ⊗ P_minus
P_mp = tensor_product(P_minus, P_plus)  # P_minus ⊗ P_plus
P_mm = tensor_product(P_minus, P_minus)  # P_minus ⊗ P_minus

print("Projector P_pp:\n", P_pp)
print("Projector P_pm:\n", P_pm)
print("Projector P_mp:\n", P_mp)
print("Projector P_mm:\n", P_mm)

# Simulation of repeated measurements

'''
I spoke with Professor Grosshans and he said that we should be returning a list
of indices of the projectors, where the indices appear in the list with the same
probability as the result obtained by measuring the state with the corresponding 
projector.

For example, simulate_measurement(bell_00, [P_pp, P_pm, P_mp, P_mm] , 1000)
should return a list of 1000 indices, where P_pp appears with the probability of 
measuring bell_00 with P_pp, P_pm appears with the probability of measuring 
bell_00 with P_pm, P_mp appears with the probability of measuring bell_00 with 
P_mp, and P_mm appears with the probability of measuring bell_00 with P_mm.

This should result in about 50/50 P_pp and P_mm.
'''
def simulate_measurement(state: nd.ndlist, projectors: List[nd.ndlist], n: int) -> List[int]:
    """
    Simulate n projective measurements on the given state using the provided projectors.
    Returns a list of measurement outcomes (indices of projectors).
    """
    # Compute probabilities for each projector
    probs = [measurement_probability(state, P) for P in projectors]
    total = sum(probs)
    if total.real <= 0:
        raise ValueError("Total probability is zero; invalid projectors or state.")
    # Normalize to guard against numerical drift
    probs = [p / total for p in probs]

    cdf = []
    s = 0.0
    for p in probs:
        s += p
        cdf.append(s)

    outcomes: List[int] = []
    for _ in range(n):
        r = random()
        for i, c in enumerate(cdf):
            if r <= c.real:
                outcomes.append(i)
                break
    return outcomes


# Simulate measurements of |Φ+> in the Hadamard basis
simulations = simulate_measurement(bell_00, [P_pp, P_pm, P_mp, P_mm] , 1000)
print(simulations)

simulate_measurement(nd.ndlist([[.1],[sqrt(.99)]]),[P0,P1],1000)
    
