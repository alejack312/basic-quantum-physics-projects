"""
TME9 - Amplitude Damping (AD) Code Functions

This module implements quantum error correction for amplitude damping errors,
including encoding, error application, and recovery operations.
"""
import numpy as np
import math
import itertools


def amplitude_damping_bloch(x, y, z, gamma):
    """
    Apply amplitude damping channel to a Bloch vector.
    
    Args:
        x, y, z: Bloch vector coordinates
        gamma: damping probability
        
    Returns:
        Tuple (x', y', z') of new Bloch vector coordinates
    """
    s = math.sqrt(1 - gamma)
    return (s * x, s * y, (1 - gamma) * z + gamma)


def encoding_circuit(alpha, beta):
    """
    Encode a single qubit state into a 4-qubit logical state.
    
    Encodes |psi> = alpha|0> + beta|1> into:
    |psi_L> = alpha/sqrt(2)(|0000> + |1111>) + beta/sqrt(2)(|0011> + |1100>)
    
    Args:
        alpha: amplitude for |0>
        beta: amplitude for |1>
        
    Returns:
        16-element statevector (4 qubits)
    """
    psiL = np.zeros(16, dtype=complex)
    a = alpha / math.sqrt(2)
    b = beta / math.sqrt(2)

    psiL[0b0000] += a
    psiL[0b1111] += a
    psiL[0b0011] += b
    psiL[0b1100] += b
    return psiL


def _kraus_ops(gamma):
    """
    Get Kraus operators for amplitude damping channel.
    
    Args:
        gamma: damping probability
        
    Returns:
        List of Kraus operators [E0, E1]
    """
    E0 = np.array([[1.0, 0.0],
                   [0.0, np.sqrt(1.0 - gamma)]], dtype=complex)
    E1 = np.array([[0.0, np.sqrt(gamma)],
                   [0.0, 0.0]], dtype=complex)
    return [E0, E1]


def _apply_single_qubit_op(statevector, op, qubit_index, n_qubits=4):
    """
    Apply a 2x2 operator to a specific qubit in an n-qubit statevector.
    
    Args:
        statevector: n-qubit statevector (length 2^n)
        op: 2x2 operator matrix
        qubit_index: index of qubit to apply operator to (0 = MSB)
        n_qubits: total number of qubits (default 4)
        
    Returns:
        New statevector after applying operator
    """
    psi = np.asarray(statevector, dtype=complex).reshape([2]*n_qubits)

    # tensordot op (out,in) with psi axis=qubit_index (in)
    out = np.tensordot(op, psi, axes=([1], [qubit_index]))
    # out axes are: (op_out, psi_axes_without_target). Put back to original order.
    axes = list(range(1, n_qubits))               # psi axes after removing target
    out = np.moveaxis(out, 0, qubit_index)        # place op_out into target position

    return out.reshape(-1)


def apply_amplitude_damping(statevector, gamma):
    """
    Apply amplitude damping independently to all 4 qubits.
    
    Args:
        statevector: 4-qubit encoded statevector
        gamma: damping probability
        
    Returns:
        Tuple (rho_out, branches):
        - rho_out: 16x16 density matrix after the channel
        - branches: dict mapping outcome bits (k0,k1,k2,k3) to unnormalized statevectors
    """
    E = _kraus_ops(gamma)

    branches = {}
    rho_out = np.zeros((16, 16), dtype=complex)

    for ks in itertools.product([0, 1], repeat=4):
        psi = np.asarray(statevector, dtype=complex)
        for q, k in enumerate(ks):
            psi = _apply_single_qubit_op(psi, E[k], qubit_index=q, n_qubits=4)
        branches[ks] = psi
        rho_out += np.outer(psi, np.conjugate(psi))

    return rho_out, branches


def apply_amplitude_damping_to_one_qubit(statevector, gamma, i):
    """
    Apply amplitude damping channel to only the i-th qubit.
    
    Args:
        statevector: 4-qubit encoded statevector
        gamma: damping probability
        i: qubit index (0-3)
        
    Returns:
        Tuple (rho_out, branches):
        - rho_out: 16x16 density matrix after the channel
        - branches: list of statevectors [E0|psi>, E1|psi>]
    """
    E = _kraus_ops(gamma)

    branches = []
    rho_out = np.zeros((16, 16), dtype=complex)

    for k in [0, 1]:
        psi_k = _apply_single_qubit_op(statevector, E[k], qubit_index=i, n_qubits=4)
        branches.append(psi_k)
        rho_out += np.outer(psi_k, np.conjugate(psi_k))

    return rho_out, branches


def _basis_state(idx, n=4):
    """Create a computational basis state |idx> for n qubits."""
    v = np.zeros(2**n, dtype=complex)
    v[idx] = 1.0
    return v


def _logical_codewords():
    """
    Get the logical codewords |0_L> and |1_L>.
    
    Returns:
        Tuple (zeroL, oneL) of logical codeword statevectors
    """
    s2 = math.sqrt(2)
    zeroL = (_basis_state(0b0000) + _basis_state(0b1111)) / s2
    oneL = (_basis_state(0b0011) + _basis_state(0b1100)) / s2
    return zeroL, oneL


def recovery_operator(j):
    """
    Get the recovery operator R_j for correcting amplitude damping on qubit j.
    
    Args:
        j: qubit index (0-3)
        
    Returns:
        16x16 recovery operator matrix
    """
    zeroL, oneL = _logical_codewords()

    v_idx = [0b0111, 0b1011, 0b1101, 0b1110][j]  # |v_j>
    w_idx = [0b0100, 0b1000, 0b0001, 0b0010][j]  # |w_j>

    vj = _basis_state(v_idx)
    wj = _basis_state(w_idx)

    # R_j = |0_L><v_j| + |1_L><w_j|
    Rj = np.outer(zeroL, np.conjugate(vj)) + np.outer(oneL, np.conjugate(wj))
    return Rj


def _projector_from_vectors(vectors):
    """
    Create a projector from a list of orthonormal vectors.
    
    Args:
        vectors: list of orthonormal statevectors
        
    Returns:
        Projector matrix
    """
    P = np.zeros((16, 16), dtype=complex)
    for v in vectors:
        P += np.outer(v, np.conjugate(v))
    return P


def _syndrome_projectors():
    """
    Get projectors for code space and error spaces.
    
    Returns:
        Tuple (P, Pjs):
        - P: projector onto code space
        - Pjs: list of projectors onto error spaces for each qubit
    """
    zeroL, oneL = _logical_codewords()
    P = _projector_from_vectors([zeroL, oneL])

    Pjs = []
    v_list = [0b0111, 0b1011, 0b1101, 0b1110]
    w_list = [0b0100, 0b1000, 0b0001, 0b0010]
    for j in range(4):
        vj = _basis_state(v_list[j])
        wj = _basis_state(w_list[j])
        Pjs.append(_projector_from_vectors([vj, wj]))
    return P, Pjs


def error_correction(statevector):
    """
    Apply error correction to a statevector based on syndrome measurement.
    
    Args:
        statevector: 4-qubit statevector (possibly with errors)
        
    Returns:
        Corrected and normalized statevector
    """
    psi = np.asarray(statevector, dtype=complex)
    P, Pjs = _syndrome_projectors()

    # compute weights
    w0 = np.vdot(psi, P @ psi).real
    wj = [np.vdot(psi, Pjs[j] @ psi).real for j in range(4)]

    # choose syndrome: code space vs one of the jump spaces
    best = np.argmax([w0] + wj)

    if best == 0:
        # already in code space (no-jump sector)
        psi_corr = P @ psi
    else:
        j = best - 1
        psi_proj = Pjs[j] @ psi
        Rj = recovery_operator(j)
        psi_corr = Rj @ psi_proj

    # renormalize
    norm = np.linalg.norm(psi_corr)
    if norm > 0:
        psi_corr = psi_corr / norm
    return psi_corr


# Code search functions for testing minimum qubit requirements
def sigma_minus_on_qubit(n, j):
    """
    Create sigma-minus operator on qubit j for n-qubit system.
    
    Args:
        n: number of qubits
        j: qubit index
        
    Returns:
        2^n x 2^n operator matrix
    """
    sm = np.array([[0, 1],
                   [0, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)
    ops = [I]*n
    ops[j] = sm
    op = ops[0]
    for k in range(1, n):
        op = np.kron(op, ops[k])
    return op


def random_codewords(dim):
    """
    Generate random orthonormal codewords.
    
    Args:
        dim: dimension of Hilbert space
        
    Returns:
        Tuple (v0, v1) of orthonormal vectors
    """
    A = (np.random.randn(dim, 2) + 1j*np.random.randn(dim, 2))
    Q, _ = np.linalg.qr(A)  # columns orthonormal
    return Q[:,0], Q[:,1]


def kl_violation(n, v0, v1):
    """
    Compute Knill-Laflamme violation for amplitude damping errors.
    
    Args:
        n: number of qubits
        v0, v1: codeword vectors
        
    Returns:
        Violation measure (0 = perfect code)
    """
    # error set: {I, sigma^-_0,...,sigma^-_{n-1}}
    dim = 2**n
    E = [np.eye(dim, dtype=complex)] + [sigma_minus_on_qubit(n, j) for j in range(n)]

    # KL wants <0|Ea†Eb|1>=0 and <0|Ea†Eb|0> = <1|Ea†Eb|1> for all a,b.
    viol = 0.0
    for Ea in E:
        for Eb in E:
            M = Ea.conj().T @ Eb
            off = np.vdot(v0, M @ v1)                       # should be 0
            diag0 = np.vdot(v0, M @ v0)
            diag1 = np.vdot(v1, M @ v1)
            viol += (abs(off)**2 + abs(diag0 - diag1)**2)
    return float(np.real(viol))


def search_min_violation(n, trials=20000, seed=0):
    """
    Search for minimum KL violation over random codewords.
    
    Args:
        n: number of qubits
        trials: number of random trials
        seed: random seed
        
    Returns:
        Tuple (best_violation, best_pair) of minimum violation and codewords
    """
    np.random.seed(seed)
    dim = 2**n
    best = 1e99
    best_pair = None
    for _ in range(trials):
        v0, v1 = random_codewords(dim)
        val = kl_violation(n, v0, v1)
        if val < best:
            best = val
            best_pair = (v0, v1)
    return best, best_pair
