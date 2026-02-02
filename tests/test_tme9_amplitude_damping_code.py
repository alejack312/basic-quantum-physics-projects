"""
Comprehensive test suite for TME9 Amplitude Damping Code

Tests quantum error correction for amplitude damping errors, including:
- Bloch vector transformations
- Encoding circuits
- Amplitude damping channel application
- Recovery operators
- Error correction procedures
"""
import math
import numpy as np
import pytest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TME9_DIR = ROOT / "tme-9"
if str(TME9_DIR) not in sys.path:
    sys.path.insert(0, str(TME9_DIR))

import TME9_functions as tme9


class TestAmplitudeDampingBloch:
    """Tests for amplitude damping on Bloch vectors."""
    
    def test_amplitude_damping_bloch_ground_state(self):
        """Test that ground state |0> (z=1) is preserved."""
        x, y, z = 0.0, 0.0, 1.0
        gamma = 0.1
        
        x_new, y_new, z_new = tme9.amplitude_damping_bloch(x, y, z, gamma)
        
        assert math.isclose(x_new, 0.0, abs_tol=1e-10)
        assert math.isclose(y_new, 0.0, abs_tol=1e-10)
        assert math.isclose(z_new, 1.0, abs_tol=1e-10)
    
    def test_amplitude_damping_bloch_excited_state(self):
        """Test that excited state |1> (z=-1) moves toward ground state."""
        x, y, z = 0.0, 0.0, -1.0
        gamma = 0.5
        
        x_new, y_new, z_new = tme9.amplitude_damping_bloch(x, y, z, gamma)
        
        assert z_new > z, "z should increase toward +1"
        expected_z = (1 - gamma) * z + gamma
        assert math.isclose(z_new, expected_z, abs_tol=1e-10)
    
    def test_amplitude_damping_bloch_xy_decay(self):
        """Test that x and y components decay by sqrt(1-gamma)."""
        x, y, z = 1.0, 1.0, 0.0
        gamma = 0.2
        
        x_new, y_new, z_new = tme9.amplitude_damping_bloch(x, y, z, gamma)
        
        expected_scale = math.sqrt(1 - gamma)
        assert math.isclose(x_new, x * expected_scale, abs_tol=1e-10)
        assert math.isclose(y_new, y * expected_scale, abs_tol=1e-10)
    
    def test_amplitude_damping_bloch_at_gamma_zero(self):
        """Test that gamma=0 preserves the state."""
        x, y, z = 0.5, 0.3, 0.4
        x_new, y_new, z_new = tme9.amplitude_damping_bloch(x, y, z, 0.0)
        
        assert math.isclose(x_new, x, abs_tol=1e-10)
        assert math.isclose(y_new, y, abs_tol=1e-10)
        assert math.isclose(z_new, z, abs_tol=1e-10)
    
    def test_amplitude_damping_bloch_at_gamma_one(self):
        """Test that gamma=1 drives state to ground state."""
        x, y, z = 0.8, 0.6, -0.5
        x_new, y_new, z_new = tme9.amplitude_damping_bloch(x, y, z, 1.0)
        
        assert math.isclose(x_new, 0.0, abs_tol=1e-10)
        assert math.isclose(y_new, 0.0, abs_tol=1e-10)
        assert math.isclose(z_new, 1.0, abs_tol=1e-10)


class TestEncodingCircuit:
    """Tests for logical qubit encoding."""
    
    def test_encoding_circuit_normalization(self):
        """Test that encoded state is normalized."""
        alpha = 1.0 / math.sqrt(2)
        beta = 1.0 / math.sqrt(2)
        
        psiL = tme9.encoding_circuit(alpha, beta)
        
        norm = np.linalg.norm(psiL)
        assert math.isclose(norm, 1.0, abs_tol=1e-10)
    
    def test_encoding_circuit_logical_zero(self):
        """Test encoding of |0> state."""
        alpha = 1.0
        beta = 0.0
        
        psiL = tme9.encoding_circuit(alpha, beta)
        
        # Should be |0_L> = (|0000> + |1111>)/sqrt(2)
        expected = np.zeros(16, dtype=complex)
        expected[0b0000] = 1.0 / math.sqrt(2)
        expected[0b1111] = 1.0 / math.sqrt(2)
        
        np.testing.assert_allclose(psiL, expected, rtol=1e-10)
    
    def test_encoding_circuit_logical_one(self):
        """Test encoding of |1> state."""
        alpha = 0.0
        beta = 1.0
        
        psiL = tme9.encoding_circuit(alpha, beta)
        
        # Should be |1_L> = (|0011> + |1100>)/sqrt(2)
        expected = np.zeros(16, dtype=complex)
        expected[0b0011] = 1.0 / math.sqrt(2)
        expected[0b1100] = 1.0 / math.sqrt(2)
        
        np.testing.assert_allclose(psiL, expected, rtol=1e-10)
    
    def test_encoding_circuit_superposition(self):
        """Test encoding of superposition state."""
        alpha = 0.6
        beta = 0.8
        # Normalize
        norm = math.sqrt(alpha**2 + beta**2)
        alpha /= norm
        beta /= norm
        
        psiL = tme9.encoding_circuit(alpha, beta)
        
        # Check specific components
        assert math.isclose(abs(psiL[0b0000]), alpha / math.sqrt(2), abs_tol=1e-10)
        assert math.isclose(abs(psiL[0b1111]), alpha / math.sqrt(2), abs_tol=1e-10)
        assert math.isclose(abs(psiL[0b0011]), beta / math.sqrt(2), abs_tol=1e-10)
        assert math.isclose(abs(psiL[0b1100]), beta / math.sqrt(2), abs_tol=1e-10)
        
        # Check normalization
        assert math.isclose(np.linalg.norm(psiL), 1.0, abs_tol=1e-10)


class TestKrausOperators:
    """Tests for Kraus operator construction."""
    
    def test_kraus_ops_completeness(self):
        """Test that Kraus operators satisfy completeness relation."""
        gamma = 0.1
        E0, E1 = tme9._kraus_ops(gamma)
        
        # Sum E_k^\dagger E_k should equal identity
        completeness = E0.conj().T @ E0 + E1.conj().T @ E1
        identity = np.eye(2, dtype=complex)
        
        np.testing.assert_allclose(completeness, identity, rtol=1e-10)
    
    def test_kraus_ops_at_gamma_zero(self):
        """Test Kraus operators at gamma=0."""
        E0, E1 = tme9._kraus_ops(0.0)
        
        # E0 should be identity
        identity = np.eye(2, dtype=complex)
        np.testing.assert_allclose(E0, identity, rtol=1e-10)
        
        # E1 should be zero
        zero = np.zeros((2, 2), dtype=complex)
        np.testing.assert_allclose(E1, zero, rtol=1e-10)
    
    def test_kraus_ops_structure(self):
        """Test structure of Kraus operators."""
        gamma = 0.2
        E0, E1 = tme9._kraus_ops(gamma)
        
        # E0 should be diagonal
        assert math.isclose(E0[0, 0], 1.0, abs_tol=1e-10)
        assert math.isclose(E0[0, 1], 0.0, abs_tol=1e-10)
        assert math.isclose(E0[1, 0], 0.0, abs_tol=1e-10)
        assert math.isclose(E0[1, 1], math.sqrt(1 - gamma), abs_tol=1e-10)
        
        # E1 should have only (0,1) element
        assert math.isclose(E1[0, 0], 0.0, abs_tol=1e-10)
        assert math.isclose(E1[0, 1], math.sqrt(gamma), abs_tol=1e-10)
        assert math.isclose(E1[1, 0], 0.0, abs_tol=1e-10)
        assert math.isclose(E1[1, 1], 0.0, abs_tol=1e-10)


class TestSingleQubitOperator:
    """Tests for applying operators to single qubits."""
    
    def test_apply_single_qubit_op_identity(self):
        """Test applying identity operator."""
        statevector = tme9.encoding_circuit(1.0, 0.0)
        I = np.eye(2, dtype=complex)
        
        result = tme9._apply_single_qubit_op(statevector, I, qubit_index=0)
        
        np.testing.assert_allclose(result, statevector, rtol=1e-10)
    
    def test_apply_single_qubit_op_preserves_norm(self):
        """Test that operator application preserves norm (for unitary ops)."""
        statevector = tme9.encoding_circuit(0.6, 0.8)
        norm_initial = np.linalg.norm(statevector)
        
        # Apply X gate (bit flip)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        result = tme9._apply_single_qubit_op(statevector, X, qubit_index=0)
        norm_final = np.linalg.norm(result)
        
        assert math.isclose(norm_initial, norm_final, abs_tol=1e-10)


class TestAmplitudeDampingChannel:
    """Tests for amplitude damping channel application."""
    
    def test_apply_amplitude_damping_preserves_trace(self):
        """Test that channel preserves trace of density matrix."""
        statevector = tme9.encoding_circuit(1.0/math.sqrt(2), 1.0/math.sqrt(2))
        gamma = 0.1
        
        rho, _ = tme9.apply_amplitude_damping(statevector, gamma)
        
        trace = np.trace(rho)
        assert math.isclose(trace, 1.0, abs_tol=1e-10)
    
    def test_apply_amplitude_damping_hermitian(self):
        """Test that output density matrix is Hermitian."""
        statevector = tme9.encoding_circuit(0.8, 0.6)
        gamma = 0.05
        
        rho, _ = tme9.apply_amplitude_damping(statevector, gamma)
        
        np.testing.assert_allclose(rho, rho.conj().T, rtol=1e-10)
    
    def test_apply_amplitude_damping_to_one_qubit(self):
        """Test applying damping to single qubit."""
        statevector = tme9.encoding_circuit(1.0, 0.0)
        gamma = 0.1
        
        rho, branches = tme9.apply_amplitude_damping_to_one_qubit(
            statevector, gamma, i=0
        )
        
        assert rho.shape == (16, 16)
        assert len(branches) == 2
        assert math.isclose(np.trace(rho), 1.0, abs_tol=1e-10)
    
    def test_apply_amplitude_damping_branches(self):
        """Test that branches are correctly computed."""
        statevector = tme9.encoding_circuit(1.0, 0.0)
        gamma = 0.1
        
        _, branches = tme9.apply_amplitude_damping(statevector, gamma)
        
        # Should have 2^4 = 16 branches
        assert len(branches) == 16
        
        # Each branch should be a 16-element statevector
        for branch in branches.values():
            assert branch.shape == (16,)


class TestRecoveryOperator:
    """Tests for recovery operators."""
    
    def test_recovery_operator_shape(self):
        """Test that recovery operator has correct shape."""
        for j in range(4):
            Rj = tme9.recovery_operator(j)
            assert Rj.shape == (16, 16)
    
    def test_recovery_operator_corrects_error(self):
        """Test that recovery operator corrects single-qubit errors."""
        # Create error state: E1 on qubit 0 applied to |0_L>
        zeroL, _ = tme9._logical_codewords()
        gamma = 0.1
        E0, E1 = tme9._kraus_ops(gamma)
        
        # Apply E1 to qubit 0
        error_state = tme9._apply_single_qubit_op(zeroL, E1, qubit_index=0)
        error_state = error_state / np.linalg.norm(error_state)
        
        # Apply recovery
        R0 = tme9.recovery_operator(0)
        recovered = R0 @ error_state
        recovered = recovered / np.linalg.norm(recovered)
        
        # Should recover to |0_L> (up to phase)
        fidelity = abs(np.vdot(zeroL, recovered))**2
        assert fidelity > 0.99, "Recovery should restore logical state"
    
    def test_recovery_operator_orthogonality(self):
        """Test that recovery operators map to orthogonal spaces."""
        R0 = tme9.recovery_operator(0)
        R1 = tme9.recovery_operator(1)
        
        # R_j^\dagger R_j should be projector
        P0 = R0.conj().T @ R0
        P1 = R1.conj().T @ R1
        
        # Should be Hermitian projectors
        np.testing.assert_allclose(P0, P0.conj().T, rtol=1e-10)
        np.testing.assert_allclose(P1, P1.conj().T, rtol=1e-10)
        
        # P0^2 = P0 (idempotent)
        np.testing.assert_allclose(P0 @ P0, P0, rtol=1e-10)


class TestErrorCorrection:
    """Tests for error correction procedure."""
    
    def test_error_correction_preserves_norm(self):
        """Test that error correction produces normalized state."""
        statevector = tme9.encoding_circuit(1.0/math.sqrt(2), 1.0/math.sqrt(2))
        
        # Apply error
        gamma = 0.1
        rho, _ = tme9.apply_amplitude_damping_to_one_qubit(statevector, gamma, i=0)
        
        # Get a statevector from the density matrix (take eigenvector)
        eigenvalues, eigenvectors = np.linalg.eigh(rho)
        error_state = eigenvectors[:, -1]  # Dominant eigenvector
        
        # Correct
        corrected = tme9.error_correction(error_state)
        
        assert math.isclose(np.linalg.norm(corrected), 1.0, abs_tol=1e-10)
    
    def test_error_correction_on_code_space(self):
        """Test that error correction preserves states already in code space."""
        statevector = tme9.encoding_circuit(0.8, 0.6)
        # Normalize
        statevector = statevector / np.linalg.norm(statevector)
        
        corrected = tme9.error_correction(statevector)
        
        # Should remain in code space (high fidelity with original)
        fidelity = abs(np.vdot(statevector, corrected))**2
        assert fidelity > 0.99
    
    def test_error_correction_single_jump(self):
        """Test error correction on single-jump error state."""
        # Start with |0_L>
        zeroL, _ = tme9._logical_codewords()
        gamma = 0.1
        
        # Apply E1 to qubit 0
        E0, E1 = tme9._kraus_ops(gamma)
        error_state = tme9._apply_single_qubit_op(zeroL, E1, qubit_index=0)
        error_state = error_state / np.linalg.norm(error_state)
        
        # Correct
        corrected = tme9.error_correction(error_state)
        
        # Should recover |0_L>
        fidelity = abs(np.vdot(zeroL, corrected))**2
        assert fidelity > 0.95, "Should recover logical state"


class TestLogicalCodewords:
    """Tests for logical codeword states."""
    
    def test_logical_codewords_orthogonal(self):
        """Test that |0_L> and |1_L> are orthogonal."""
        zeroL, oneL = tme9._logical_codewords()
        
        overlap = np.vdot(zeroL, oneL)
        assert abs(overlap) < 1e-10, "Logical codewords should be orthogonal"
    
    def test_logical_codewords_normalized(self):
        """Test that logical codewords are normalized."""
        zeroL, oneL = tme9._logical_codewords()
        
        assert math.isclose(np.linalg.norm(zeroL), 1.0, abs_tol=1e-10)
        assert math.isclose(np.linalg.norm(oneL), 1.0, abs_tol=1e-10)
    
    def test_logical_codewords_structure(self):
        """Test structure of logical codewords."""
        zeroL, oneL = tme9._logical_codewords()
        
        # |0_L> should have components at |0000> and |1111>
        assert abs(zeroL[0b0000]) > 0.5
        assert abs(zeroL[0b1111]) > 0.5
        
        # |1_L> should have components at |0011> and |1100>
        assert abs(oneL[0b0011]) > 0.5
        assert abs(oneL[0b1100]) > 0.5


class TestCodeSearch:
    """Tests for code search functions."""
    
    def test_sigma_minus_on_qubit(self):
        """Test sigma-minus operator construction."""
        n = 3
        j = 1
        
        op = tme9.sigma_minus_on_qubit(n, j)
        
        assert op.shape == (2**n, 2**n)
        # Should be non-zero
        assert np.linalg.norm(op) > 0
    
    def test_kl_violation_zero_for_perfect_code(self):
        """Test that KL violation is zero for known perfect code."""
        # Use the 4-qubit AD code
        zeroL, oneL = tme9._logical_codewords()
        
        # For n=4, should have very low violation
        violation = tme9.kl_violation(4, zeroL, oneL)
        
        assert violation < 1e-10, "Known perfect code should have zero violation"
    
    def test_search_min_violation(self):
        """Test code search function."""
        n = 2
        best, best_pair = tme9.search_min_violation(n, trials=1000, seed=42)
        
        assert best >= 0, "Violation should be non-negative"
        assert best_pair is not None
        v0, v1 = best_pair
        assert v0.shape == (2**n,)
        assert v1.shape == (2**n,)
