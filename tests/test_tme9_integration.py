"""
Integration tests for TME9 Amplitude Damping Code

Tests complete workflows: encoding -> error -> correction -> decoding
"""
import numpy as np
import pytest
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TME9_DIR = ROOT / "tme-9"
if str(TME9_DIR) not in sys.path:
    sys.path.insert(0, str(TME9_DIR))

import TME9_functions as tme9


class TestEndToEndWorkflow:
    """End-to-end integration tests."""
    
    def test_complete_encoding_error_correction_workflow(self):
        """Test complete workflow: encode -> error -> correct."""
        # Encode a state
        alpha = 0.8
        beta = 0.6
        norm = math.sqrt(alpha**2 + beta**2)
        alpha /= norm
        beta /= norm
        
        encoded = tme9.encoding_circuit(alpha, beta)
        
        # Apply amplitude damping error to qubit 0
        gamma = 0.1
        rho, branches = tme9.apply_amplitude_damping_to_one_qubit(
            encoded, gamma, i=0
        )
        
        # Get error state (dominant branch)
        error_state = branches[1]  # E1 branch
        error_state = error_state / np.linalg.norm(error_state)
        
        # Correct the error
        corrected = tme9.error_correction(error_state)
        
        # Verify correction worked (should be close to original)
        fidelity = abs(np.vdot(encoded, corrected))**2
        assert fidelity > 0.9, "Error correction should restore encoded state"
    
    def test_multiple_error_syndromes(self):
        """Test error correction for errors on different qubits."""
        encoded = tme9.encoding_circuit(1.0, 0.0)
        gamma = 0.1
        
        for qubit_idx in range(4):
            # Apply error to qubit i
            rho, branches = tme9.apply_amplitude_damping_to_one_qubit(
                encoded, gamma, i=qubit_idx
            )
            
            # Get error state
            error_state = branches[1]
            error_state = error_state / np.linalg.norm(error_state)
            
            # Correct
            corrected = tme9.error_correction(error_state)
            
            # Should recover original
            fidelity = abs(np.vdot(encoded, corrected))**2
            assert fidelity > 0.9, f"Should correct error on qubit {qubit_idx}"
    
    def test_error_correction_fidelity_vs_gamma(self):
        """Test that error correction works for different gamma values."""
        encoded = tme9.encoding_circuit(1.0/math.sqrt(2), 1.0/math.sqrt(2))
        
        gammas = [0.01, 0.05, 0.1, 0.2]
        fidelities = []
        
        for gamma in gammas:
            rho, branches = tme9.apply_amplitude_damping_to_one_qubit(
                encoded, gamma, i=0
            )
            error_state = branches[1]
            error_state = error_state / np.linalg.norm(error_state)
            
            corrected = tme9.error_correction(error_state)
            fidelity = abs(np.vdot(encoded, corrected))**2
            fidelities.append(fidelity)
        
        # All fidelities should be high (error correction works)
        assert all(f > 0.9 for f in fidelities), \
            "Error correction should work well for all tested gamma values"
        
        # Larger gamma should generally give lower fidelity (though may be very close)
        # Check that at least the largest gamma doesn't exceed the smallest by too much
        assert fidelities[-1] >= fidelities[0] - 0.1, \
            "Very large gamma should not dramatically reduce fidelity"


class TestPhysicalProperties:
    """Tests for physical properties and constraints."""
    
    def test_all_density_matrices_valid(self):
        """Test that all generated density matrices are valid."""
        encoded = tme9.encoding_circuit(0.8, 0.6)
        gamma = 0.1
        
        # Apply damping to all qubits
        rho, _ = tme9.apply_amplitude_damping(encoded, gamma)
        
        # Check trace
        assert math.isclose(np.trace(rho), 1.0, abs_tol=1e-10)
        
        # Check Hermitian
        np.testing.assert_allclose(rho, rho.conj().T, rtol=1e-10)
        
        # Check positive semidefinite
        eigenvalues = np.linalg.eigvals(rho)
        assert np.all(eigenvalues >= -1e-10)
        
        # Check trace of rho^2 <= 1 (purity)
        assert np.trace(rho @ rho) <= 1.0 + 1e-10
    
    def test_recovery_operators_completeness(self):
        """Test that recovery operators cover all error spaces."""
        zeroL, oneL = tme9._logical_codewords()
        gamma = 0.1
        E0, E1 = tme9._kraus_ops(gamma)
        
        # Apply E1 to each qubit and verify recovery
        for j in range(4):
            error_state = tme9._apply_single_qubit_op(zeroL, E1, qubit_index=j)
            error_state = error_state / np.linalg.norm(error_state)
            
            Rj = tme9.recovery_operator(j)
            recovered = Rj @ error_state
            recovered = recovered / np.linalg.norm(recovered)
            
            # Should recover to |0_L>
            fidelity = abs(np.vdot(zeroL, recovered))**2
            assert fidelity > 0.99, f"Recovery operator {j} should work"
    
    def test_syndrome_projectors_orthogonality(self):
        """Test that syndrome projectors are orthogonal."""
        P, Pjs = tme9._syndrome_projectors()
        
        # Code space projector should be orthogonal to error spaces
        for Pj in Pjs:
            product = P @ Pj
            # Should be approximately zero
            assert np.linalg.norm(product) < 1e-10, \
                "Code space and error spaces should be orthogonal"
        
        # Error spaces should be orthogonal to each other
        for i in range(4):
            for j in range(i+1, 4):
                product = Pjs[i] @ Pjs[j]
                assert np.linalg.norm(product) < 1e-10, \
                    f"Error spaces {i} and {j} should be orthogonal"


class TestCodeProperties:
    """Tests for code structure and properties."""
    
    def test_code_dimension_counting(self):
        """Test that code satisfies dimension counting argument."""
        # Code space: 2 dimensions
        # Error spaces: 4 spaces of 2 dimensions each = 8 dimensions
        # Total: 10 dimensions out of 16
        
        zeroL, oneL = tme9._logical_codewords()
        P, Pjs = tme9._syndrome_projectors()
        
        # Trace of projectors gives dimension
        code_dim = np.trace(P).real
        assert math.isclose(code_dim, 2.0, abs_tol=1e-10)
        
        error_dims = sum(np.trace(Pj).real for Pj in Pjs)
        assert math.isclose(error_dims, 8.0, abs_tol=1e-10)
        
        total_dim = code_dim + error_dims
        assert math.isclose(total_dim, 10.0, abs_tol=1e-10)
    
    def test_error_states_orthogonality(self):
        """Test that single-jump error states are orthogonal."""
        zeroL, _ = tme9._logical_codewords()
        gamma = 0.1
        E0, E1 = tme9._kraus_ops(gamma)
        
        error_states = []
        for j in range(4):
            error_state = tme9._apply_single_qubit_op(zeroL, E1, qubit_index=j)
            error_state = error_state / np.linalg.norm(error_state)
            error_states.append(error_state)
        
        # All error states should be orthogonal
        for i in range(4):
            for j in range(i+1, 4):
                overlap = abs(np.vdot(error_states[i], error_states[j]))
                assert overlap < 1e-10, \
                    f"Error states {i} and {j} should be orthogonal"
    
    def test_minimum_qubit_requirement(self):
        """Test that n=4 is minimum for exact AD code."""
        # Test that n=2 and n=3 cannot have perfect codes
        for n in [2, 3]:
            best, _ = tme9.search_min_violation(n, trials=5000, seed=42)
            assert best > 0.1, f"n={n} should not have perfect code"
        
        # Test that n=4 has perfect code (known codewords)
        zeroL, oneL = tme9._logical_codewords()
        violation = tme9.kl_violation(4, zeroL, oneL)
        assert violation < 1e-10, "n=4 should have perfect code"
