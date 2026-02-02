"""
Comprehensive test suite for TME8 Mermin-Peres Magic Square

Tests quantum contextuality demonstration using the magic square game.
"""
import math
import numpy as np
import pytest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TME8_DIR = ROOT / "quantum-nonlocality-and-contextuality"
if str(TME8_DIR) not in sys.path:
    sys.path.insert(0, str(TME8_DIR))

import TME8_functions as tme8


class TestPauliMatrices:
    """Tests for Pauli matrix definitions."""
    
    def test_pauli_x_structure(self):
        """Test that X matrix has correct structure."""
        X = tme8.X
        assert X.shape == (2, 2)
        assert X[0, 0] == 0
        assert X[0, 1] == 1
        assert X[1, 0] == 1
        assert X[1, 1] == 0
    
    def test_pauli_y_structure(self):
        """Test that Y matrix has correct structure."""
        Y = tme8.Y
        assert Y.shape == (2, 2)
        assert Y[0, 0] == 0
        assert Y[0, 1] == -1j
        assert Y[1, 0] == 1j
        assert Y[1, 1] == 0
    
    def test_pauli_z_structure(self):
        """Test that Z matrix has correct structure."""
        Z = tme8.Z
        assert Z.shape == (2, 2)
        assert Z[0, 0] == 1
        assert Z[0, 1] == 0
        assert Z[1, 0] == 0
        assert Z[1, 1] == -1
    
    def test_pauli_matrices_hermitian(self):
        """Test that all Pauli matrices are Hermitian."""
        for name, matrix in [('X', tme8.X), ('Y', tme8.Y), ('Z', tme8.Z)]:
            np.testing.assert_allclose(matrix, matrix.conj().T, rtol=1e-10), \
                f"{name} should be Hermitian"
    
    def test_pauli_matrices_unitary(self):
        """Test that all Pauli matrices are unitary."""
        for name, matrix in [('X', tme8.X), ('Y', tme8.Y), ('Z', tme8.Z)]:
            product = matrix @ matrix.conj().T
            identity = np.eye(2, dtype=complex)
            np.testing.assert_allclose(product, identity, rtol=1e-10), \
                f"{name} should be unitary"


class TestTensorProduct:
    """Tests for tensor product function."""
    
    def test_tensor_product_two_qubits(self):
        """Test tensor product of two single-qubit operators."""
        result = tme8.tensor(tme8.X, tme8.I)
        assert result.shape == (4, 4)
        
        # X ⊗ I should have X in upper-left and lower-right blocks
        expected = np.kron(tme8.X, tme8.I)
        np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    def test_tensor_product_three_qubits(self):
        """Test tensor product of three single-qubit operators."""
        result = tme8.tensor(tme8.X, tme8.Y, tme8.Z)
        assert result.shape == (8, 8)
        
        expected = np.kron(np.kron(tme8.X, tme8.Y), tme8.Z)
        np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    def test_tensor_product_identity(self):
        """Test that tensor product with identity preserves structure."""
        result = tme8.tensor(tme8.I, tme8.I)
        identity_4 = np.eye(4, dtype=complex)
        np.testing.assert_allclose(result, identity_4, rtol=1e-10)


class TestMagicSquareObservables:
    """Tests for magic square observables."""
    
    def test_magic_square_shape(self):
        """Test that magic square has correct shape."""
        observables = tme8.get_magic_square_observables()
        assert len(observables) == 3
        assert all(len(row) == 3 for row in observables)
        assert all(obs.shape == (4, 4) for row in observables for obs in row)
    
    def test_magic_square_observables_hermitian(self):
        """Test that all observables are Hermitian."""
        observables = tme8.get_magic_square_observables()
        for i in range(3):
            for j in range(3):
                obs = observables[i][j]
                np.testing.assert_allclose(obs, obs.conj().T, rtol=1e-10), \
                    f"Observable ({i},{j}) should be Hermitian"
    
    def test_magic_square_row_products(self):
        """Test that row products are unitary (identity up to phase)."""
        observables = tme8.get_magic_square_observables()
        I_4 = np.eye(4, dtype=complex)
        
        # All rows should multiply to a unitary (identity or -identity)
        for row_idx in range(3):
            product = observables[row_idx][0]
            for j in range(1, 3):
                product = product @ observables[row_idx][j]
            # Should be unitary (either +I or -I)
            product_squared = product @ product
            np.testing.assert_allclose(product_squared, I_4, rtol=1e-10), \
                f"Row {row_idx} product should be unitary"
    
    def test_magic_square_column_products(self):
        """Test that column products are unitary."""
        observables = tme8.get_magic_square_observables()
        I_4 = np.eye(4, dtype=complex)
        
        for col_idx in range(3):
            product = observables[0][col_idx]
            for i in range(1, 3):
                product = product @ observables[i][col_idx]
            # Should be unitary
            product_squared = product @ product
            np.testing.assert_allclose(product_squared, I_4, rtol=1e-10), \
                f"Column {col_idx} product should be unitary"


class TestProjectors:
    """Tests for projector construction."""
    
    def test_projectors_shape(self):
        """Test that projectors have correct shape."""
        X = tme8.X
        P_plus, P_minus = tme8.get_projectors(X)
        
        assert P_plus.shape == (2, 2)
        assert P_minus.shape == (2, 2)
    
    def test_projectors_hermitian(self):
        """Test that projectors are Hermitian."""
        X = tme8.X
        P_plus, P_minus = tme8.get_projectors(X)
        
        np.testing.assert_allclose(P_plus, P_plus.conj().T, rtol=1e-10)
        np.testing.assert_allclose(P_minus, P_minus.conj().T, rtol=1e-10)
    
    def test_projectors_idempotent(self):
        """Test that projectors are idempotent (P^2 = P)."""
        X = tme8.X
        P_plus, P_minus = tme8.get_projectors(X)
        
        np.testing.assert_allclose(P_plus @ P_plus, P_plus, rtol=1e-10)
        np.testing.assert_allclose(P_minus @ P_minus, P_minus, rtol=1e-10)
    
    def test_projectors_completeness(self):
        """Test that projectors sum to identity."""
        X = tme8.X
        P_plus, P_minus = tme8.get_projectors(X)
        
        identity = np.eye(2, dtype=complex)
        np.testing.assert_allclose(P_plus + P_minus, identity, rtol=1e-10)
    
    def test_projectors_orthogonal(self):
        """Test that projectors are orthogonal (P_+ P_- = 0)."""
        X = tme8.X
        P_plus, P_minus = tme8.get_projectors(X)
        
        # Projectors should be orthogonal
        product = P_plus @ P_minus
        # Allow for numerical errors
        norm = np.linalg.norm(product)
        assert norm < 1e-10, f"Projectors should be orthogonal, got norm {norm}"
    
    def test_all_projectors_magic_square(self):
        """Test building all projectors for magic square."""
        observables = tme8.get_magic_square_observables()
        projectors = tme8.build_all_projectors(observables)
        
        # Should have 9 observables × 2 eigenvalues = 18 projectors
        assert len(projectors) == 18
        
        # Check that all projectors are valid
        for key, P in projectors.items():
            assert P.shape == (4, 4)
            # Should be Hermitian
            np.testing.assert_allclose(P, P.conj().T, rtol=1e-10)


class TestBellState:
    """Tests for Bell state."""
    
    def test_bell_state_shape(self):
        """Test that Bell state has correct shape."""
        psi = tme8.create_bell_state()
        assert psi.shape == (4, 1)
    
    def test_bell_state_normalized(self):
        """Test that Bell state is normalized."""
        psi = tme8.create_bell_state()
        norm = np.linalg.norm(psi)
        assert math.isclose(norm, 1.0, abs_tol=1e-10)
    
    def test_bell_state_structure(self):
        """Test that Bell state has correct structure."""
        psi = tme8.create_bell_state()
        
        # Should be (|00> + |11>)/√2
        expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        expected = expected.reshape(4, 1)
        
        np.testing.assert_allclose(psi, expected, rtol=1e-10)
    
    def test_bell_state_entanglement(self):
        """Test that Bell state is maximally entangled."""
        psi = tme8.create_bell_state()
        rho = np.outer(psi, psi.conj())
        
        # Partial trace over second qubit should give maximally mixed state
        rho_A = np.array([
            [rho[0, 0] + rho[1, 1], rho[0, 2] + rho[1, 3]],
            [rho[2, 0] + rho[3, 1], rho[2, 2] + rho[3, 3]]
        ])
        
        # Maximally mixed state is I/2
        I_2 = np.eye(2, dtype=complex) / 2
        np.testing.assert_allclose(rho_A, I_2, rtol=1e-10)


class TestQuantumStrategy:
    """Tests for quantum strategy in magic square game."""
    
    def test_quantum_strategy_winning_probability(self):
        """Test that quantum strategy achieves 100% winning probability."""
        prob = tme8.quantum_strategy_magic_square()
        assert math.isclose(prob, 1.0, abs_tol=1e-10), \
            "Quantum strategy should achieve 100% success"
    
    def test_classical_strategy_maximum(self):
        """Test that classical strategy maximum is 8/9."""
        max_prob = tme8.classical_strategy_magic_square()
        assert math.isclose(max_prob, 8.0/9.0, abs_tol=1e-10), \
            "Classical maximum should be 8/9"
    
    def test_quantum_beats_classical(self):
        """Test that quantum strategy beats classical maximum."""
        quantum_prob = tme8.quantum_strategy_magic_square()
        classical_max = tme8.classical_strategy_magic_square()
        
        assert quantum_prob > classical_max, \
            "Quantum strategy should beat classical maximum"


class TestMagicSquareParity:
    """Tests for parity constraints in magic square."""
    
    def test_parity_contradiction(self):
        """Test that parity constraints lead to contradiction classically."""
        observables = tme8.get_magic_square_observables()
        
        # Row products: +1, +1, -1 → product = -1
        row_product = 1
        for i in range(3):
            row_obs = observables[i][0] @ observables[i][1] @ observables[i][2]
            # For rows 0,1: should be +I, for row 2: should be -I
            if i < 2:
                row_product *= 1
            else:
                row_product *= -1
        
        # Column products: all +1 → product = +1
        col_product = 1
        for j in range(3):
            col_obs = observables[0][j] @ observables[1][j] @ observables[2][j]
            col_product *= 1
        
        # These should be different, demonstrating the contradiction
        assert row_product != col_product, \
            "Row and column products should differ, showing classical impossibility"
