"""
Comprehensive test suite for TME8 GHZ Paradox

Tests quantum nonlocality demonstration using the GHZ state and game.
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


class TestGHZState:
    """Tests for GHZ state creation."""
    
    def test_ghz_state_shape_3_qubits(self):
        """Test that 3-qubit GHZ state has correct shape."""
        ghz = tme8.create_ghz_state(3)
        assert ghz.shape == (8, 1)
    
    def test_ghz_state_shape_n_qubits(self):
        """Test that n-qubit GHZ state has correct shape."""
        for n in [2, 4, 5]:
            ghz = tme8.create_ghz_state(n)
            assert ghz.shape == (2**n, 1)
    
    def test_ghz_state_normalized(self):
        """Test that GHZ state is normalized."""
        for n in [3, 4, 5]:
            ghz = tme8.create_ghz_state(n)
            norm = np.linalg.norm(ghz)
            assert math.isclose(norm, 1.0, abs_tol=1e-10)
    
    def test_ghz_state_structure_3_qubits(self):
        """Test that 3-qubit GHZ state has correct structure."""
        ghz = tme8.create_ghz_state(3)
        
        # Should be (|000> + |111>)/√2
        assert abs(ghz[0, 0] - 1/np.sqrt(2)) < 1e-10  # |000>
        assert abs(ghz[7, 0] - 1/np.sqrt(2)) < 1e-10  # |111>
        # All other components should be zero
        for i in range(1, 7):
            assert abs(ghz[i, 0]) < 1e-10
    
    def test_ghz_state_structure_n_qubits(self):
        """Test that n-qubit GHZ state has correct structure."""
        for n in [2, 4, 5]:
            ghz = tme8.create_ghz_state(n)
            dim = 2**n
            
            # First component (|00...0>)
            assert abs(ghz[0, 0] - 1/np.sqrt(2)) < 1e-10
            # Last component (|11...1>)
            assert abs(ghz[dim-1, 0] - 1/np.sqrt(2)) < 1e-10
            # All other components should be zero
            for i in range(1, dim-1):
                assert abs(ghz[i, 0]) < 1e-10


class TestGHZInputs:
    """Tests for valid GHZ game inputs."""
    
    def test_valid_inputs_3_players(self):
        """Test valid inputs for 3-player GHZ game."""
        valid_inputs = tme8.get_valid_ghz_inputs(3)
        
        # Should have 4 valid combinations
        assert len(valid_inputs) == 4
        
        # Check that all satisfy sum ≡ 0 (mod 2)
        for inputs_tuple in valid_inputs:
            assert sum(inputs_tuple) % 2 == 0, \
                f"Input {inputs_tuple} should have even sum"
        
        # Check specific combinations
        expected = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
        assert set(valid_inputs) == set(expected)
    
    def test_valid_inputs_n_players(self):
        """Test valid inputs for n-player GHZ game."""
        for n in [2, 4, 5]:
            valid_inputs = tme8.get_valid_ghz_inputs(n)
            expected_count = 2**(n-1)
            assert len(valid_inputs) == expected_count
            
            # All should satisfy sum ≡ 0 (mod 2)
            for inputs_tuple in valid_inputs:
                assert sum(inputs_tuple) % 2 == 0
    
    def test_count_valid_inputs(self):
        """Test counting function for valid inputs."""
        for n in [2, 3, 4, 5]:
            count = tme8.count_valid_inputs_n_players(n)
            expected = 2**(n-1)
            assert count == expected


class TestGHZFunction:
    """Tests for GHZ game winning condition function."""
    
    def test_f_ghz_3_players(self):
        """Test f function for 3-player GHZ game."""
        assert tme8.f_ghz(0, 0, 0) == 0
        assert tme8.f_ghz(0, 1, 1) == 1
        assert tme8.f_ghz(1, 0, 1) == 1
        assert tme8.f_ghz(1, 1, 0) == 1
    
    def test_f_ghz_n_players(self):
        """Test generalized f function for n players."""
        # All zeros should return 0
        assert tme8.f_ghz_n(0, 0, 0) == 0
        assert tme8.f_ghz_n(0, 0, 0, 0) == 0
        
        # Any non-zero should return 1
        assert tme8.f_ghz_n(1, 0, 0) == 1
        assert tme8.f_ghz_n(0, 1, 0) == 1
        assert tme8.f_ghz_n(0, 0, 1) == 1
        assert tme8.f_ghz_n(1, 1, 1) == 1


class TestMeasurementProjectors:
    """Tests for X and Y measurement projectors."""
    
    def test_x_projectors_structure(self):
        """Test that X projectors have correct structure."""
        P_plus, P_minus = tme8.get_x_projectors()
        
        assert P_plus.shape == (2, 2)
        assert P_minus.shape == (2, 2)
        
        # Should be Hermitian
        np.testing.assert_allclose(P_plus, P_plus.conj().T, rtol=1e-10)
        np.testing.assert_allclose(P_minus, P_minus.conj().T, rtol=1e-10)
    
    def test_y_projectors_structure(self):
        """Test that Y projectors have correct structure."""
        P_plus, P_minus = tme8.get_y_projectors()
        
        assert P_plus.shape == (2, 2)
        assert P_minus.shape == (2, 2)
        
        # Should be Hermitian
        np.testing.assert_allclose(P_plus, P_plus.conj().T, rtol=1e-10)
        np.testing.assert_allclose(P_minus, P_minus.conj().T, rtol=1e-10)
    
    def test_projectors_completeness(self):
        """Test that projectors sum to identity."""
        P_x_plus, P_x_minus = tme8.get_x_projectors()
        P_y_plus, P_y_minus = tme8.get_y_projectors()
        
        I = np.eye(2, dtype=complex)
        np.testing.assert_allclose(P_x_plus + P_x_minus, I, rtol=1e-10)
        np.testing.assert_allclose(P_y_plus + P_y_minus, I, rtol=1e-10)
    
    def test_projectors_orthogonal(self):
        """Test that projectors are orthogonal."""
        P_x_plus, P_x_minus = tme8.get_x_projectors()
        P_y_plus, P_y_minus = tme8.get_y_projectors()
        
        # Projectors should be orthogonal (allow for numerical errors)
        norm_x = np.linalg.norm(P_x_plus @ P_x_minus)
        norm_y = np.linalg.norm(P_y_plus @ P_y_minus)
        assert norm_x < 1e-10, f"X projectors should be orthogonal, got norm {norm_x}"
        assert norm_y < 1e-10, f"Y projectors should be orthogonal, got norm {norm_y}"


class TestQuantumGHZStrategy:
    """Tests for quantum strategy in GHZ game."""
    
    def test_quantum_strategy_3_players(self):
        """Test that quantum strategy achieves 100% for 3 players."""
        prob = tme8.simulate_quantum_ghz_game(3)
        assert math.isclose(prob, 1.0, abs_tol=1e-10), \
            "Quantum strategy should achieve 100% success for 3 players"
    
    def test_quantum_strategy_n_players(self):
        """Test that quantum strategy achieves 100% for n players."""
        for n in [3, 4, 5]:
            prob = tme8.simulate_quantum_ghz_game(n)
            assert math.isclose(prob, 1.0, abs_tol=1e-10), \
                f"Quantum strategy should achieve 100% success for {n} players"


class TestClassicalGHZStrategy:
    """Tests for classical strategy in GHZ game."""
    
    def test_classical_strategy_3_players(self):
        """Test maximum classical winning probability for 3 players."""
        max_prob, best_strategy = tme8.find_max_classical_win_prob_ghz(3)
        
        # Classical maximum should be 3/4 = 0.75
        assert math.isclose(max_prob, 0.75, abs_tol=1e-10), \
            "Classical maximum should be 75% for 3 players"
        
        assert best_strategy is not None
        assert best_strategy['wins'] == 3
        assert best_strategy['total'] == 4
    
    def test_classical_vs_quantum_3_players(self):
        """Test that quantum beats classical for 3 players."""
        quantum_prob = tme8.simulate_quantum_ghz_game(3)
        classical_max, _ = tme8.find_max_classical_win_prob_ghz(3)
        
        assert quantum_prob > classical_max, \
            "Quantum strategy should beat classical maximum"
        assert math.isclose(quantum_prob, 1.0, abs_tol=1e-10), \
            "Quantum should achieve 100%"
        assert math.isclose(classical_max, 0.75, abs_tol=1e-10), \
            "Classical should achieve at most 75%"


class TestGHZParadox:
    """Tests for GHZ paradox properties."""
    
    def test_ghz_state_parity_properties(self):
        """Test that GHZ state exhibits expected parity properties."""
        ghz = tme8.create_ghz_state(3)
        rho = np.outer(ghz, ghz.conj())
        
        # X ⊗ X ⊗ X should have eigenvalue +1
        XXX = tme8.tensor(tme8.X, tme8.X, tme8.X)
        expectation = np.trace(XXX @ rho).real
        assert math.isclose(expectation, 1.0, abs_tol=1e-10), \
            "X⊗X⊗X should have expectation +1"
        
        # X ⊗ Y ⊗ Y should have eigenvalue -1
        XYY = tme8.tensor(tme8.X, tme8.Y, tme8.Y)
        expectation = np.trace(XYY @ rho).real
        assert math.isclose(expectation, -1.0, abs_tol=1e-10), \
            "X⊗Y⊗Y should have expectation -1"
    
    def test_ghz_paradox_contradiction(self):
        """Test that GHZ paradox leads to contradiction classically."""
        # The paradox: if we assign predetermined values a_X, a_Y, b_X, b_Y, c_X, c_Y
        # We get: (a_X b_Y c_Y)(a_Y b_X c_Y)(a_Y b_Y c_X) = -1
        # But this simplifies to: a_X b_X c_X = -1
        # However, we also have: a_X b_X c_X = +1
        # This is a contradiction
        
        # In quantum mechanics, this is resolved because measurements
        # don't have predetermined values
        ghz = tme8.create_ghz_state(3)
        rho = np.outer(ghz, ghz.conj())
        
        # Verify the quantum correlations
        XYY = tme8.tensor(tme8.X, tme8.Y, tme8.Y)
        YXY = tme8.tensor(tme8.Y, tme8.X, tme8.Y)
        YYX = tme8.tensor(tme8.Y, tme8.Y, tme8.X)
        XXX = tme8.tensor(tme8.X, tme8.X, tme8.X)
        
        exp_XYY = np.trace(XYY @ rho).real
        exp_YXY = np.trace(YXY @ rho).real
        exp_YYX = np.trace(YYX @ rho).real
        exp_XXX = np.trace(XXX @ rho).real
        
        # First three should be -1
        assert math.isclose(exp_XYY, -1.0, abs_tol=1e-10)
        assert math.isclose(exp_YXY, -1.0, abs_tol=1e-10)
        assert math.isclose(exp_YYX, -1.0, abs_tol=1e-10)
        
        # Last should be +1
        assert math.isclose(exp_XXX, 1.0, abs_tol=1e-10)
        
        # The product of first three is -1, but this equals XXX which is +1
        # This contradiction shows classical hidden variables are impossible
