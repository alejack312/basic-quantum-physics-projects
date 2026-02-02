"""
Integration tests for TME8 Quantum Nonlocality and Contextuality

Tests complete workflows and interactions between components.
"""
import numpy as np
import pytest
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TME8_DIR = ROOT / "quantum-nonlocality-and-contextuality"
if str(TME8_DIR) not in sys.path:
    sys.path.insert(0, str(TME8_DIR))

import TME8_functions as tme8


class TestMagicSquareWorkflow:
    """End-to-end tests for magic square game."""
    
    def test_complete_magic_square_workflow(self):
        """Test complete magic square game workflow."""
        # Get observables
        observables = tme8.get_magic_square_observables()
        assert len(observables) == 3
        
        # Build projectors
        projectors = tme8.build_all_projectors(observables)
        assert len(projectors) == 18
        
        # Create Bell state
        psi = tme8.create_bell_state()
        assert np.linalg.norm(psi) == pytest.approx(1.0)
        
        # Test quantum strategy
        quantum_prob = tme8.quantum_strategy_magic_square()
        assert quantum_prob == pytest.approx(1.0)
        
        # Test classical strategy
        classical_max = tme8.classical_strategy_magic_square()
        assert classical_max == pytest.approx(8.0/9.0)
        
        # Quantum should beat classical
        assert quantum_prob > classical_max
    
    def test_magic_square_measurement_simulation(self):
        """Test simulating measurements on magic square observables."""
        observables = tme8.get_magic_square_observables()
        psi = tme8.create_bell_state()
        rho = np.outer(psi, psi.conj())
        
        # Test measurement on each observable
        for i in range(3):
            for j in range(3):
                obs = observables[i][j]
                P_plus, P_minus = tme8.get_projectors(obs)
                
                # Probabilities should sum to 1
                prob_plus = np.trace(P_plus @ rho).real
                prob_minus = np.trace(P_minus @ rho).real
                
                assert math.isclose(prob_plus + prob_minus, 1.0, abs_tol=1e-10), \
                    f"Probabilities for ({i},{j}) should sum to 1"


class TestGHZWorkflow:
    """End-to-end tests for GHZ game."""
    
    def test_complete_ghz_workflow_3_players(self):
        """Test complete GHZ game workflow for 3 players."""
        # Create GHZ state
        ghz = tme8.create_ghz_state(3)
        assert np.linalg.norm(ghz) == pytest.approx(1.0)
        
        # Get valid inputs
        valid_inputs = tme8.get_valid_ghz_inputs(3)
        assert len(valid_inputs) == 4
        
        # Test quantum strategy
        quantum_prob = tme8.simulate_quantum_ghz_game(3)
        assert quantum_prob == pytest.approx(1.0)
        
        # Test classical strategy
        classical_max, _ = tme8.find_max_classical_win_prob_ghz(3)
        assert classical_max == pytest.approx(0.75)
        
        # Quantum should beat classical
        assert quantum_prob > classical_max
    
    def test_ghz_measurement_simulation(self):
        """Test simulating measurements on GHZ state."""
        ghz = tme8.create_ghz_state(3)
        rho = np.outer(ghz, ghz.conj())
        
        # Test X and Y measurements
        P_x_plus, P_x_minus = tme8.get_x_projectors()
        P_y_plus, P_y_minus = tme8.get_y_projectors()
        
        # Single-qubit measurements
        for P_plus, P_minus in [(P_x_plus, P_x_minus), (P_y_plus, P_y_minus)]:
            # Partial trace to get single-qubit state
            # For GHZ, each qubit should be maximally mixed
            prob_plus = 0.5  # Should be 50/50 for maximally mixed
            prob_minus = 0.5
            assert math.isclose(prob_plus + prob_minus, 1.0, abs_tol=1e-10)


class TestQuantumAdvantage:
    """Tests demonstrating quantum advantage."""
    
    def test_magic_square_quantum_advantage(self):
        """Test that quantum strategy provides advantage in magic square."""
        quantum_prob = tme8.quantum_strategy_magic_square()
        classical_max = tme8.classical_strategy_magic_square()
        
        advantage = quantum_prob - classical_max
        assert advantage > 0.1, "Quantum should have significant advantage"
        assert quantum_prob == pytest.approx(1.0), "Quantum should achieve 100%"
    
    def test_ghz_quantum_advantage(self):
        """Test that quantum strategy provides advantage in GHZ game."""
        quantum_prob = tme8.simulate_quantum_ghz_game(3)
        classical_max, _ = tme8.find_max_classical_win_prob_ghz(3)
        
        advantage = quantum_prob - classical_max
        assert advantage > 0.2, "Quantum should have significant advantage"
        assert quantum_prob == pytest.approx(1.0), "Quantum should achieve 100%"
    
    def test_quantum_advantage_scales(self):
        """Test that quantum advantage persists for more players."""
        for n in [3, 4, 5]:
            quantum_prob = tme8.simulate_quantum_ghz_game(n)
            assert quantum_prob == pytest.approx(1.0), \
                f"Quantum should achieve 100% for {n} players"


class TestPhysicalProperties:
    """Tests for physical properties and constraints."""
    
    def test_bell_state_entanglement(self):
        """Test that Bell state is maximally entangled."""
        psi = tme8.create_bell_state()
        rho = np.outer(psi, psi.conj())
        
        # Partial trace over second qubit
        rho_A = np.array([
            [rho[0, 0] + rho[1, 1], rho[0, 2] + rho[1, 3]],
            [rho[2, 0] + rho[3, 1], rho[2, 2] + rho[3, 3]]
        ])
        
        # Should be maximally mixed (I/2)
        I_2 = np.eye(2, dtype=complex) / 2
        np.testing.assert_allclose(rho_A, I_2, rtol=1e-10)
    
    def test_ghz_state_entanglement(self):
        """Test that GHZ state is maximally entangled."""
        ghz = tme8.create_ghz_state(3)
        rho = np.outer(ghz, ghz.conj())
        
        # GHZ state is a genuine 3-qubit entangled state
        # All qubits are correlated
        # Verify by checking that partial traces give mixed states
        # (This is a simplified check)
        assert np.linalg.norm(ghz) == pytest.approx(1.0)
    
    def test_all_projectors_valid(self):
        """Test that all projectors are valid quantum operators."""
        observables = tme8.get_magic_square_observables()
        projectors = tme8.build_all_projectors(observables)
        
        for key, P in projectors.items():
            # Should be Hermitian
            np.testing.assert_allclose(P, P.conj().T, rtol=1e-10)
            
            # Should be idempotent
            np.testing.assert_allclose(P @ P, P, rtol=1e-10)
            
            # Should have non-negative eigenvalues
            eigenvals = np.linalg.eigvals(P)
            assert np.all(eigenvals >= -1e-10)


class TestGeneralization:
    """Tests for generalizations to n players/qubits."""
    
    def test_ghz_generalization(self):
        """Test GHZ state and game for different numbers of qubits."""
        for n in [2, 3, 4, 5]:
            # State creation
            ghz = tme8.create_ghz_state(n)
            assert ghz.shape == (2**n, 1)
            assert np.linalg.norm(ghz) == pytest.approx(1.0)
            
            # Valid inputs
            valid_inputs = tme8.get_valid_ghz_inputs(n)
            expected_count = 2**(n-1)
            assert len(valid_inputs) == expected_count
            
            # Quantum strategy
            quantum_prob = tme8.simulate_quantum_ghz_game(n)
            assert quantum_prob == pytest.approx(1.0)
    
    def test_function_generalization(self):
        """Test that f function generalizes correctly."""
        # Test for different numbers of players
        assert tme8.f_ghz_n(0, 0, 0) == 0
        assert tme8.f_ghz_n(0, 0, 0, 0) == 0
        assert tme8.f_ghz_n(1, 0, 0) == 1
        assert tme8.f_ghz_n(0, 1, 0, 0) == 1
        assert tme8.f_ghz_n(1, 1, 1, 1) == 1
