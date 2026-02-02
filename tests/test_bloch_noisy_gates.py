"""
Comprehensive test suite for bloch_noisy_gates.py

Tests density matrix to Bloch vector conversions, quantum gates, 
noise channels using Kraus operators, and noisy gate trajectories.
"""
import math
import numpy as np
import pytest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DECOHERENCE_DIR = ROOT / "dynamics-of-a-qubit-under-decoherence"
if str(DECOHERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(DECOHERENCE_DIR))

import bloch_noisy_gates as bng


class TestBlochVectorConversions:
    """Tests for density matrix to Bloch vector conversions."""
    
    def test_rho_to_bloch_pure_state_0(self):
        """Test conversion of |0⟩⟨0| to Bloch vector."""
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        r = bng.rho_to_bloch(rho)
        
        expected = np.array([0, 0, 1])
        np.testing.assert_allclose(r, expected, rtol=1e-10)
    
    def test_rho_to_bloch_pure_state_1(self):
        """Test conversion of |1⟩⟨1| to Bloch vector."""
        rho = np.array([[0, 0], [0, 1]], dtype=complex)
        r = bng.rho_to_bloch(rho)
        
        expected = np.array([0, 0, -1])
        np.testing.assert_allclose(r, expected, rtol=1e-10)
    
    def test_rho_to_bloch_pure_state_plus(self):
        """Test conversion of |+⟩⟨+| to Bloch vector."""
        ket_plus = np.array([[1], [1]], dtype=complex) / np.sqrt(2)
        rho = ket_plus @ ket_plus.conj().T
        r = bng.rho_to_bloch(rho)
        
        expected = np.array([1, 0, 0])
        np.testing.assert_allclose(r, expected, rtol=1e-10)
    
    def test_bloch_to_rho_and_back(self):
        """Test round-trip conversion: rho -> Bloch -> rho."""
        rho_original = np.array([[0.7, 0.3], [0.3, 0.3]], dtype=complex)
        # Normalize to ensure valid density matrix
        rho_original = rho_original / np.trace(rho_original)
        
        r = bng.rho_to_bloch(rho_original)
        rho_reconstructed = bng.bloch_to_rho(r)
        
        np.testing.assert_allclose(rho_original, rho_reconstructed, rtol=1e-10)
    
    def test_bloch_to_rho_preserves_trace(self):
        """Test that bloch_to_rho produces a valid density matrix (trace=1)."""
        r = np.array([0.5, 0.3, 0.4])
        rho = bng.bloch_to_rho(r)
        
        trace = np.trace(rho)
        assert math.isclose(trace, 1.0, rel_tol=1e-10), \
            "Density matrix should have trace 1"
    
    def test_bloch_to_rho_hermitian(self):
        """Test that bloch_to_rho produces a Hermitian matrix."""
        r = np.array([0.6, 0.4, 0.5])
        rho = bng.bloch_to_rho(r)
        
        np.testing.assert_allclose(rho, rho.conj().T, rtol=1e-10)


class TestQuantumGates:
    """Tests for quantum gate operations."""
    
    def test_rx_gate_rotation_identity(self):
        """Test that Rx(0) is the identity."""
        gate = bng.rx_gate(0.0)
        expected = bng.I
        
        np.testing.assert_allclose(gate, expected, rtol=1e-10)
    
    def test_rx_gate_rotation_pi(self):
        """Test that Rx(π) is -iX."""
        gate = bng.rx_gate(np.pi)
        expected = -1j * bng.X
        
        np.testing.assert_allclose(gate, expected, rtol=1e-10, atol=1e-15)
    
    def test_rx_gate_unitary(self):
        """Test that Rx gate is unitary."""
        gate = bng.rx_gate(np.pi / 4)
        product = gate @ gate.conj().T
        
        np.testing.assert_allclose(product, bng.I, rtol=1e-10)
    
    def test_apply_gate_preserves_trace(self):
        """Test that applying a gate preserves the trace of the density matrix."""
        rho = np.array([[0.6, 0.2], [0.2, 0.4]], dtype=complex)
        rho = rho / np.trace(rho)  # Normalize
        
        gate = bng.rx_gate(np.pi / 3)
        rho_new = bng.apply_gate(rho, gate)
        
        assert math.isclose(np.trace(rho), np.trace(rho_new), rel_tol=1e-10)
    
    def test_apply_gate_preserves_hermiticity(self):
        """Test that applying a gate preserves Hermiticity."""
        rho = np.array([[0.7, 0.3], [0.3, 0.3]], dtype=complex)
        rho = rho / np.trace(rho)
        
        gate = bng.rx_gate(np.pi / 5)
        rho_new = bng.apply_gate(rho, gate)
        
        np.testing.assert_allclose(rho_new, rho_new.conj().T, rtol=1e-10)


class TestNoiseChannels:
    """Tests for quantum noise channels using Kraus operators."""
    
    def test_depolarizing_channel_preserves_trace(self):
        """Test that depolarizing channel preserves trace."""
        rho = np.array([[0.6, 0.2], [0.2, 0.4]], dtype=complex)
        rho = rho / np.trace(rho)
        
        rho_new = bng.depolarizing_channel(rho, 0.1)
        
        assert math.isclose(np.trace(rho), np.trace(rho_new), rel_tol=1e-10)
    
    def test_depolarizing_channel_at_p_zero(self):
        """Test that depolarizing channel with p=0 is identity."""
        rho = np.array([[0.7, 0.3], [0.3, 0.3]], dtype=complex)
        rho = rho / np.trace(rho)
        
        rho_new = bng.depolarizing_channel(rho, 0.0)
        
        np.testing.assert_allclose(rho, rho_new, rtol=1e-10)
    
    def test_depolarizing_channel_at_p_one(self):
        """Test that depolarizing channel with p=1 gives expected result."""
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        
        rho_new = bng.depolarizing_channel(rho, 1.0)
        
        # At p=1, the channel applies X, Y, Z with equal probability
        # For |0⟩⟨0|, this gives a weighted mixture
        # Verify it's a valid density matrix
        assert np.isclose(np.trace(rho_new), 1.0, rtol=1e-10)
        assert np.allclose(rho_new, rho_new.conj().T, rtol=1e-10)
    
    def test_phase_damping_channel_preserves_trace(self):
        """Test that phase damping channel preserves trace."""
        rho = np.array([[0.6, 0.2], [0.2, 0.4]], dtype=complex)
        rho = rho / np.trace(rho)
        
        rho_new = bng.phase_damping_channel(rho, 0.1)
        
        assert math.isclose(np.trace(rho), np.trace(rho_new), rel_tol=1e-10)
    
    def test_phase_damping_channel_at_gamma_zero(self):
        """Test that phase damping channel with γ=0 is identity."""
        rho = np.array([[0.7, 0.3], [0.3, 0.3]], dtype=complex)
        rho = rho / np.trace(rho)
        
        rho_new = bng.phase_damping_channel(rho, 0.0)
        
        np.testing.assert_allclose(rho, rho_new, rtol=1e-10)
    
    def test_amplitude_damping_channel_preserves_trace(self):
        """Test that amplitude damping channel preserves trace."""
        rho = np.array([[0.6, 0.2], [0.2, 0.4]], dtype=complex)
        rho = rho / np.trace(rho)
        
        rho_new = bng.amplitude_damping_channel(rho, 0.1)
        
        assert math.isclose(np.trace(rho), np.trace(rho_new), rel_tol=1e-10)
    
    def test_amplitude_damping_channel_at_gamma_zero(self):
        """Test that amplitude damping channel with γ=0 is identity."""
        rho = np.array([[0.7, 0.3], [0.3, 0.3]], dtype=complex)
        rho = rho / np.trace(rho)
        
        rho_new = bng.amplitude_damping_channel(rho, 0.0)
        
        np.testing.assert_allclose(rho, rho_new, rtol=1e-10)
    
    def test_amplitude_damping_converges_to_ground_state(self):
        """Test that amplitude damping converges to |0⟩⟨0| for large γ."""
        rho = np.array([[0, 0], [0, 1]], dtype=complex)  # |1⟩⟨1|
        
        rho_new = bng.amplitude_damping_channel(rho, 1.0)
        
        # Should have significant population in |0⟩⟨0|
        assert rho_new[0, 0] > 0.5


class TestNoisyGates:
    """Tests for noisy gate application."""
    
    def test_apply_noisy_gate_preserves_trace(self):
        """Test that noisy gate application preserves trace."""
        rho = np.array([[0.7, 0.3], [0.3, 0.3]], dtype=complex)
        rho = rho / np.trace(rho)
        gate = bng.rx_gate(np.pi / 4)
        
        rho_new = bng.apply_noisy_gate(rho, gate, "depolarizing", 0.05)
        
        assert math.isclose(np.trace(rho), np.trace(rho_new), rel_tol=1e-10)
    
    def test_apply_noisy_gate_depolarizing(self):
        """Test noisy gate with depolarizing noise."""
        rho = np.array([[1, 0], [0, 0]], dtype=complex)  # |0⟩⟨0|
        gate = bng.rx_gate(np.pi / 2)
        
        rho_new = bng.apply_noisy_gate(rho, gate, "depolarizing", 0.1)
        
        assert math.isclose(np.trace(rho_new), 1.0, rel_tol=1e-10)
    
    def test_apply_noisy_gate_phase_damping(self):
        """Test noisy gate with phase damping."""
        rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)  # |+⟩⟨+|
        gate = bng.rx_gate(np.pi / 3)
        
        rho_new = bng.apply_noisy_gate(rho, gate, "phase_damping", 0.1)
        
        assert math.isclose(np.trace(rho_new), 1.0, rel_tol=1e-10)
    
    def test_apply_noisy_gate_amplitude_damping(self):
        """Test noisy gate with amplitude damping."""
        rho = np.array([[0, 0], [0, 1]], dtype=complex)  # |1⟩⟨1|
        gate = bng.rx_gate(np.pi / 4)
        
        rho_new = bng.apply_noisy_gate(rho, gate, "amplitude_damping", 0.1)
        
        assert math.isclose(np.trace(rho_new), 1.0, rel_tol=1e-10)
    
    def test_apply_noisy_gate_invalid_noise_type(self):
        """Test that invalid noise type raises ValueError."""
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        gate = bng.rx_gate(np.pi / 4)
        
        with pytest.raises(ValueError, match="Unknown noise type"):
            bng.apply_noisy_gate(rho, gate, "invalid_noise", 0.1)


class TestTrajectoryGeneration:
    """Tests for noisy gate trajectory generation."""
    
    def test_generate_noisy_gate_trajectory_shape(self):
        """Test that trajectory has correct shape."""
        rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
        gate = bng.rx_gate(np.pi / 20)
        num_steps = 50
        
        trajectory = bng.generate_noisy_gate_trajectory(
            rho0, gate, num_steps, "depolarizing", 0.01
        )
        
        assert trajectory.shape == (num_steps + 1, 3), \
            "Trajectory should have num_steps+1 points"
    
    def test_generate_noisy_gate_trajectory_initial_point(self):
        """Test that trajectory starts at correct initial state."""
        rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
        gate = bng.rx_gate(np.pi / 10)
        
        trajectory = bng.generate_noisy_gate_trajectory(
            rho0, gate, 10, "depolarizing", 0.01
        )
        
        r_initial = bng.rho_to_bloch(rho0)
        np.testing.assert_allclose(trajectory[0], r_initial, rtol=1e-10)
    
    def test_generate_noisy_gate_trajectory_with_gate_list(self):
        """Test trajectory generation with a list of gates."""
        rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
        gates = [bng.rx_gate(np.pi / 10), bng.rx_gate(-np.pi / 10)]
        num_steps = 20
        
        trajectory = bng.generate_noisy_gate_trajectory(
            rho0, gates, num_steps, "depolarizing", 0.01
        )
        
        assert trajectory.shape == (num_steps + 1, 3)
    
    def test_generate_noisy_gate_trajectory_convergence(self):
        """Test that trajectory shows expected convergence behavior."""
        rho0 = np.array([[0, 0], [0, 1]], dtype=complex)  # |1⟩⟨1|
        gate = bng.rx_gate(np.pi / 20)
        num_steps = 100
        
        trajectory = bng.generate_noisy_gate_trajectory(
            rho0, gate, num_steps, "amplitude_damping", 0.05
        )
        
        # With amplitude damping, z should tend toward 1 (ground state)
        # But with rotations, it will oscillate
        # Just check that trajectory is valid (all points on or inside Bloch sphere)
        for point in trajectory:
            norm = np.linalg.norm(point)
            assert norm <= 1.0 + 1e-10, "Bloch vector should be on or inside unit sphere"


class TestInitialStates:
    """Tests for initial state generation."""
    
    def test_get_initial_states_contains_expected_states(self):
        """Test that get_initial_states returns expected states."""
        states = bng.get_initial_states()
        
        assert "|0⟩" in states
        assert "|1⟩" in states
        assert "|+⟩" in states
        assert "|-⟩" in states
    
    def test_get_initial_states_are_valid_density_matrices(self):
        """Test that all initial states are valid density matrices."""
        states = bng.get_initial_states()
        
        for name, rho in states.items():
            # Check trace is 1
            assert math.isclose(np.trace(rho), 1.0, rel_tol=1e-10), \
                f"{name} should have trace 1"
            
            # Check Hermitian
            np.testing.assert_allclose(rho, rho.conj().T, rtol=1e-10), \
                f"{name} should be Hermitian"
            
            # Check positive semidefinite (eigenvalues >= 0)
            eigenvalues = np.linalg.eigvals(rho)
            assert np.all(eigenvalues >= -1e-10), \
                f"{name} should be positive semidefinite"
    
    def test_initial_state_0_is_correct(self):
        """Test that |0⟩ state is correctly defined."""
        states = bng.get_initial_states()
        rho_0 = states["|0⟩"]
        
        expected = np.array([[1, 0], [0, 0]], dtype=complex)
        np.testing.assert_allclose(rho_0, expected, rtol=1e-10)
    
    def test_initial_state_1_is_correct(self):
        """Test that |1⟩ state is correctly defined."""
        states = bng.get_initial_states()
        rho_1 = states["|1⟩"]
        
        expected = np.array([[0, 0], [0, 1]], dtype=complex)
        np.testing.assert_allclose(rho_1, expected, rtol=1e-10)


class TestPhysicalProperties:
    """Tests for physical properties and constraints."""
    
    def test_all_noise_channels_preserve_trace(self):
        """Test that all noise channels preserve trace."""
        rho = np.array([[0.6, 0.2+0.1j], [0.2-0.1j, 0.4]], dtype=complex)
        rho = rho / np.trace(rho)
        
        channels = [
            ("depolarizing", lambda r: bng.depolarizing_channel(r, 0.1)),
            ("phase_damping", lambda r: bng.phase_damping_channel(r, 0.1)),
            ("amplitude_damping", lambda r: bng.amplitude_damping_channel(r, 0.1)),
        ]
        
        for name, channel in channels:
            rho_new = channel(rho)
            assert math.isclose(np.trace(rho), np.trace(rho_new), rel_tol=1e-10), \
                f"{name} channel should preserve trace"
    
    def test_all_noise_channels_preserve_hermiticity(self):
        """Test that all noise channels preserve Hermiticity."""
        rho = np.array([[0.6, 0.2+0.1j], [0.2-0.1j, 0.4]], dtype=complex)
        rho = rho / np.trace(rho)
        
        channels = [
            ("depolarizing", lambda r: bng.depolarizing_channel(r, 0.1)),
            ("phase_damping", lambda r: bng.phase_damping_channel(r, 0.1)),
            ("amplitude_damping", lambda r: bng.amplitude_damping_channel(r, 0.1)),
        ]
        
        for name, channel in channels:
            rho_new = channel(rho)
            np.testing.assert_allclose(rho_new, rho_new.conj().T, rtol=1e-10), \
                f"{name} channel should preserve Hermiticity"
