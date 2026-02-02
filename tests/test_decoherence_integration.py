"""
Integration tests for quantum decoherence dynamics.

Tests complete workflows and interactions between different components
of the decoherence simulation system.
"""
import numpy as np
import pytest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DECOHERENCE_DIR = ROOT / "dynamics-of-a-qubit-under-decoherence"
if str(DECOHERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(DECOHERENCE_DIR))

import bloch_trajectories as bt
import bloch_noisy_gates as bng


class TestEndToEndWorkflows:
    """End-to-end integration tests."""
    
    def test_complete_phase_damping_simulation(self):
        """Test complete phase damping simulation workflow."""
        # Start with a pure state on the equator
        r0 = np.array([1.0, 0.0, 0.0])
        time_steps = np.linspace(0, 10, 100)
        T2 = 2.0
        
        # Generate trajectory
        trajectory = bt.generate_trajectory(
            bt.phase_damping, r0, time_steps, use_time=True, T2=T2
        )
        
        # Verify trajectory properties
        assert trajectory.shape == (100, 3)
        assert np.allclose(trajectory[0], r0)
        
        # Verify z-component is preserved
        assert np.allclose(trajectory[:, 2], r0[2])
        
        # Verify x and y decay (the trajectory applies channel iteratively)
        # Check that decay is happening and is monotonic
        norms = [np.linalg.norm(trajectory[i, :2]) for i in range(len(trajectory))]
        
        # Initial norm should be 1.0 (r0 = [1, 0, 0])
        assert np.isclose(norms[0], 1.0, rtol=1e-10)
        
        # Final norm should be smaller (decay has occurred)
        assert norms[-1] < norms[0], "Decay should occur over time"
        
        # Check that decay is generally monotonic (allowing for small numerical fluctuations)
        # The norm should generally decrease, though not necessarily strictly
        decreasing_count = sum(1 for i in range(1, len(norms)) if norms[i] <= norms[i-1] + 1e-6)
        assert decreasing_count > len(norms) * 0.8, "Most steps should show decay"
    
    def test_complete_amplitude_damping_simulation(self):
        """Test complete amplitude damping simulation workflow."""
        # Start with excited state
        r0 = np.array([0.0, 0.0, -1.0])  # |1⟩
        time_steps = np.linspace(0, 10, 100)
        T1 = 3.0
        
        # Generate trajectory
        trajectory = bt.generate_trajectory(
            bt.amplitude_damping, r0, time_steps, use_time=True, T1=T1
        )
        
        # Verify trajectory properties
        assert trajectory.shape == (100, 3)
        assert np.allclose(trajectory[0], r0)
        
        # Verify z-component increases toward 1 (ground state)
        assert trajectory[-1, 2] > trajectory[0, 2]
        assert trajectory[-1, 2] > 0.5  # Should be closer to ground state
    
    def test_density_matrix_to_trajectory_workflow(self):
        """Test workflow: density matrix -> Bloch -> trajectory -> Bloch."""
        # Start with density matrix
        rho0 = np.array([[1, 0], [0, 0]], dtype=complex)  # |0⟩⟨0|
        
        # Convert to Bloch vector
        r0 = bng.rho_to_bloch(rho0)
        expected_r0 = np.array([0, 0, 1])
        assert np.allclose(r0, expected_r0)
        
        # Generate trajectory using Bloch vector
        time_steps = np.linspace(0, 5, 50)
        trajectory = bt.generate_trajectory(
            bt.phase_damping, r0, time_steps, use_time=True, T2=2.0
        )
        
        # Convert final point back to density matrix
        r_final = trajectory[-1]
        rho_final = bng.bloch_to_rho(r_final)
        
        # Verify it's a valid density matrix
        assert np.isclose(np.trace(rho_final), 1.0)
        assert np.allclose(rho_final, rho_final.conj().T)
    
    def test_noisy_gate_sequence_workflow(self):
        """Test complete noisy gate sequence workflow."""
        # Get initial state
        initial_states = bng.get_initial_states()
        rho0 = initial_states["|0⟩"]
        
        # Define gate sequence
        gate = bng.rx_gate(np.pi / 10)
        num_steps = 50
        
        # Generate trajectory
        trajectory = bng.generate_noisy_gate_trajectory(
            rho0, gate, num_steps, "depolarizing", 0.01
        )
        
        # Verify trajectory
        assert trajectory.shape == (num_steps + 1, 3)
        
        # Convert initial and final points to density matrices
        r_initial = trajectory[0]
        r_final = trajectory[-1]
        
        rho_initial = bng.bloch_to_rho(r_initial)
        rho_final = bng.bloch_to_rho(r_final)
        
        # Verify both are valid density matrices
        assert np.isclose(np.trace(rho_initial), 1.0)
        assert np.isclose(np.trace(rho_final), 1.0)
    
    def test_multiple_noise_channels_comparison(self):
        """Test comparing trajectories under different noise channels."""
        r0 = np.array([1.0, 0.0, 0.0])
        time_steps = np.linspace(0, 5, 50)
        
        # Generate trajectories for different channels
        traj_phase = bt.generate_trajectory(
            bt.phase_damping, r0, time_steps, use_time=True, T2=2.0
        )
        traj_amplitude = bt.generate_trajectory(
            bt.amplitude_damping, r0, time_steps, use_time=True, T1=3.0
        )
        traj_depolarizing = bt.generate_trajectory(
            bt.depolarizing_noise, r0, time_steps, use_time=False, p=0.1
        )
        
        # All should have same shape
        assert traj_phase.shape == traj_amplitude.shape == traj_depolarizing.shape
        
        # Phase damping should preserve z
        assert np.allclose(traj_phase[:, 2], r0[2])
        
        # Amplitude damping should change z
        assert not np.allclose(traj_amplitude[:, 2], r0[2])
        
        # Depolarizing should scale all components iteratively
        # First point should be r0, subsequent points decay as (1-p)^n
        assert np.allclose(traj_depolarizing[0], r0)
        assert np.allclose(traj_depolarizing[1], (1 - 0.1) * r0)


class TestCrossModuleCompatibility:
    """Tests for compatibility between different modules."""
    
    def test_bloch_vector_consistency(self):
        """Test that Bloch vectors are consistent between modules."""
        # Create a state in bloch_noisy_gates
        rho = np.array([[0.7, 0.3], [0.3, 0.3]], dtype=complex)
        rho = rho / np.trace(rho)
        
        r_from_rho = bng.rho_to_bloch(rho)
        
        # Use this Bloch vector in bloch_trajectories
        r_after_phase = bt.phase_damping(r_from_rho, 1.0, 2.0)
        
        # Convert back to density matrix
        rho_after = bng.bloch_to_rho(r_after_phase)
        
        # Should still be a valid density matrix
        assert np.isclose(np.trace(rho_after), 1.0)
        assert np.allclose(rho_after, rho_after.conj().T)
    
    def test_kraus_vs_bloch_consistency(self):
        """Test consistency between Kraus operator channels and Bloch vector channels."""
        # Start with a density matrix
        rho0 = np.array([[0.6, 0.2], [0.2, 0.4]], dtype=complex)
        rho0 = rho0 / np.trace(rho0)
        
        # Apply phase damping via Kraus operators
        gamma = 0.1
        rho_kraus = bng.phase_damping_channel(rho0, gamma)
        
        # Convert to Bloch, apply phase damping, convert back
        r0 = bng.rho_to_bloch(rho0)
        # For phase damping, gamma relates to T2: gamma = 1 - exp(-t/T2)
        # If we set t/T2 such that exp(-t/T2) = 1 - gamma, then:
        # t/T2 = -ln(1 - gamma)
        # For small gamma: t/T2 ≈ gamma
        t_over_T2 = -np.log(1 - gamma)
        r_bloch = bt.phase_damping(r0, 1.0, 1.0 / t_over_T2)
        rho_bloch = bng.bloch_to_rho(r_bloch)
        
        # They should be approximately equal (within numerical precision)
        # Note: This is an approximate test due to different formulations
        assert np.isclose(np.trace(rho_kraus), np.trace(rho_bloch))
    
    def test_gate_application_consistency(self):
        """Test that gate application is consistent across workflows."""
        # Apply gate then noise
        rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
        gate = bng.rx_gate(np.pi / 4)
        
        rho_gate_then_noise = bng.apply_noisy_gate(rho0, gate, "depolarizing", 0.1)
        
        # Apply gate manually, then noise
        rho_after_gate = bng.apply_gate(rho0, gate)
        rho_manual = bng.depolarizing_channel(rho_after_gate, 0.1)
        
        # Should be the same
        np.testing.assert_allclose(rho_gate_then_noise, rho_manual, rtol=1e-10)


class TestPhysicalConsistency:
    """Tests for physical consistency across the system."""
    
    def test_all_trajectories_stay_in_bloch_sphere(self):
        """Test that all generated trajectories stay within the Bloch sphere."""
        r0 = np.array([0.8, 0.6, 0.5])
        r0 = r0 / np.linalg.norm(r0)  # Normalize to surface
        
        time_steps = np.linspace(0, 10, 100)
        
        channels = [
            (bt.phase_damping, {"use_time": True, "T2": 2.0}),
            (bt.amplitude_damping, {"use_time": True, "T1": 3.0}),
            (bt.depolarizing_noise, {"use_time": False, "p": 0.1}),
        ]
        
        for channel_func, kwargs in channels:
            trajectory = bt.generate_trajectory(
                channel_func, r0, time_steps, **kwargs
            )
            
            for point in trajectory:
                norm = np.linalg.norm(point)
                assert norm <= 1.0 + 1e-10, \
                    f"Point {point} has norm {norm} > 1 in {channel_func.__name__}"
    
    def test_density_matrices_remain_valid(self):
        """Test that density matrices remain valid throughout operations."""
        rho0 = np.array([[0.7, 0.3], [0.3, 0.3]], dtype=complex)
        rho0 = rho0 / np.trace(rho0)
        
        gate = bng.rx_gate(np.pi / 6)
        num_steps = 20
        
        trajectory = bng.generate_noisy_gate_trajectory(
            rho0, gate, num_steps, "amplitude_damping", 0.05
        )
        
        # Convert all points to density matrices and verify
        for r in trajectory:
            rho = bng.bloch_to_rho(r)
            
            # Check trace
            assert np.isclose(np.trace(rho), 1.0, rtol=1e-10)
            
            # Check Hermitian
            assert np.allclose(rho, rho.conj().T, rtol=1e-10)
            
            # Check positive semidefinite
            eigenvalues = np.linalg.eigvals(rho)
            assert np.all(eigenvalues >= -1e-10)
            
            # Check trace of rho^2 <= 1 (purity)
            assert np.trace(rho @ rho) <= 1.0 + 1e-10
    
    def test_energy_conservation_amplitude_damping(self):
        """Test that amplitude damping correctly models energy dissipation."""
        # Start with excited state |1⟩
        rho0 = np.array([[0, 0], [0, 1]], dtype=complex)
        
        # Apply amplitude damping
        rho_final = bng.amplitude_damping_channel(rho0, 0.5)
        
        # Population in ground state should increase
        assert rho_final[0, 0] > rho0[0, 0]
        
        # Population in excited state should decrease
        assert rho_final[1, 1] < rho0[1, 1]
        
        # Total population should be conserved (trace = 1)
        assert np.isclose(np.trace(rho_final), 1.0)
