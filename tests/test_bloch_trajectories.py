"""
Comprehensive test suite for bloch_trajectories.py

Tests quantum decoherence channels (phase damping, amplitude damping, depolarizing noise)
and trajectory generation on the Bloch sphere.
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

import bloch_trajectories as bt


class TestPhaseDamping:
    """Tests for phase damping channel."""
    
    def test_phase_damping_preserves_z_component(self):
        """Phase damping should preserve the z-component of the Bloch vector."""
        r0 = np.array([0.5, 0.5, 0.7])
        t = 1.0
        T2 = 2.0
        
        r_final = bt.phase_damping(r0, t, T2)
        
        assert math.isclose(r_final[2], r0[2], rel_tol=1e-10), \
            "Z-component should be preserved under phase damping"
    
    def test_phase_damping_decays_xy_components(self):
        """Phase damping should exponentially decay x and y components."""
        r0 = np.array([1.0, 1.0, 0.0])
        t = 1.0
        T2 = 2.0
        
        r_final = bt.phase_damping(r0, t, T2)
        expected_decay = np.exp(-t / T2)
        
        assert math.isclose(r_final[0], r0[0] * expected_decay, rel_tol=1e-10)
        assert math.isclose(r_final[1], r0[1] * expected_decay, rel_tol=1e-10)
    
    def test_phase_damping_at_zero_time(self):
        """At t=0, phase damping should return the original vector."""
        r0 = np.array([0.6, 0.8, 0.5])
        r_final = bt.phase_damping(r0, 0.0, 2.0)
        
        np.testing.assert_allclose(r_final, r0, rtol=1e-10)
    
    def test_phase_damping_at_infinite_time(self):
        """At very large time, x and y should approach zero."""
        r0 = np.array([1.0, 1.0, 0.5])
        r_final = bt.phase_damping(r0, 1000.0, 1.0)
        
        assert abs(r_final[0]) < 1e-10
        assert abs(r_final[1]) < 1e-10
        assert math.isclose(r_final[2], r0[2], rel_tol=1e-10)


class TestAmplitudeDamping:
    """Tests for amplitude damping channel."""
    
    def test_amplitude_damping_converges_to_ground_state(self):
        """Amplitude damping should drive the state toward |0⟩ (z=1)."""
        r0 = np.array([0.5, 0.5, -0.5])  # State with negative z
        t = 10.0
        T1 = 1.0
        
        r_final = bt.amplitude_damping(r0, t, T1)
        
        # At large time, z should approach 1
        assert r_final[2] > 0.9, "State should converge to ground state"
    
    def test_amplitude_damping_preserves_ground_state(self):
        """The ground state |0⟩ (z=1) should be preserved."""
        r0 = np.array([0.0, 0.0, 1.0])
        t = 5.0
        T1 = 2.0
        
        r_final = bt.amplitude_damping(r0, t, T1)
        
        np.testing.assert_allclose(r_final, r0, rtol=1e-10)
    
    def test_amplitude_damping_decay_rates(self):
        """Verify the exponential decay rates for x, y, and z components."""
        r0 = np.array([1.0, 1.0, 0.0])
        t = 1.0
        T1 = 2.0
        
        r_final = bt.amplitude_damping(r0, t, T1)
        
        expected_x = r0[0] * np.exp(-t / (2 * T1))
        expected_y = r0[1] * np.exp(-t / (2 * T1))
        expected_z = r0[2] * np.exp(-t / T1) + (1 - np.exp(-t / T1))
        
        np.testing.assert_allclose(r_final[0], expected_x, rtol=1e-10)
        np.testing.assert_allclose(r_final[1], expected_y, rtol=1e-10)
        np.testing.assert_allclose(r_final[2], expected_z, rtol=1e-10)
    
    def test_amplitude_damping_at_zero_time(self):
        """At t=0, amplitude damping should return the original vector."""
        r0 = np.array([0.6, 0.8, 0.5])
        r_final = bt.amplitude_damping(r0, 0.0, 2.0)
        
        np.testing.assert_allclose(r_final, r0, rtol=1e-10)


class TestDepolarizingNoise:
    """Tests for depolarizing noise channel."""
    
    def test_depolarizing_noise_scales_vector(self):
        """Depolarizing noise should scale the Bloch vector by (1-p)."""
        r0 = np.array([0.8, 0.6, 0.5])
        p = 0.2
        
        r_final = bt.depolarizing_noise(r0, p)
        expected = (1 - p) * r0
        
        np.testing.assert_allclose(r_final, expected, rtol=1e-10)
    
    def test_depolarizing_noise_at_p_zero(self):
        """At p=0, depolarizing noise should preserve the state."""
        r0 = np.array([0.7, 0.5, 0.3])
        r_final = bt.depolarizing_noise(r0, 0.0)
        
        np.testing.assert_allclose(r_final, r0, rtol=1e-10)
    
    def test_depolarizing_noise_at_p_one(self):
        """At p=1, depolarizing noise should completely depolarize (zero vector)."""
        r0 = np.array([0.8, 0.6, 0.5])
        r_final = bt.depolarizing_noise(r0, 1.0)
        
        np.testing.assert_allclose(r_final, np.zeros(3), rtol=1e-10)
    
    def test_depolarizing_noise_handles_2d_array(self):
        """Should handle both 1D and 2D input arrays."""
        r0_2d = np.array([[0.8], [0.6], [0.5]])
        p = 0.3
        
        r_final = bt.depolarizing_noise(r0_2d, p)
        expected = (1 - p) * r0_2d.flatten()
        
        np.testing.assert_allclose(r_final, expected, rtol=1e-10)


class TestTrajectoryGeneration:
    """Tests for trajectory generation functions."""
    
    def test_generate_trajectory_phase_damping(self):
        """Test trajectory generation for phase damping."""
        r0 = np.array([1.0, 0.0, 0.5])
        time_steps = np.linspace(0, 5, 10)
        T2 = 2.0
        
        trajectory = bt.generate_trajectory(bt.phase_damping, r0, time_steps, 
                                            use_time=True, T2=T2)
        
        assert trajectory.shape == (10, 3), "Trajectory should have correct shape"
        np.testing.assert_allclose(trajectory[0], r0, rtol=1e-10)
        assert trajectory[0][2] == trajectory[-1][2], "Z-component should be preserved"
    
    def test_generate_trajectory_amplitude_damping(self):
        """Test trajectory generation for amplitude damping."""
        r0 = np.array([0.0, 0.0, -0.5])
        time_steps = np.linspace(0, 10, 20)
        T1 = 3.0
        
        trajectory = bt.generate_trajectory(bt.amplitude_damping, r0, time_steps,
                                            use_time=True, T1=T1)
        
        assert trajectory.shape == (20, 3)
        np.testing.assert_allclose(trajectory[0], r0, rtol=1e-10)
        # z should increase toward 1
        assert trajectory[-1][2] > trajectory[0][2]
    
    def test_generate_trajectory_depolarizing(self):
        """Test trajectory generation for depolarizing noise."""
        r0 = np.array([0.8, 0.6, 0.5])
        time_steps = np.linspace(0, 1, 5)
        p = 0.1
        
        trajectory = bt.generate_trajectory(bt.depolarizing_noise, r0, time_steps,
                                           use_time=False, p=p)
        
        assert trajectory.shape == (5, 3)
        # First point should be r0
        np.testing.assert_allclose(trajectory[0], r0, rtol=1e-10)
        # Subsequent points decay exponentially: (1-p)^n * r0
        for i, point in enumerate(trajectory[1:], start=1):
            expected = ((1 - p) ** i) * r0
            np.testing.assert_allclose(point, expected, rtol=1e-10)
    
    def test_generate_trajectory_single_step(self):
        """Test trajectory with single time step."""
        r0 = np.array([1.0, 0.0, 0.0])
        time_steps = np.array([0.0])
        
        trajectory = bt.generate_trajectory(bt.phase_damping, r0, time_steps,
                                            use_time=True, T2=2.0)
        
        assert trajectory.shape == (1, 3)
        np.testing.assert_allclose(trajectory[0], r0, rtol=1e-10)


class TestBlochSphereVisualization:
    """Tests for Bloch sphere visualization functions."""
    
    def test_bloch_sphere_function_exists(self):
        """Test that bloch_sphere function exists and is callable."""
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Should not raise an error
        bt.bloch_sphere(ax)
        
        plt.close(fig)
    
    def test_bloch_sphere_sets_axis_limits(self):
        """Test that bloch_sphere sets appropriate axis limits."""
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        bt.bloch_sphere(ax)
        
        assert ax.get_xlim() == (-1.01, 1.01)
        assert ax.get_ylim() == (-1.01, 1.01)
        assert ax.get_zlim() == (-1.01, 1.01)
        
        plt.close(fig)


class TestPhysicalProperties:
    """Tests for physical properties and constraints."""
    
    def test_phase_damping_preserves_bloch_vector_norm_decreases(self):
        """Phase damping decreases the length of the Bloch vector in xy-plane."""
        r0 = np.array([1.0, 0.0, 0.5])
        t = 1.0
        T2 = 2.0
        
        r_final = bt.phase_damping(r0, t, T2)
        
        norm_xy_initial = np.sqrt(r0[0]**2 + r0[1]**2)
        norm_xy_final = np.sqrt(r_final[0]**2 + r_final[1]**2)
        
        assert norm_xy_final < norm_xy_initial, \
            "Phase damping should decrease the xy-plane norm"
    
    def test_amplitude_damping_monotonic_z_increase(self):
        """For states with z < 1, amplitude damping should increase z monotonically."""
        r0 = np.array([0.0, 0.0, 0.0])
        t1 = 1.0
        t2 = 2.0
        T1 = 2.0
        
        r1 = bt.amplitude_damping(r0, t1, T1)
        r2 = bt.amplitude_damping(r0, t2, T1)
        
        assert r2[2] > r1[2], "z should increase monotonically with time"
    
    def test_depolarizing_noise_contracts_bloch_vector(self):
        """Depolarizing noise should contract the Bloch vector toward origin."""
        r0 = np.array([0.8, 0.6, 0.5])
        p = 0.3
        
        r_final = bt.depolarizing_noise(r0, p)
        
        norm_initial = np.linalg.norm(r0)
        norm_final = np.linalg.norm(r_final)
        
        assert norm_final < norm_initial, \
            "Depolarizing noise should decrease the Bloch vector norm"
        assert math.isclose(norm_final, (1 - p) * norm_initial, rel_tol=1e-10)


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_phase_damping_with_zero_vector(self):
        """Test phase damping with zero vector."""
        r0 = np.array([0.0, 0.0, 0.0])
        r_final = bt.phase_damping(r0, 1.0, 2.0)
        
        np.testing.assert_allclose(r_final, r0, rtol=1e-10)
    
    def test_amplitude_damping_with_zero_vector(self):
        """Test amplitude damping with zero vector."""
        r0 = np.array([0.0, 0.0, 0.0])
        r_final = bt.amplitude_damping(r0, 1.0, 2.0)
        
        # Should converge to ground state
        assert r_final[2] > 0
    
    def test_depolarizing_noise_with_zero_vector(self):
        """Test depolarizing noise with zero vector."""
        r0 = np.array([0.0, 0.0, 0.0])
        r_final = bt.depolarizing_noise(r0, 0.5)
        
        np.testing.assert_allclose(r_final, r0, rtol=1e-10)
    
    def test_trajectory_with_empty_time_steps(self):
        """Test trajectory generation with minimal time steps."""
        r0 = np.array([1.0, 0.0, 0.0])
        time_steps = np.array([0.0, 1.0])
        
        trajectory = bt.generate_trajectory(bt.phase_damping, r0, time_steps,
                                            use_time=True, T2=2.0)
        
        assert trajectory.shape == (2, 3)
        np.testing.assert_allclose(trajectory[0], r0, rtol=1e-10)
