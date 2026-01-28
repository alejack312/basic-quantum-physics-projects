import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.colors import CSS4_COLORS

from random import choice


# --- Helper to later get a random color for a vector on the Bloch sphere (cannot be black, cyan, red, or green to avoid confusing with axes)
forbidden_colors = ['black', 'cyan', 'red', 'green']
colors_CSS4 = list(CSS4_COLORS.keys())

for color in forbidden_colors:
    if color in colors_CSS4:
        del CSS4_COLORS[color]

colors = list(CSS4_COLORS.keys())


# --- Pauli matrices ---
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.array([[1, 0], [0, 1]], dtype=complex)


# --- Computational basis states ---
ket0 = np.array([[1], [0]], dtype=complex)
ket1 = np.array([[0], [1]], dtype=complex)
bra0 = ket0.conj().T
bra1 = ket1.conj().T


# --- Convert density matrix to Bloch vector ---
def rho_to_bloch(rho: np.ndarray) -> np.ndarray:
    """
    Convert a density matrix to its corresponding Bloch vector.
    :param rho: 2x2 density matrix
    :return: Bloch vector (x, y, z) as 1D array
    """
    r_x = np.real(np.trace(rho @ X))
    r_y = np.real(np.trace(rho @ Y))
    r_z = np.real(np.trace(rho @ Z))
    return np.array([r_x, r_y, r_z])


# --- Convert Bloch vector to density matrix ---
def bloch_to_rho(r: np.ndarray) -> np.ndarray:
    """
    Convert a Bloch vector to its corresponding density matrix.
    :param r: Bloch vector (x, y, z)
    :return: 2x2 density matrix
    """
    r = r.flatten()
    return 0.5 * (I + r[0] * X + r[1] * Y + r[2] * Z)


# --- Define initial states ---
def get_initial_states():
    """
    Define the four initial states: |0⟩, |1⟩, |+⟩, |i⟩
    :return: dictionary mapping state names to density matrices
    """
    # |0⟩
    rho_0 = ket0 @ bra0
    
    # |1⟩
    rho_1 = ket1 @ bra1
    
    # |+⟩ = (|0⟩ + |1⟩)/√2
    ket_plus = (ket0 + ket1) / np.sqrt(2)
    rho_plus = ket_plus @ ket_plus.conj().T
    
    # |-⟩ = (|0⟩ - |1⟩)/√2
    ket_minus = (ket0 - ket1) / np.sqrt(2)
    rho_minus = ket_minus @ ket_minus.conj().T
    
    return {
        "|0⟩": rho_0,
        "|1⟩": rho_1,
        "|+⟩": rho_plus,
        "|-⟩": rho_minus
    }


# --- Quantum gates ---
def rx_gate(theta: float) -> np.ndarray:
    """
    Rotation gate around X axis: Rx(θ) = cos(θ/2)I - i*sin(θ/2)X
    """
    return np.cos(theta/2) * I - 1j * np.sin(theta/2) * X

def apply_gate(rho: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """
    Apply a unitary gate to a density matrix: U ρ U†
    :param rho: density matrix
    :param gate: unitary gate matrix
    :return: transformed density matrix
    """
    return gate @ rho @ gate.conj().T


# --- Noise channels using Kraus operators ---

def depolarizing_channel(rho: np.ndarray, p: float) -> np.ndarray:
    """
    Apply depolarizing noise channel with probability p.
    Kraus operators: K_0 = √(1-p) I, K_1 = √(p/3) X, K_2 = √(p/3) Y, K_3 = √(p/3) Z
    :param rho: density matrix
    :param p: depolarizing probability
    :return: transformed density matrix
    """
    sqrt_1_p = np.sqrt(1 - p)
    sqrt_p3 = np.sqrt(p / 3)
    
    K0 = sqrt_1_p * I
    K1 = sqrt_p3 * X
    K2 = sqrt_p3 * Y
    K3 = sqrt_p3 * Z
    
    rho_new = K0 @ rho @ K0.conj().T
    rho_new += K1 @ rho @ K1.conj().T
    rho_new += K2 @ rho @ K2.conj().T
    rho_new += K3 @ rho @ K3.conj().T
    
    return rho_new


def phase_damping_channel(rho: np.ndarray, gamma: float) -> np.ndarray:
    """
    Apply phase damping channel with probability gamma.
    Kraus operators: K_0 = √(1-γ) I, K_1 = √γ |0⟩⟨0|, K_2 = √γ |1⟩⟨1|
    :param rho: density matrix
    :param gamma: phase damping probability
    :return: transformed density matrix
    """
    sqrt_1_gamma = np.sqrt(1 - gamma)
    sqrt_gamma = np.sqrt(gamma)
    
    K0 = sqrt_1_gamma * I
    K1 = sqrt_gamma * (ket0 @ bra0)
    K2 = sqrt_gamma * (ket1 @ bra1)
    
    rho_new = K0 @ rho @ K0.conj().T
    rho_new += K1 @ rho @ K1.conj().T
    rho_new += K2 @ rho @ K2.conj().T
    
    return rho_new


def amplitude_damping_channel(rho: np.ndarray, gamma: float) -> np.ndarray:
    """
    Apply amplitude damping channel with probability gamma.
    Kraus operators:
    K_0 = [[1, 0], [0, √(1-γ)]]
    K_1 = [[0, √γ], [0, 0]]
    :param rho: density matrix
    :param gamma: amplitude damping probability
    :return: transformed density matrix
    """
    sqrt_1_gamma = np.sqrt(1 - gamma)
    sqrt_gamma = np.sqrt(gamma)
    
    K0 = np.array([[1, 0], [0, sqrt_1_gamma]], dtype=complex)
    K1 = np.array([[0, sqrt_gamma], [0, 0]], dtype=complex)
    
    rho_new = K0 @ rho @ K0.conj().T
    rho_new += K1 @ rho @ K1.conj().T
    
    return rho_new


# --- Apply noisy gate: gate followed by noise channel ---
def apply_noisy_gate(rho: np.ndarray, gate: np.ndarray, noise_type: str = "depolarizing", 
                     noise_param: float = 0.01) -> np.ndarray:
    """
    Apply a perfect gate followed by a noise channel.
    :param rho: density matrix
    :param gate: unitary gate matrix
    :param noise_type: type of noise ("depolarizing", "phase_damping", "amplitude_damping")
    :param noise_param: noise parameter (p for depolarizing, gamma for phase/amplitude damping)
    :return: transformed density matrix
    """
    # First apply the perfect gate
    rho = apply_gate(rho, gate)
    
    # Then apply noise channel
    if noise_type == "depolarizing":
        rho = depolarizing_channel(rho, noise_param)
    elif noise_type == "phase_damping":
        rho = phase_damping_channel(rho, noise_param)
    elif noise_type == "amplitude_damping":
        rho = amplitude_damping_channel(rho, noise_param)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return rho


# --- Generate trajectory for repeated noisy gates ---
def generate_noisy_gate_trajectory(rho0: np.ndarray, gates: list, num_steps: int,
                                   noise_type: str = "depolarizing", 
                                   noise_param: float = 0.01) -> np.ndarray:
    """
    Generate a trajectory by repeatedly applying noisy gates.
    :param rho0: initial density matrix
    :param gates: list of gates to apply (can be a single gate repeated, or alternating gates)
    :param num_steps: number of gate applications
    :param noise_type: type of noise channel
    :param noise_param: noise parameter
    :return: array of Bloch vectors representing the trajectory
    """
    trajectory = [rho_to_bloch(rho0)]
    rho = rho0.copy()
    
    for step in range(num_steps):
        # Select gate (if gates is a list, cycle through it; if single gate, use it)
        if isinstance(gates, list):
            gate = gates[step % len(gates)]
        else:
            gate = gates
        
        # Apply noisy gate
        rho = apply_noisy_gate(rho, gate, noise_type, noise_param)
        
        # Convert to Bloch vector and add to trajectory
        trajectory.append(rho_to_bloch(rho))
    
    return np.array(trajectory)


# --- Plotting function for trajectories ---
def bloch_sphere(ax):
    """
    Plot the empty Bloch sphere.
    :param ax: matplotlib 3D axis
    """
    u, v = np.mgrid[0:2*np.pi:200j, 0:np.pi:100j]

    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)

    # Get rid of colored axes planes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    # Plot the sphere surface
    ax.plot_wireframe(x, y, z, color="lightgray", alpha=0.15)

    # Add colored orthonormal axes
    alpha = 0.5
    ax.quiver(0,0,0,1,0,0,color='r',linewidth=2, arrow_length_ratio=0.1, alpha=alpha)
    ax.quiver(0,0,0,0,1,0,color='g',linewidth=2, arrow_length_ratio=0.1, alpha=alpha)
    ax.quiver(0,0,0,0,0,1,color='cyan',linewidth=2, arrow_length_ratio=0.1, alpha=alpha)
    ax.text(1.2, 0, 0, 'X', color='r', fontsize=18, alpha=0.6)
    ax.text(0, 1.2, 0, 'Y', color='g', fontsize=18, alpha=0.6)
    ax.text(0, 0, 1.2, 'Z', color='cyan', fontsize=18, alpha=0.6)

    # Add black orthonormal axes
    ax.plot([-1.2, 1.2], [0, 0], [0, 0], color='black', alpha=0.3, linewidth=1)
    ax.plot([0, 0], [-1.2, 1.2], [0, 0], color='black', alpha=0.3, linewidth=1)
    ax.plot([0, 0], [0, 0], [-1.2, 1.2], color='black', alpha=0.3, linewidth=1)

    # Set axis limits
    ax.set_xlim([-1.01, 1.01])
    ax.set_ylim([-1.01, 1.01])
    ax.set_zlim([-1.01, 1.01])

    # Hide the axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_box_aspect([1,1,1])
    ax.set_axis_off()


def plot_trajectories(trajectories, num_steps, interval=100, save_anim=False, 
                     anim_name='bloch_noisy_gates.gif'):
    """
    Plot and animate multiple Bloch sphere trajectories in a 2x2 quadrant layout.
    Each quadrant shows one initial state's trajectory.
    :param trajectories: list of tuples (trajectory, label, color) - should have exactly 4 elements
    :param num_steps: number of steps in the trajectory
    :param interval: interval between frames in milliseconds
    :param save_anim: whether to save the animation
    :param anim_name: name of the output file
    """
    if len(trajectories) != 4:
        raise ValueError("Must provide exactly 4 trajectories for the 4-quadrant layout")
    
    fig = plt.figure(figsize=(16, 16))
    
    # Create 2x2 subplot layout
    axes = []
    scatter_plots = []  # For trajectory dots
    current_points = []  # For current position marker
    
    # Quadrant positions: top-left, top-right, bottom-left, bottom-right
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for idx, (traj, label, color) in enumerate(trajectories):
        ax = fig.add_subplot(2, 2, idx+1, projection='3d')
        bloch_sphere(ax)
        ax.set_title(label, fontsize=14, fontweight='bold')
        axes.append(ax)
        
        # Create scatter plot for trajectory dots (will build up over time)
        scatter = ax.scatter([], [], [], c=color, s=20, alpha=0.7, edgecolors='none')
        scatter_plots.append(scatter)
        
        # Create a larger marker for the current position
        current_point, = ax.plot([], [], [], 'o', color=color, markersize=12, 
                                 markeredgecolor='black', markeredgewidth=1.5, zorder=10)
        current_points.append(current_point)

    def update(frame):
        all_artists = []
        for i, (traj, _, color) in enumerate(trajectories):
            # Ensure frame doesn't exceed trajectory length
            max_frame = min(frame, len(traj) - 1)
            
            # Show trajectory up to current frame as dots (building up over time)
            if max_frame >= 0:
                # Remove old scatter plot and create new one with updated points
                scatter_plots[i].remove()
                scatter_plots[i] = axes[i].scatter(
                    traj[:max_frame+1, 0], 
                    traj[:max_frame+1, 1], 
                    traj[:max_frame+1, 2],
                    c=color, s=20, alpha=0.7, edgecolors='none'
                )
                
                # Update the current position marker (shows where we are now)
                current_points[i].set_data([traj[max_frame, 0]], [traj[max_frame, 1]])
                current_points[i].set_3d_properties([traj[max_frame, 2]])
                
                all_artists.extend([scatter_plots[i], current_points[i]])
            else:
                # At frame 0, show just the initial point
                current_points[i].set_data([traj[0, 0]], [traj[0, 1]])
                current_points[i].set_3d_properties([traj[0, 2]])
                all_artists.append(current_points[i])
        return all_artists

    plt.tight_layout()
    # Note: blit=False for scatter plots as they need to be recreated each frame
    ani = FuncAnimation(fig, update, frames=num_steps+1, interval=interval, blit=False, repeat=True)

    if save_anim:
        try:
            writer = FFMpegWriter(fps=1000 // interval)
            ani.save(anim_name.replace('.gif', '.mp4'), writer=writer)
            print(f"Animation saved as MP4: {anim_name.replace('.gif', '.mp4')}")
        except Exception as e:
            print(f"Failed to save as MP4: {e}")
            print(f"Attempting to save as GIF instead")
            try:
                writer = PillowWriter(fps=1000 // interval)
                ani.save(anim_name, writer=writer)
                print(f"Animation saved as GIF: {anim_name}")
            except Exception as gif_error:
                print(f"Failed to save as GIF: {gif_error}")
                print("Animation could not be saved in any format.")
    else:
        plt.show()


if __name__ == "__main__":
    # Get initial states
    initial_states = get_initial_states()
    
    # Parameters
    num_steps = 200  # Increased steps to show the spiral better
    noise_type = "amplitude_damping"  # Options: "depolarizing", "phase_damping", "amplitude_damping"
    noise_param = 0.02  # Slightly stronger noise to show decay
    
    # Define gate sequence
    # Use a small rotation around X to create the spiral effect seen in Figure 3
    # Rotation angle approx pi/20 per step
    gate_sequence = rx_gate(np.pi / 20)
    
    # Or alternating gates: gate_sequence = [X, Y]
    
    # Generate trajectories for each initial state in a specific order
    # Order: |0⟩, |1⟩, |+⟩, |-⟩ (matching Figure 3: a, b, c, d)
    state_order = ["|0⟩", "|1⟩", "|+⟩", "|-⟩"]
    state_colors = {
        "|0⟩": 'red',     # Red in the figure
        "|1⟩": 'red',     # Red in the figure
        "|+⟩": 'red',     # Red in the figure
        "|-⟩": 'red'      # Red in the figure
    }
    
    trajectories = []
    for state_name in state_order:
        rho0 = initial_states[state_name]
        traj = generate_noisy_gate_trajectory(
            rho0, gate_sequence, num_steps, 
            noise_type=noise_type, 
            noise_param=noise_param
        )
        color = state_colors.get(state_name, choice(colors))
        trajectories.append((traj, state_name, color))
    
    # Plot trajectories in 4-quadrant layout
    plot_trajectories(
        trajectories, num_steps,
        interval=50,
        save_anim=True,
        anim_name='bloch_noisy_gates_amplitude_damping.gif'
    )
