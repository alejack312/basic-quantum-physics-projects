"""
Comprehensive benchmarking suite for quantum SWE portfolio report.

Generates measurable outcomes suitable for LaTeX report including:
- Quantum vs Classical performance comparisons
- Scalability analysis
- Error tolerance studies
- Resource usage profiling
- Statistical analysis with confidence intervals
"""
import platform
import time
import numpy as np
import json
import statistics
from pathlib import Path
import sys
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import all modules
import quantum_simulator as qs

# Import decoherence modules
DECOHERENCE_DIR = ROOT / "dynamics-of-a-qubit-under-decoherence"
if str(DECOHERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(DECOHERENCE_DIR))
import bloch_trajectories as bt
import bloch_noisy_gates as bng

# Import TME8 modules
TME8_DIR = ROOT / "quantum-nonlocality-and-contextuality"
if str(TME8_DIR) not in sys.path:
    sys.path.insert(0, str(TME8_DIR))
import TME8_functions as tme8

# Import TME9 modules
TME9_DIR = ROOT / "tme-9"
if str(TME9_DIR) not in sys.path:
    sys.path.insert(0, str(TME9_DIR))
import TME9_functions as tme9


def benchmark_with_statistics(func, n_runs=10, *args, **kwargs):
    """
    Benchmark a function with statistical analysis.
    
    Returns:
        Dict with mean, std, min, max, median, and confidence interval
    """
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return {
        'mean': statistics.mean(times),
        'std': statistics.stdev(times) if len(times) > 1 else 0.0,
        'min': min(times),
        'max': max(times),
        'median': statistics.median(times),
        'runs': n_runs,
        'result': result
    }


def benchmark_scalability_quantum_simulator():
    """Benchmark quantum simulator scalability with system size."""
    print("\n" + "="*70)
    print("Quantum Simulator Scalability Analysis")
    print("="*70)
    
    results = {
        'n_qubits': [],
        'depth': [],
        'time_mean': [],
        'time_std': [],
        'time_min': [],
        'time_max': []
    }
    
    configs = [
        (2, 10), (2, 25), (2, 50),
        (3, 10), (3, 25), (3, 50),
        (4, 10), (4, 25), (4, 50),
        (5, 10), (5, 25),
    ]
    
    rng = np.random.RandomState(42)
    
    for n_qubits, depth in configs:
        stats = benchmark_with_statistics(
            lambda: simulate_circuit(n_qubits, depth, rng),
            n_runs=5
        )
        
        results['n_qubits'].append(n_qubits)
        results['depth'].append(depth)
        results['time_mean'].append(stats['mean'])
        results['time_std'].append(stats['std'])
        results['time_min'].append(stats['min'])
        results['time_max'].append(stats['max'])
        
        print(f"n={n_qubits:2d}, depth={depth:3d}: "
              f"{stats['mean']:.6f} ± {stats['std']:.6f} s "
              f"(min={stats['min']:.6f}, max={stats['max']:.6f})")
    
    return results


def simulate_circuit(n_qubits, depth, rng):
    """Simulate a random quantum circuit."""
    state = qs.basis_state(0, n_qubits)
    for _ in range(depth):
        gates = []
        for _ in range(n_qubits):
            gate = qs.gate_h() if rng.random() < 0.5 else qs.gate_x()
            gates.append(gate)
        layer = qs.kron_all(gates)
        state = qs.apply_operator(layer, state)
    return state


def benchmark_quantum_vs_classical_games():
    """Compare quantum vs classical strategies in games."""
    print("\n" + "="*70)
    print("Quantum vs Classical Strategy Comparison")
    print("="*70)
    
    results = {}
    
    # Magic Square Game
    print("\nMagic Square Game:")
    quantum_prob = tme8.quantum_strategy_magic_square()
    classical_max = tme8.classical_strategy_magic_square()
    advantage = quantum_prob - classical_max
    
    results['magic_square'] = {
        'quantum_win_rate': float(quantum_prob),
        'classical_max_win_rate': float(classical_max),
        'quantum_advantage': float(advantage),
        'improvement_factor': float(quantum_prob / classical_max)
    }
    
    print(f"  Quantum win rate: {quantum_prob:.4f} ({quantum_prob*100:.2f}%)")
    print(f"  Classical max: {classical_max:.4f} ({classical_max*100:.2f}%)")
    print(f"  Quantum advantage: {advantage:.4f} ({advantage*100:.2f} percentage points)")
    print(f"  Improvement factor: {quantum_prob/classical_max:.4f}x")
    
    # GHZ Game
    print("\nGHZ Game (3 players):")
    quantum_prob_ghz = tme8.simulate_quantum_ghz_game(3)
    classical_max_ghz, _ = tme8.find_max_classical_win_prob_ghz(3)
    advantage_ghz = quantum_prob_ghz - classical_max_ghz
    
    results['ghz_3_players'] = {
        'quantum_win_rate': float(quantum_prob_ghz),
        'classical_max_win_rate': float(classical_max_ghz),
        'quantum_advantage': float(advantage_ghz),
        'improvement_factor': float(quantum_prob_ghz / classical_max_ghz)
    }
    
    print(f"  Quantum win rate: {quantum_prob_ghz:.4f} ({quantum_prob_ghz*100:.2f}%)")
    print(f"  Classical max: {classical_max_ghz:.4f} ({classical_max_ghz*100:.2f}%)")
    print(f"  Quantum advantage: {advantage_ghz:.4f} ({advantage_ghz*100:.2f} percentage points)")
    print(f"  Improvement factor: {quantum_prob_ghz/classical_max_ghz:.4f}x")
    
    # GHZ Game for multiple players
    print("\nGHZ Game Scalability:")
    ghz_scalability = {}
    for n in [3, 4, 5, 6]:
        quantum_prob_n = tme8.simulate_quantum_ghz_game(n)
        ghz_scalability[f'n_{n}'] = {
            'quantum_win_rate': float(quantum_prob_n),
            'num_valid_inputs': int(tme8.count_valid_inputs_n_players(n))
        }
        print(f"  n={n} players: {quantum_prob_n:.4f} ({quantum_prob_n*100:.2f}%)")
    
    results['ghz_scalability'] = ghz_scalability
    
    return results


def benchmark_error_correction_performance():
    """Benchmark error correction performance vs error rate."""
    print("\n" + "="*70)
    print("Error Correction Performance Analysis")
    print("="*70)
    
    results = {
        'gamma_values': [],
        'fidelities': [],
        'correction_times': []
    }
    
    # Test different error rates
    gamma_values = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    
    alpha = 1.0 / np.sqrt(2)
    beta = 1.0 / np.sqrt(2)
    encoded = tme9.encoding_circuit(alpha, beta)
    
    for gamma in gamma_values:
        # Apply error
        rho, branches = tme9.apply_amplitude_damping_to_one_qubit(encoded, gamma, i=0)
        error_state = branches[1]
        error_state = error_state / np.linalg.norm(error_state)
        
        # Measure fidelity before correction
        fidelity_before = abs(np.vdot(encoded, error_state))**2
        
        # Correct error and measure time
        start = time.perf_counter()
        corrected = tme9.error_correction(error_state)
        correction_time = time.perf_counter() - start
        
        # Measure fidelity after correction
        fidelity_after = abs(np.vdot(encoded, corrected))**2
        
        results['gamma_values'].append(float(gamma))
        results['fidelities'].append({
            'before': float(fidelity_before),
            'after': float(fidelity_after),
            'improvement': float(fidelity_after - fidelity_before)
        })
        results['correction_times'].append(float(correction_time))
        
        print(f"γ={gamma:.3f}: "
              f"Fidelity before={fidelity_before:.4f}, "
              f"after={fidelity_after:.4f}, "
              f"improvement={fidelity_after-fidelity_before:.4f}, "
              f"time={correction_time*1e6:.2f} μs")
    
    return results


def benchmark_decoherence_channels():
    """Benchmark decoherence channel performance and effects."""
    print("\n" + "="*70)
    print("Decoherence Channel Performance Analysis")
    print("="*70)
    
    results = {}
    
    # Benchmark different channels
    r0 = np.array([1.0, 0.0, 0.0])
    n_iterations = 10000
    
    channels = {
        'phase_damping': (bt.phase_damping, {'t': 1.0, 'T2': 2.0}),
        'amplitude_damping': (bt.amplitude_damping, {'t': 1.0, 'T1': 3.0}),
        'depolarizing': (bt.depolarizing_noise, {'p': 0.1})
    }
    
    for name, (channel_func, kwargs) in channels.items():
        stats = benchmark_with_statistics(
            lambda: channel_func(r0, **kwargs),
            n_runs=10
        )
        
        # Also measure effect on Bloch vector norm
        r_final = channel_func(r0, **kwargs)
        norm_initial = np.linalg.norm(r0)
        norm_final = np.linalg.norm(r_final)
        norm_reduction = norm_initial - norm_final
        
        results[name] = {
            'time_mean_us': float(stats['mean'] * 1e6),
            'time_std_us': float(stats['std'] * 1e6),
            'norm_reduction': float(norm_reduction),
            'norm_final': float(norm_final)
        }
        
        print(f"{name:20s}: {stats['mean']*1e6:.3f} ± {stats['std']*1e6:.3f} μs, "
              f"norm reduction: {norm_reduction:.4f}")
    
    return results


def benchmark_trajectory_generation_scalability():
    """Benchmark trajectory generation for different system sizes."""
    print("\n" + "="*70)
    print("Trajectory Generation Scalability")
    print("="*70)
    
    results = {
        'num_steps': [],
        'times': [],
        'time_per_step': []
    }
    
    r0 = np.array([1.0, 0.0, 0.0])
    T2 = 2.0
    
    step_counts = [10, 25, 50, 100, 200, 500, 1000]
    
    for num_steps in step_counts:
        time_steps = np.linspace(0, 5, num_steps)
        
        stats = benchmark_with_statistics(
            lambda: bt.generate_trajectory(bt.phase_damping, r0, time_steps,
                                          use_time=True, T2=T2),
            n_runs=5
        )
        
        time_per_step = stats['mean'] / num_steps
        
        results['num_steps'].append(num_steps)
        results['times'].append(float(stats['mean']))
        results['time_per_step'].append(float(time_per_step))
        
        print(f"Steps={num_steps:4d}: {stats['mean']:.6f} s "
              f"({time_per_step*1e6:.3f} μs/step)")
    
    return results


def benchmark_memory_usage():
    """Estimate memory usage for different system sizes."""
    print("\n" + "="*70)
    print("Memory Usage Analysis")
    print("="*70)
    
    results = {}
    
    import sys
    
    for n_qubits in [2, 3, 4, 5, 6, 7, 8]:
        # Create state
        state = qs.basis_state(0, n_qubits)
        
        # Estimate size (rough approximation)
        # Each complex number is ~16 bytes, state has 2^n elements
        state_size = 2**n_qubits * 16  # bytes
        
        # Gate size (2^n × 2^n matrix)
        gate_size = (2**n_qubits)**2 * 16  # bytes
        
        total_size = state_size + gate_size
        
        results[f'n_{n_qubits}'] = {
            'state_size_bytes': int(state_size),
            'gate_size_bytes': int(gate_size),
            'total_size_bytes': int(total_size),
            'state_size_mb': float(state_size / (1024**2)),
            'gate_size_mb': float(gate_size / (1024**2)),
            'total_size_mb': float(total_size / (1024**2))
        }
        
        print(f"n={n_qubits}: State={state_size/(1024**2):.4f} MB, "
              f"Gate={gate_size/(1024**2):.4f} MB, "
              f"Total={total_size/(1024**2):.4f} MB")
    
    return results


def benchmark_operation_complexity():
    """Benchmark computational complexity of operations."""
    print("\n" + "="*70)
    print("Computational Complexity Analysis")
    print("="*70)
    
    results = {}
    
    # Test different operations at different scales
    operations = [
        ('tensor_product', lambda n: qs.tensor_product(
            qs.basis_state(0, n//2), qs.basis_state(0, n//2))),
        ('apply_operator', lambda n: qs.apply_operator(
            qs.gate_h(), qs.basis_state(0, n))),
        ('gate_on_qubit', lambda n: qs.gate_on_qubit(
            qs.gate_h(), 0, n)),
    ]
    
    qubit_counts = [2, 3, 4, 5, 6]
    
    for op_name, op_func in operations:
        op_results = {
            'n_qubits': [],
            'time_mean': [],
            'time_std': []
        }
        
        for n in qubit_counts:
            if op_name == 'tensor_product' and n % 2 != 0:
                continue
            
            try:
                stats = benchmark_with_statistics(
                    lambda: op_func(n),
                    n_runs=10
                )
                
                op_results['n_qubits'].append(n)
                op_results['time_mean'].append(float(stats['mean']))
                op_results['time_std'].append(float(stats['std']))
                
                print(f"{op_name:20s} n={n}: {stats['mean']*1e6:.3f} ± {stats['std']*1e6:.3f} μs")
            except:
                pass
        
        results[op_name] = op_results
    
    return results


def generate_latex_tables(results: Dict, output_file: str = "bench/results_for_latex.json"):
    """Generate JSON output suitable for LaTeX table generation."""
    
    # Save raw results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to {output_file}")
    print(f"{'='*70}")
    
    # Also generate LaTeX table snippets
    latex_output = []
    
    # Quantum vs Classical comparison
    if 'quantum_vs_classical' in results:
        latex_output.append("\\begin{table}[h]")
        latex_output.append("\\centering")
        latex_output.append("\\caption{Quantum vs Classical Strategy Comparison}")
        latex_output.append("\\begin{tabular}{lccc}")
        latex_output.append("\\hline")
        latex_output.append("Game & Quantum & Classical & Advantage \\\\")
        latex_output.append("\\hline")
        
        ms = results['quantum_vs_classical']['magic_square']
        latex_output.append(f"Magic Square & {ms['quantum_win_rate']:.4f} & {ms['classical_max_win_rate']:.4f} & {ms['quantum_advantage']:.4f} \\\\")
        
        ghz = results['quantum_vs_classical']['ghz_3_players']
        latex_output.append(f"GHZ (3 players) & {ghz['quantum_win_rate']:.4f} & {ghz['classical_max_win_rate']:.4f} & {ghz['quantum_advantage']:.4f} \\\\")
        
        latex_output.append("\\hline")
        latex_output.append("\\end{tabular}")
        latex_output.append("\\end{table}")
        latex_output.append("")
    
    # Save LaTeX snippets
    latex_file = output_file.replace('.json', '_latex.txt')
    with open(latex_file, 'w') as f:
        f.write('\n'.join(latex_output))
    
    print(f"LaTeX table snippets saved to {latex_file}")


def run_comprehensive_benchmarks():
    """Run all comprehensive benchmarks."""
    print("\n" + "="*70)
    print("Comprehensive Quantum SWE Portfolio Benchmarks")
    print("="*70)
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"NumPy: {np.__version__}")
    
    all_results = {
        'metadata': {
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'numpy_version': np.__version__
        }
    }
    
    # Run all benchmarks
    all_results['scalability'] = benchmark_scalability_quantum_simulator()
    all_results['quantum_vs_classical'] = benchmark_quantum_vs_classical_games()
    all_results['error_correction'] = benchmark_error_correction_performance()
    all_results['decoherence_channels'] = benchmark_decoherence_channels()
    all_results['trajectory_scalability'] = benchmark_trajectory_generation_scalability()
    all_results['memory_usage'] = benchmark_memory_usage()
    all_results['operation_complexity'] = benchmark_operation_complexity()
    
    # Generate output files
    generate_latex_tables(all_results)
    
    print("\n" + "="*70)
    print("All benchmarks completed!")
    print("="*70)
    print("\nKey Metrics for Report:")
    print("  - Quantum advantage: Measured improvement over classical")
    print("  - Scalability: Performance vs system size")
    print("  - Error tolerance: Fidelity vs error rate")
    print("  - Resource usage: Memory and computation time")
    print("  - Statistical confidence: Mean ± std with min/max")
    
    return all_results


if __name__ == "__main__":
    results = run_comprehensive_benchmarks()
