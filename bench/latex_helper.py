"""
Helper functions to generate LaTeX code from benchmark results.

This module provides functions to generate LaTeX tables, figures, and
statistical summaries from benchmark data.
"""
import json
from pathlib import Path
from typing import Dict, List


def generate_table_latex(headers: List[str], rows: List[List[str]], 
                        caption: str, label: str) -> str:
    """Generate LaTeX table code."""
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:{label}}}",
        "\\begin{tabular}{" + "c" * len(headers) + "}",
        "\\hline",
        " & ".join(headers) + " \\\\",
        "\\hline"
    ]
    
    for row in rows:
        lines.append(" & ".join(str(x) for x in row) + " \\\\")
    
    lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)


def generate_pgfplot_data(x_data: List[float], y_data: List[float],
                         y_err: List[float] = None, xlabel: str = "x",
                         ylabel: str = "y", title: str = "") -> str:
    """Generate pgfplots data file format."""
    lines = [
        f"% Data for {title}",
        f"% xlabel: {xlabel}, ylabel: {ylabel}",
        ""
    ]
    
    if y_err:
        lines.append("x y yerr")
        for x, y, err in zip(x_data, y_data, y_err):
            lines.append(f"{x} {y} {err}")
    else:
        lines.append("x y")
        for x, y in zip(x_data, y_data):
            lines.append(f"{x} {y}")
    
    return "\n".join(lines)


def generate_pgfplot_code(data_file: str, xlabel: str, ylabel: str,
                         title: str, log_x: bool = False,
                         log_y: bool = False) -> str:
    """Generate pgfplots LaTeX code."""
    log_axis = ""
    if log_x and log_y:
        log_axis = "loglog"
    elif log_x:
        log_axis = "semilogx"
    elif log_y:
        log_axis = "semilogy"
    else:
        log_axis = ""
    
    # Build axis environment name
    axis_env = f"{log_axis}axis" if log_axis else "axis"
    
    lines = [
        "\\begin{figure}[h]",
        "\\centering",
        "\\begin{tikzpicture}",
        f"\\begin{{{axis_env}}}[",
        f"    xlabel={{{xlabel}}},",
        f"    ylabel={{{ylabel}}},",
        "    grid=major,",
        "    legend pos=north west",
        "]",
        f"\\addplot[mark=*, blue, thick] table[x=x, y=y] {{{data_file}}};",
        f"\\addlegendentry{{{title}}}",
        f"\\end{{{axis_env}}}",
        "\\end{tikzpicture}",
        f"\\caption{{{title}}}",
        f"\\label{{fig:{title.lower().replace(' ', '_')}}}",
        "\\end{figure}"
    ]
    
    return "\n".join(lines)


def generate_statistics_summary(results: Dict) -> str:
    """Generate LaTeX summary of statistics."""
    lines = [
        "\\section{Summary Statistics}",
        "",
        "\\begin{itemize}",
    ]
    
    if 'quantum_vs_classical' in results:
        qvc = results['quantum_vs_classical']
        ms = qvc['magic_square']
        ghz = qvc['ghz_3_players']
        
        lines.extend([
            f"\\item Magic Square Game: Quantum achieves {ms['quantum_win_rate']*100:.2f}\\% "
            f"vs. classical maximum of {ms['classical_max_win_rate']*100:.2f}\\% "
            f"(improvement factor: {ms['improvement_factor']:.3f}x)",
            f"\\item GHZ Game (3 players): Quantum achieves {ghz['quantum_win_rate']*100:.2f}\\% "
            f"vs. classical maximum of {ghz['classical_max_win_rate']*100:.2f}\\% "
            f"(improvement factor: {ghz['improvement_factor']:.3f}x)",
        ])
    
    if 'scalability' in results:
        scal = results['scalability']
        if len(scal['time_mean']) > 0:
            avg_time = sum(scal['time_mean']) / len(scal['time_mean'])
            lines.append(
                f"\\item Average simulation time: {avg_time:.6f} seconds "
                f"across {len(scal['time_mean'])} configurations"
            )
    
    lines.extend([
        "\\end{itemize}",
        ""
    ])
    
    return "\n".join(lines)


def load_results(file_path: str = "bench/results_for_latex.json") -> Dict:
    """Load benchmark results from JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {file_path}")
    
    with open(path, 'r') as f:
        return json.load(f)


def generate_all_latex(output_dir: str = "bench/latex_output"):
    """Generate all LaTeX code from benchmark results."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load results
    results = load_results()
    
    # Generate tables
    if 'quantum_vs_classical' in results:
        qvc = results['quantum_vs_classical']
        table = generate_table_latex(
            headers=['Game', 'Quantum', 'Classical', 'Advantage (\\%)'],
            rows=[
                ['Magic Square',
                 f"{qvc['magic_square']['quantum_win_rate']:.4f}",
                 f"{qvc['magic_square']['classical_max_win_rate']:.4f}",
                 f"{qvc['magic_square']['quantum_advantage']*100:.2f}"],
                ['GHZ (3 players)',
                 f"{qvc['ghz_3_players']['quantum_win_rate']:.4f}",
                 f"{qvc['ghz_3_players']['classical_max_win_rate']:.4f}",
                 f"{qvc['ghz_3_players']['quantum_advantage']*100:.2f}"]
            ],
            caption="Quantum vs Classical Strategy Comparison",
            label="quantum_vs_classical"
        )
        
        with open(output_path / "table_quantum_vs_classical.tex", 'w') as f:
            f.write(table)
    
    # Generate plot data
    if 'scalability' in results:
        scal = results['scalability']
        plot_data = generate_pgfplot_data(
            x_data=scal['n_qubits'],
            y_data=scal['time_mean'],
            y_err=scal['time_std'],
            xlabel="Number of Qubits",
            ylabel="Time (seconds)",
            title="Quantum Simulator Scalability"
        )
        
        with open(output_path / "plot_scalability.dat", 'w') as f:
            f.write(plot_data)
        
        plot_code = generate_pgfplot_code(
            data_file="plot_scalability.dat",
            xlabel="Number of Qubits",
            ylabel="Time (seconds)",
            title="Quantum Simulator Scalability",
            log_y=True
        )
        
        with open(output_path / "plot_scalability.tex", 'w') as f:
            f.write(plot_code)
    
    # Generate statistics summary
    summary = generate_statistics_summary(results)
    with open(output_path / "summary_statistics.tex", 'w') as f:
        f.write(summary)
    
    print(f"LaTeX files generated in {output_path}/")
    print("  - table_quantum_vs_classical.tex")
    print("  - plot_scalability.dat and .tex")
    print("  - summary_statistics.tex")


if __name__ == "__main__":
    generate_all_latex()
