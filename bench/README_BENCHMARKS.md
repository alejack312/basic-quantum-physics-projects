# Comprehensive Benchmarking Suite

This directory contains comprehensive benchmarking tools for generating measurable outcomes for a quantum SWE portfolio LaTeX report.

## Quick Start

```bash
# Run all comprehensive benchmarks
python bench/bench_comprehensive.py

# Generate LaTeX-ready data
python bench/generate_report_data.py

# Generate LaTeX code
python bench/latex_helper.py
```

## Files

### Benchmark Scripts

1. **bench_comprehensive.py**: Main benchmarking suite
   - Quantum vs Classical comparisons
   - Scalability analysis
   - Error correction performance
   - Resource usage profiling
   - Statistical analysis

2. **generate_report_data.py**: Data processing for LaTeX
   - Processes raw benchmark results
   - Generates LaTeX-ready formats
   - Creates summary statistics

3. **latex_helper.py**: LaTeX code generation
   - Generates LaTeX tables
   - Creates pgfplots data and code
   - Generates statistical summaries

### Documentation

1. **REPORT_METRICS.md**: Detailed metrics to include in report
2. **ADDITIONAL_BENCHMARKS.md**: Additional benchmarking suggestions
3. **BENCHMARK_SUGGESTIONS.md**: Quick reference for key metrics

## Key Metrics Generated

### Quantum Advantage
- Win rates: Quantum vs Classical
- Advantage percentages
- Improvement factors

### Scalability
- Time vs. number of qubits
- Memory vs. number of qubits
- Scaling factors

### Error Correction
- Fidelity vs. error rate
- Error tolerance thresholds
- Correction performance

### Performance
- Operation timings (mean ± std)
- Throughput metrics
- Resource usage

## Output Files

All results are saved to:
- `bench/results_for_latex.json`: Raw benchmark data
- `bench/report_data.json`: Processed LaTeX-ready data
- `bench/latex_output/`: Generated LaTeX code

## Usage in LaTeX Report

1. Run benchmarks to generate data
2. Use generated LaTeX code in your report
3. Include tables and figures from `latex_output/`
4. Reference statistics from summary files

## Example LaTeX Integration

```latex
% Include generated table
\input{bench/latex_output/table_quantum_vs_classical.tex}

% Include generated plot
\input{bench/latex_output/plot_scalability.tex}

% Include statistics summary
\input{bench/latex_output/summary_statistics.tex}
```
