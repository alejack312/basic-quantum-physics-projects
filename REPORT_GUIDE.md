# LaTeX Report Generation Guide

## Overview

This guide explains how to compile and customize your quantum software engineering portfolio LaTeX report.

## Files Generated

### Main Report
- **`quantum_swe_portfolio_report.tex`** - Main LaTeX document

### LaTeX Tables (in `bench/latex_output/`)
- `table_quantum_vs_classical.tex` - Quantum vs Classical comparison
- `table_scalability.tex` - Scalability analysis
- `table_error_correction.tex` - Error correction performance
- `table_memory_usage.tex` - Memory usage vs qubits
- `table_operation_performance.tex` - Operation timing metrics
- `table_decoherence_channels.tex` - Decoherence channel performance
- `table_test_coverage.tex` - Test coverage summary

### LaTeX Figures (in `bench/latex_output/`)
- `plot_scalability.tex` - Scalability plot (log scale)
- `figure_quantum_advantage.tex` - Quantum advantage bar chart
- `figure_error_correction.tex` - Error correction fidelity plot

### Data Files (in `bench/latex_output/`)
- `plot_scalability.dat` - Scalability plot data
- `plot_error_correction.dat` - Error correction plot data

### Summary
- `summary_statistics.tex` - Statistical summary section

## Compiling the Report

### Option 1: Using pdflatex (Command Line)

```bash
# Navigate to project root
cd /path/to/BQPh

# Compile (run twice for references)
pdflatex quantum_swe_portfolio_report.tex
pdflatex quantum_swe_portfolio_report.tex

# Output: quantum_swe_portfolio_report.pdf
```

### Option 2: Using LaTeX Editor

1. Open `quantum_swe_portfolio_report.tex` in your LaTeX editor:
   - **Overleaf**: Upload the file and all `bench/latex_output/` files
   - **TeXstudio**: Open the file, ensure paths are correct
   - **VS Code**: Use LaTeX Workshop extension

2. Compile using your editor's build command

### Option 3: Using Overleaf

1. Create a new project in Overleaf
2. Upload `quantum_swe_portfolio_report.tex`
3. Create folder `bench/latex_output/` and upload all files from that directory
4. Compile

## Required LaTeX Packages

The report uses these packages (all standard):
- `amsmath`, `amsfonts`, `amssymb` - Math
- `graphicx` - Graphics
- `booktabs` - Professional tables
- `pgfplots` - Plots
- `siunitx` - Units
- `hyperref` - Links
- `geometry` - Page layout
- `float` - Figure/table positioning

## Customizing the Report

### Change Author Name

Edit line 20 in `quantum_swe_portfolio_report.tex`:
```latex
\author{Your Name}
```

### Add/Remove Sections

The main document includes these sections:
1. Introduction
2. Methodology
3. Quantum Advantage Results
4. Performance Analysis
5. Error Correction Analysis
6. Resource Requirements
7. Statistical Analysis
8. Conclusions

To modify, edit the corresponding `\section{}` commands.

### Regenerate Tables/Figures

If you update benchmark data:

```bash
# Regenerate all benchmark data
python bench/bench_comprehensive.py

# Process data for LaTeX
python bench/generate_report_data.py

# Generate LaTeX code
python bench/latex_helper.py

# Generate additional tables/figures
python bench/generate_additional_latex.py
```

## Key Metrics in the Report

### Quantum Advantages
- **Magic Square**: 11.11% advantage, 1.125x improvement
- **GHZ Game**: 25.00% advantage, 1.333x improvement

### Performance Metrics
- **Scalability**: Exponential with base ~2-4 per qubit
- **Operation Times**: Microsecond-level (10-200 μs)
- **Practical Limit**: 8-10 qubits

### Error Correction
- **>95% fidelity**: Up to γ = 0.15
- **>99% fidelity**: Up to γ = 0.10
- **Correction Overhead**: <60 μs

### Test Coverage
- **Total**: 167+ test cases
- **Types**: Unit, Integration, Performance

## Troubleshooting

### Error: File not found (table/figure files)

**Solution**: Ensure all files in `bench/latex_output/` are in the correct location relative to the main `.tex` file.

If compiling from a different directory, adjust paths:
```latex
% Instead of:
\input{bench/latex_output/table_scalability.tex}

% Use absolute or relative paths:
\input{./bench/latex_output/table_scalability.tex}
```

### Error: pgfplots not found

**Solution**: Install pgfplots package:
```bash
# On Linux (TeX Live)
sudo apt-get install texlive-pictures

# On Windows (MiKTeX)
# Package manager will prompt to install
```

### Error: Missing data files (.dat)

**Solution**: Ensure `plot_scalability.dat` and `plot_error_correction.dat` are in `bench/latex_output/`.

### Tables/Figures not appearing

**Solution**: 
1. Check that all `\input{}` commands use correct paths
2. Verify files exist in `bench/latex_output/`
3. Check LaTeX log for specific error messages

## Report Structure

```
quantum_swe_portfolio_report.tex
├── Abstract
├── Table of Contents
├── Section 1: Introduction
├── Section 2: Methodology
├── Section 3: Quantum Advantage Results
│   ├── Table: Quantum vs Classical
│   └── Figure: Quantum Advantage Bar Chart
├── Section 4: Performance Analysis
│   ├── Figure: Scalability Plot
│   ├── Table: Scalability Metrics
│   └── Table: Operation Performance
├── Section 5: Error Correction Analysis
│   ├── Table: Error Correction Performance
│   └── Figure: Error Correction Plot
├── Section 6: Resource Requirements
│   ├── Table: Memory Usage
│   └── Table: Decoherence Channels
├── Section 7: Statistical Analysis
│   ├── Summary Statistics
│   └── Table: Test Coverage
└── Section 8: Conclusions
```

## Next Steps

1. **Review the generated PDF** - Check all tables and figures render correctly
2. **Customize content** - Add your own analysis and insights
3. **Add references** - Include citations if needed
4. **Polish formatting** - Adjust spacing, fonts, etc.
5. **Export final PDF** - Ensure high-quality output

## Quick Reference

### Compile Command
```bash
pdflatex quantum_swe_portfolio_report.tex && pdflatex quantum_swe_portfolio_report.tex
```

### Regenerate All Data
```bash
python bench/bench_comprehensive.py && python bench/generate_report_data.py && python bench/latex_helper.py && python bench/generate_additional_latex.py
```

### Key Files Location
- Main report: `quantum_swe_portfolio_report.tex`
- Tables/Figures: `bench/latex_output/*.tex`
- Data files: `bench/latex_output/*.dat`
- Source data: `bench/results_for_latex.json`

## Support

If you encounter issues:
1. Check LaTeX log file (`quantum_swe_portfolio_report.log`)
2. Verify all required packages are installed
3. Ensure all data files are generated
4. Check file paths are correct

Good luck with your report!
