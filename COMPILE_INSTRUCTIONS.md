# Quick Compile Instructions

## One-Command Compilation

```bash
# From project root directory
pdflatex quantum_swe_portfolio_report.tex && pdflatex quantum_swe_portfolio_report.tex
```

## What Was Generated

✅ **Main Report**: `quantum_swe_portfolio_report.tex`
✅ **11 LaTeX Files**: All tables and figures in `bench/latex_output/`
✅ **2 Data Files**: Plot data in `bench/latex_output/`
✅ **Documentation**: Complete guides and summaries

## Files Included in Report

### Tables (7)
- Quantum vs Classical comparison
- Scalability analysis
- Error correction performance
- Memory usage
- Operation performance
- Decoherence channels
- Test coverage

### Figures (3)
- Scalability plot (log scale)
- Quantum advantage bar chart
- Error correction fidelity plot

## Key Metrics

- **Quantum Advantages**: 11.11% and 25% improvements
- **Scalability**: Exponential with base ~2-4 per qubit
- **Error Correction**: >95% fidelity up to γ = 0.15
- **Test Coverage**: 167+ comprehensive test cases

## Troubleshooting

If compilation fails:
1. Check that all files in `bench/latex_output/` exist
2. Verify LaTeX packages are installed (pgfplots, booktabs, etc.)
3. Check paths are correct (relative to main .tex file)

## Full Documentation

See `REPORT_GUIDE.md` for detailed instructions.
