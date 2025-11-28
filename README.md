# Evaluating Fractal Dimension Estimates Across Scale Ranges

How does the choice of box-size window $[\varepsilon_{\min}, \varepsilon_{\max}]$ affect the box-counting dimension estimate $D_{\text{est}}$ for finite-iteration approximations $(n = 1..6)$ of the Koch curve and the Sierpiński triangle when the grid is anchored at $(0, 0)$ and scaled to $[0, 1]^2$? Using power-of-two scales $\varepsilon_i = 2^{-i}$ for $i = 1..10$, grids anchored at $(0, 0)$, and normalized geometries in $[0, 1]^2$, we fit log-log regressions across full, coarse, fine, and sliding windows.

We report:

- slope-based estimates $D_{\text{est}} = -m$
- coefficient of determination $R^2$
- residual diagnostics (RMSE, max residual)
- absolute and relative errors against the theoretical benchmarks

  - $D_{\text{Koch}} = \ln(4) / \ln(3) \approx 1.2619$
  - $D_{\text{Sier}} = \ln(3) / \ln(2) \approx 1.5850$

## Reproduce Results

### 1. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the end-to-end pipeline

(Produces tables and plots in `results/`.)

```bash
python scripts/run_analysis.py --output results --max-iter 6 --quick-density
```

Arguments:

- `--output`: Directory for output files (default: `results`)
- `--max-iter`: Maximum iteration depth (default: 6)
- `--quick-density`: Sample a subset of epsilons for density check (faster)
- `--parallel`: Run density check in parallel

### 3. Inspect outputs

**Data Tables:**

- `iteration_accuracy.csv`: Summary of convergence metrics (D_est, Error, RMSE, Stability) for each iteration.
- `*_all_residuals.csv`: Detailed residual data for every epsilon and iteration.
- `*_window_residuals.csv`: Residuals for specific window presets (fine, coarse, full).
- `*_counts.csv` and `*_regressions.csv`: Raw box counts and regression results for each fractal/iteration.

**Visualizations:**

- `*_residuals_scatter.png`: Scatter plot of residuals vs. epsilon, colored by iteration.
- `*_window_residuals_scatter.png`: Scatter plot of residuals vs. epsilon, colored by window type.
- `*_loglog.png`: Log-log plots of box counts vs. epsilon.
- `*_convergence_loglog.png`: Combined log-log plots for multiple iterations.
- `*_error_heatmap.png`: Heatmap of relative errors across different window starts.

## License

Released under the MIT License. See [`LICENSE`](LICENSE) for details.
