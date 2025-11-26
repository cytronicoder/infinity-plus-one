# Evaluating Fractal Dimension Estimates Across Scale Ranges

How does the choice of box-size window $[\varepsilon_{\min}, \varepsilon_{\max}]$ affect the box-counting dimension estimate $D_{\text{est}}$ for finite-iteration approximations $(n = 1..6)$ of the Koch curve and the Sierpiński triangle when the grid is anchored at $(0, 0)$ and scaled to $[0, 1]^2$? Using power-of-two scales $\varepsilon_i = 2^{-i}$ for $i = 1..10$, grids anchored at $(0, 0)$, and normalized geometries in $[0, 1]^2$, we fit log-log regressions across full, coarse, fine, and sliding windows.

We report:

- slope-based estimates $D_{\text{est}} = -m$
- coefficient of determination $R^2$
- residual diagnostics
- absolute and relative errors against the theoretical benchmarks

  - $D_{\text{Koch}} = \ln(4) / \ln(3)$
  - $D_{\text{Sier}} = \ln(3) / \ln(2)$

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
python scripts/run_analysis.py --output results --max-iter 6 --density 120
```

### 3. Inspect outputs

- `*_counts.csv` and `*_regressions.csv` for each fractal/iteration
- `*_loglog.png`, `*_residuals.png`, `*_windows.png` plots
- `iteration_accuracy.csv` and `iteration_accuracy.png` summarizing convergence toward theory

## License

Released under the MIT License. See [`LICENSE`](LICENSE) for details.

---

If you'd like, I can also generate a **GitHub README badge block**, **add a TOC**, **insert collapsible sections**, or **auto-convert formulas into SVG images** for perfect rendering.
