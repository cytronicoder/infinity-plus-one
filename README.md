# Evaluating Fractal Dimension Estimates Across Scale Ranges

![lint](https://img.shields.io/badge/style-black-000000.svg)
![tests](https://img.shields.io/badge/tests-pytest-brightgreen.svg)

## Abstract
This mini-paper investigates how the choice of box-size window \([\varepsilon_{\min},\varepsilon_{\max}]\) influences box-counting estimates of fractal dimension for finite-iteration Koch curves and Sierpiński triangles. Using power-of-two scales \(\varepsilon_i = 2^{-i}\) for \(i = 1..10\), grids anchored at \((0, 0)\), and normalized geometries in \([0, 1]^2\), we fit log–log regressions across full, coarse, fine, and sliding windows. We report slope-based estimates \(D_{\text{est}}=-m\), coefficient of determination \(R^2\), residual diagnostics, and absolute/relative errors against theoretical benchmarks \(D_{\text{Koch}} = \ln 4 / \ln 3\) and \(D_{\text{Sier}} = \ln 3 / \ln 2\).

## Research question
**How does the choice of box-size window \([\varepsilon_{\min},\varepsilon_{\max}]\) affect the box-counting dimension estimate \(D_{\text{est}}\) for finite-iteration approximations (\(n = 1..6\)) of the Koch curve and the Sierpiński triangle when the grid is anchored at \((0, 0)\) and scaled to \([0, 1]^2\)?**

## IA structure
The repository mirrors the IB Mathematics AA HL Internal Assessment flow:
1. **Introduction** – motivation for comparing window choices in box-counting.
2. **Theory** – definition of box-count dimension, benchmark dimensions \(D_{\text{Koch}}\) and \(D_{\text{Sier}}\), and regression model \(\ln N(\varepsilon) = m \ln \varepsilon + b\).
3. **Method** – fractal generation for \(n = 1..6\), sampling density controls, box-counting with \(\varepsilon_i = 2^{-i}\), and predefined windows (full, coarse, mid-scale "fine" on \(i=3..7\), sliding width-5).
4. **Analysis** – computation of \(D_{\text{est}} = -m\), \(R^2\), residuals, and error metrics \(E_{\text{abs}}\), \(E_{\text{rel}}\).
5. **Discussion** – sensitivity of estimates to window width and iteration depth; stability under resampling density changes.
6. **Conclusion** – practical recommendations for IA-scale box-counting experiments.

See `report/IA.md` (source) and `report/IA.pdf` (export) for the full manuscript.

## Reproduce results
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Run the end-to-end pipeline (produces tables and plots in `results/`):
   ```bash
   python scripts/run_analysis.py --output results --max-iter 6 --density 120
   ```
3. Inspect outputs:
   - `*_counts.csv` and `*_regressions.csv` for each fractal/iteration
   - `*_loglog.png`, `*_residuals.png`, `*_windows.png` plots
   - `iteration_accuracy.csv` and `iteration_accuracy.png` summarizing convergence toward theory

## Repository layout
- `report/` – IA manuscript source (`IA.md`) and exported PDF (`IA.pdf`).
- `src/fractal_dimension/` – fractal generators, box-counting, regression utilities, and experiment runner.
- `scripts/run_analysis.py` – reproducible pipeline that generates all tables and figures.
- `notebooks/` – space for exploratory analysis (paired with the pipeline script).
- `tests/` – quick checks for monotonicity, convergence toward theoretical dimensions, and resampling stability.
- `results/` – generated outputs (deterministic filenames for each run).
- `requirements.txt`, `LICENSE`, `CITATION.cff`, `CONTRIBUTING.md` – reproducibility and contribution metadata.

## License
This project is released under the MIT License. See `LICENSE` for details.
