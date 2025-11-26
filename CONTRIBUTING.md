# Contributing

Thank you for improving this IA mini-paper. The repository is organized so that the mathematical framing and computational experiments stay aligned with the Internal Assessment structure.

## How to contribute
1. Fork the repository and create a feature branch.
2. Install dependencies with `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
3. Run `python scripts/run_analysis.py --output results` to regenerate figures and tables.
4. Add or update tests in `tests/` and run `pytest` before opening a pull request.

## Extending the study
- **Grid alignment**: Add alternative anchoring strategies in `fractal_dimension/boxcount.py` (e.g., offsets or randomized starts) and expose them through `FractalExperiment` while keeping the IA baseline anchored at `(0, 0)`.
- **Epsilon schedules**: Modify the power-of-two sequence in `FractalExperiment` or allow geometric sequences; document the new window definitions so variable names (ε, N(ε)) remain unchanged.
- **Additional fractals**: Implement new generators in `fractal_dimension/fractals.py` and register them in `FractalExperiment.specs`, providing the theoretical dimension used for benchmarking.
- **Alternative regressions**: Swap the least-squares fit in `fractal_dimension/regression.py` for robust or weighted variants, but retain the reporting of slope `m`, dimension estimate `D_est = -m`, `R^2`, residuals, and absolute/relative errors.

Please keep docstrings, type hints, and the IA naming conventions consistent throughout the code and manuscript.
