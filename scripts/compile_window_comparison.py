"""Compile comparison of box-size windows across iterations for each fractal."""

from __future__ import annotations

import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")

THEORETICAL_DIMS = {
    "koch": 1.2618595071429148,  # log(4)/log(3)
    "sierpinski": 1.584962500721156,  # log(3)/log(2)
}


def compile_window_comparison():
    """Compile window comparison stats for each fractal."""
    for fractal, d_theory in THEORETICAL_DIMS.items():
        regressions_file = RESULTS_DIR / f"{fractal}_all_regressions.csv"
        if not regressions_file.exists():
            print(f"Skipping {fractal}: {regressions_file} not found.")
            continue

        df = pd.read_csv(regressions_file)

        window_stats = []
        for window, group in df.groupby("window"):
            d_ests = group["d_est"]
            abs_errors = (d_ests - d_theory).abs()
            rel_errors = abs_errors / d_theory

            stats = {
                "window": window,
                "mean_D_est": d_ests.mean(),
                "std_D_est": d_ests.std(),
                "MAE": abs_errors.mean(),
                "MRE": rel_errors.mean(),
                "mean_std_err": group["std_err"].mean(),
            }
            window_stats.append(stats)

        stats_df = pd.DataFrame(window_stats)
        output_file = RESULTS_DIR / f"{fractal}_window_comparison.csv"
        stats_df.to_csv(output_file, index=False)
        print(f"Saved {output_file}")


if __name__ == "__main__":
    compile_window_comparison()
