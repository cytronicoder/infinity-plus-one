"""Collate individual CSVs into mega-CSVs for batch analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")

FRACTALS = ["koch", "sierpinski"]


def collate_csvs():
    """Collate CSVs by fractal type into mega-CSVs."""
    for fractal in FRACTALS:
        counts_files = sorted(RESULTS_DIR.glob(f"{fractal}_n*_counts.csv"))
        counts_dfs = []
        for file in counts_files:
            df = pd.read_csv(file)
            iteration = int(file.stem.split("_n")[1].split("_")[0])
            df["iteration"] = iteration
            counts_dfs.append(df)
        if counts_dfs:
            mega_counts = pd.concat(counts_dfs, ignore_index=True)
            mega_counts.to_csv(RESULTS_DIR / f"{fractal}_all_counts.csv", index=False)

        regressions_files = sorted(RESULTS_DIR.glob(f"{fractal}_n*_regressions.csv"))
        regressions_dfs = []
        for file in regressions_files:
            df = pd.read_csv(file)
            iteration = int(file.stem.split("_n")[1].split("_")[0])
            df["iteration"] = iteration
            regressions_dfs.append(df)
        if regressions_dfs:
            mega_regressions = pd.concat(regressions_dfs, ignore_index=True)
            mega_regressions.to_csv(
                RESULTS_DIR / f"{fractal}_all_regressions.csv", index=False
            )

        density_files = sorted(RESULTS_DIR.glob(f"{fractal}_n*_density_check.csv"))
        density_dfs = []
        for file in density_files:
            df = pd.read_csv(file)
            iteration = int(file.stem.split("_n")[1].split("_")[0])
            df["iteration"] = iteration
            density_dfs.append(df)
        if density_dfs:
            mega_density = pd.concat(density_dfs, ignore_index=True)
            mega_density.to_csv(
                RESULTS_DIR / f"{fractal}_all_density_checks.csv", index=False
            )


if __name__ == "__main__":
    collate_csvs()
