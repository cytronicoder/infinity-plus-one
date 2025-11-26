"""End-to-end pipeline to reproduce IA figures and tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from fractal_dimension.fractals import DEFAULT_SPECS
from fractal_dimension.pipeline import FractalExperiment
from fractal_dimension.regression import fit_scaling_relationship

sns.set_theme(style="whitegrid")


def plot_log_log(counts: pd.DataFrame, title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(-counts["log_epsilon"], counts["log_counts"], marker="o")
    ax.set_xlabel(r"$-\ln(\varepsilon)$")
    ax.set_ylabel(r"$\ln N(\varepsilon)$")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_residuals(counts: pd.DataFrame, output: Path) -> None:
    result = fit_scaling_relationship(
        counts["log_epsilon"].to_numpy(), counts["log_counts"].to_numpy()
    )
    predicted = result.intercept + result.slope * counts["log_epsilon"]
    residuals = counts["log_counts"] - predicted
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.axhline(0, color="black", linewidth=1)
    ax.scatter(-counts["log_epsilon"], residuals)
    ax.set_xlabel(r"$-\ln(\varepsilon)$")
    ax.set_ylabel("Residual")
    ax.set_title("Residual diagnostics (full window)")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_window_estimates(regressions: pd.DataFrame, output: Path) -> None:
    window_rows = regressions[regressions["window"].str.startswith("slide")].copy()
    window_rows["index"] = window_rows["window"].str.extract(r"(\d+)").astype(int)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(window_rows["index"], window_rows["d_est"], marker="o", label=r"$D_{est}$")
    ax.set_xlabel("Sliding window index")
    ax.set_ylabel(r"$D_{est}$")
    ax.set_title("Window-by-window slope estimates (width = 5)")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_iteration_accuracy(summary: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.lineplot(
        data=summary, x="iteration", y="abs_error", hue="fractal", marker="o", ax=ax
    )
    ax.set_ylabel(r"$E_{abs}$")
    ax.set_title("Iteration depth vs. absolute error (fine window)")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def run_pipeline(output_dir: Path, max_iter: int = 6) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    exp = FractalExperiment()
    summary_rows = []

    for spec in DEFAULT_SPECS:
        for iteration in range(1, max_iter + 1):
            print(f"Processing {spec.name} iteration {iteration}...")
            result = exp.run(spec.name, iterations=iteration)
            prefix = f"{spec.name}_n{iteration}"

            counts_path = output_dir / f"{prefix}_counts.csv"
            result.counts.to_csv(counts_path, index=False)

            regress_path = output_dir / f"{prefix}_regressions.csv"
            result.regressions.to_csv(regress_path, index=False)

            plot_log_log(
                result.counts,
                f"{spec.name.title()} n={iteration}",
                output_dir / f"{prefix}_loglog.png",
            )
            plot_residuals(result.counts, output_dir / f"{prefix}_residuals.png")
            plot_window_estimates(
                result.regressions, output_dir / f"{prefix}_windows.png"
            )

            fine_row = result.regressions[result.regressions["window"] == "fine"].iloc[
                0
            ]
            summary_rows.append(
                {
                    "fractal": spec.name,
                    "iteration": iteration,
                    "d_est": fine_row.d_est,
                    "abs_error": fine_row.abs_error,
                    "rel_error": fine_row.rel_error,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "iteration_accuracy.csv", index=False)
    plot_iteration_accuracy(summary_df, output_dir / "iteration_accuracy.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce IA tables and figures")
    parser.add_argument(
        "--output", type=Path, default=Path("results"), help="Directory for outputs"
    )
    parser.add_argument(
        "--max-iter", type=int, default=6, help="Maximum iteration depth"
    )
    args = parser.parse_args()
    run_pipeline(args.output, max_iter=args.max_iter)
