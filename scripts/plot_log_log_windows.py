"""Plot log-log scaling with multiple window fits for iteration 6."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(str(Path(__file__).parent))
from plot_config import get_window_color, get_window_label

plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": True,
        "font.size": 12,
    }
)


def fit_line(x, y):
    slope, intercept, _, _, _ = stats.linregress(x, y)
    return slope, intercept


def plot_log_log_windows(results_dir: Path, output_dir: Path) -> None:
    """Plot log-log scaling with window fits for Koch and Sierpinski (n=6)."""
    fractals = ["koch", "sierpinski"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, fractal in zip(axes, fractals):
        counts_file = results_dir / f"{fractal}_n6_counts.csv"
        regressions_file = results_dir / f"{fractal}_all_regressions.csv"

        if not counts_file.exists() or not regressions_file.exists():
            print(f"Skipping {fractal}: files not found")
            continue

        df_counts = pd.read_csv(counts_file)
        log_eps = df_counts["log_epsilon"].values
        log_counts = df_counts["log_counts"].values

        df_reg = pd.read_csv(regressions_file)
        df_reg = df_reg[df_reg["iteration"] == 6]

        ax.scatter(
            log_eps, log_counts, color="black", s=40, zorder=10, label="Data (n=6)"
        )

        windows = sorted(df_reg["window"].unique())
        main_order = ["full", "coarse", "fine"]
        sorted_windows = [w for w in main_order if w in windows] + [
            w for w in windows if w not in main_order
        ]

        for window in sorted_windows:
            row = df_reg[df_reg["window"] == window]
            if row.empty:
                continue

            slope = row["slope"].values[0]
            intercept = row["intercept"].values[0]
            d_est = row["d_est"].values[0]

            x_line = np.linspace(log_eps.min(), log_eps.max(), 100)
            y_line = slope * x_line + intercept

            color = get_window_color(window)
            label = get_window_label(window)

            ax.plot(
                x_line,
                y_line,
                "-",
                color=color,
                linewidth=2.5,
                label=f"{label}: D={d_est:.3f}",
                alpha=0.8,
                zorder=5,
            )

        ax.set_title(f"{fractal.replace('_', ' ').title()}", fontsize=18, pad=15)
        ax.set_xlabel(r"$\ln \varepsilon$", fontsize=16)
        if ax == axes[0]:
            ax.set_ylabel(r"$\ln B(\varepsilon)$", fontsize=16)

        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.grid(True, alpha=0.3, linestyle="--")

        ax.legend(
            fontsize=11,
            frameon=True,
            facecolor="white",
            framealpha=1.0,
            edgecolor="gray",
            loc="upper right",
        )

    plt.suptitle(r"Box-Counting Scaling with Window Fits (n=6)", fontsize=20, y=0.98)
    output_path = output_dir / "log_log_windows.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved log-log window plot to {output_path}")


if __name__ == "__main__":
    results_dir = Path(__file__).parent.parent / "results"
    output_dir = Path(__file__).parent.parent / "results"

    plot_log_log_windows(results_dir, output_dir)
