"""Plot convergence of error ln(e_W(n)) vs iteration n for different windows."""

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

THEORETICAL_DIMS = {
    "koch": np.log(4) / np.log(3),
    "sierpinski": np.log(3) / np.log(2),
}


def plot_convergence(results_dir: Path, output_dir: Path) -> None:
    """Plot ln(error) vs n for both fractals with linear fits."""
    fractals = ["koch", "sierpinski"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    for ax, fractal in zip(axes, fractals):
        df = pd.read_csv(results_dir / f"{fractal}_all_regressions.csv")
        d_true = THEORETICAL_DIMS[fractal]

        df["abs_error"] = (df["d_est"] - d_true).abs()
        df["log_error"] = np.log(df["abs_error"].replace(0, np.nan))

        windows = sorted(df["window"].unique())
        main_order = ["full", "coarse", "fine"]
        sorted_windows = [w for w in main_order if w in windows] + [
            w for w in windows if w not in main_order
        ]

        for window in sorted_windows:
            data = df[df["window"] == window].sort_values("iteration")
            data = data[np.isfinite(data["log_error"])]

            if len(data) < 2:
                continue

            color = get_window_color(window)
            label = get_window_label(window)

            slope, intercept, r_value, _, _ = stats.linregress(
                data["iteration"], data["log_error"]
            )

            ax.plot(
                data["iteration"],
                data["log_error"],
                "o",
                color=color,
                markersize=6,
                alpha=0.6,
                zorder=3,
            )

            x_fit = np.array([data["iteration"].min(), data["iteration"].max()])
            y_fit = slope * x_fit + intercept

            ax.plot(
                x_fit,
                y_fit,
                "-",
                color=color,
                linewidth=2,
                label=f"{label} ($R^2={r_value**2:.2f}$)",
                alpha=0.9,
                zorder=2,
            )

        ax.set_title(f"{fractal.replace('_', ' ').title()}", fontsize=18, pad=15)
        ax.set_xlabel("Iteration depth $n$", fontsize=16)
        if ax == axes[0]:
            ax.set_ylabel(r"Log Absolute Error $\ln e_W(n)$", fontsize=16)

        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.grid(True, alpha=0.3, linestyle="--")

        ax.legend(
            fontsize=11,
            frameon=True,
            facecolor="white",
            framealpha=1.0,
            edgecolor="gray",
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
        )

    plt.suptitle(
        r"Convergence of Fractal Dimension Estimates: $\ln e_W(n)$ vs $n$",
        fontsize=20,
        y=0.98,
    )
    output_path = output_dir / "convergence_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved convergence plot to {output_path}")


if __name__ == "__main__":
    results_dir = Path(__file__).parent.parent / "results"
    output_dir = Path(__file__).parent.parent / "results"

    plot_convergence(results_dir, output_dir)
