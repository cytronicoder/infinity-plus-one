"""Plot window sensitivity σ(n) vs iteration depth n for both fractals."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": True,
    }
)


def calculate_window_sensitivity(df: pd.DataFrame, iteration: int) -> float:
    """Calculate σ_window(n) from sliding window estimates.
    
    Args:
        df: Regression results dataframe
        iteration: Iteration depth n
        
    Returns:
        Standard deviation of sliding window D estimates
    """
    iteration_data = df[df["iteration"] == iteration]
    sliding_windows = iteration_data[
        iteration_data["window"].str.startswith("slide_")
    ]
    d_estimates = sliding_windows["d_est"].values
    return np.std(d_estimates, ddof=0)


def plot_window_sensitivity(results_dir: Path, output: Path) -> None:
    """Plot σ(n) vs n for both Koch curve and Sierpiński triangle."""
    koch_df = pd.read_csv(results_dir / "koch_all_regressions.csv")
    sierpinski_df = pd.read_csv(results_dir / "sierpinski_all_regressions.csv")
    iterations = range(1, 7)
    
    koch_sigma = [calculate_window_sensitivity(koch_df, n) for n in iterations]
    sierpinski_sigma = [
        calculate_window_sensitivity(sierpinski_df, n) for n in iterations
    ]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        iterations,
        koch_sigma,
        "o-",
        label="Koch curve",
        linewidth=2,
        markersize=8,
        color="#2E86AB",
    )
    ax.plot(
        iterations,
        sierpinski_sigma,
        "s-",
        label="Sierpiński triangle",
        linewidth=2,
        markersize=8,
        color="#A23B72",
    )
    
    ax.set_xlabel("Iteration depth $n$", fontsize=12)
    ax.set_ylabel(r"Window sensitivity $\sigma_{\mathrm{window}}(n)$", fontsize=12)
    ax.set_xticks(iterations)
    ax.legend(fontsize=11, frameon=False)
    ax.grid(True, alpha=0.3, linestyle="--")
    
    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved window sensitivity plot to {output}")
    
    print("\nWindow sensitivity values:")
    print("\nKoch curve:")
    for n, sigma in zip(iterations, koch_sigma):
        print(f"  n={n}: σ = {sigma:.4f}")
    
    print("\nSierpiński triangle:")
    for n, sigma in zip(iterations, sierpinski_sigma):
        print(f"  n={n}: σ = {sigma:.4f}")


if __name__ == "__main__":
    results_dir = Path(__file__).parent.parent / "results"
    output_dir = Path(__file__).parent.parent / "results"
    output_path = output_dir / "window_sensitivity.png"
    
    plot_window_sensitivity(results_dir, output_path)
