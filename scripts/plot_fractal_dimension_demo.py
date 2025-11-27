"""Generate a demonstration figure for fractal dimension calculations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from fractal_dimension.fractals import generate_koch_curve, generate_sierpinski_triangle

plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": True,
    }
)


def plot_fractal_dimension_demo(output: Path) -> None:
    """Demonstrate fractal dimension calculations for Koch and Sierpiński."""
    fig, axes = plt.subplots(2, 6, figsize=(30, 12))

    fractals = [
        ("Koch Curve", generate_koch_curve),
        ("Sierpiński Triangle", generate_sierpinski_triangle),
    ]

    for row, (fractal_name, generator_func) in enumerate(fractals):
        for col in range(6):
            iteration = col + 1
            ax = axes[row, col]
            if fractal_name == "Koch Curve":
                points = generator_func(iterations=iteration, samples_per_segment=50)
                linewidth = 2 + iteration * 0.25
                ax.plot(points[:, 0], points[:, 1], "k-", linewidth=linewidth)
            else:
                points = generator_func(iterations=iteration, samples_per_triangle=100)
                markersize = 1.5 + (4 - iteration) * 0.5
                ax.scatter(points[:, 0], points[:, 1], c="k", s=markersize)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 0.866 if fractal_name == "Sierpiński Triangle" else 1)
            ax.set_aspect("equal")
            ax.axis("off")
            if row == 0:
                ax.set_title(f"n = {iteration}", fontsize=60)

        fig.text(
            -0.025,
            0.75 - row * 0.5,
            fractal_name,
            va="center",
            ha="center",
            rotation=90,
            fontsize=40,
        )

    fig.subplots_adjust(left=0.1, wspace=0.1, hspace=0.1)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    plot_fractal_dimension_demo(output_dir / "fractal_dimension_demo.png")
