"""Generate a demonstration figure for fractal dimension calculations."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent / "src"))

from fractal_dimension.fractals import (
    get_koch_vertices,
    get_sierpinski_triangles,
    sample_sierpinski_with_epsilon,
)

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

    fractals = ["Koch Curve", "Sierpiński Triangle"]

    for row, fractal_name in enumerate(fractals):
        for col in range(6):
            iteration = col + 1
            ax = axes[row, col]
            if fractal_name == "Koch Curve":
                # For Koch, we just plot the vertices as a line
                points = get_koch_vertices(iteration)
                linewidth = 2 + iteration * 0.25
                ax.plot(points[:, 0], points[:, 1], "k-", linewidth=linewidth)
            else:
                # For Sierpinski, we sample with a fine epsilon to show the shape
                tris = get_sierpinski_triangles(iteration)
                # Use a small epsilon for visualization purposes
                points = sample_sierpinski_with_epsilon(tris, epsilon=1/128)
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
