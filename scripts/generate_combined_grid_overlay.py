"""Generate combined grid overlay visualization for both fractals."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent.parent / "src"))

from fractal_dimension.fractals import (
    get_koch_vertices,
    sample_koch_with_epsilon,
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


def plot_combined_grid_overlay(output: Path, iteration: int = 3) -> None:
    """Visualize grid counting for both fractals at different scales in a combined figure."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    fractals = ["Koch Curve", "Sierpiński Triangle"]

    epsilons = [1 / 4, 1 / 16, 1 / 64]

    for row, fractal_name in enumerate(fractals):
        for col, eps in enumerate(epsilons):
            ax = axes[row, col]

            if fractal_name == "Koch Curve":
                verts = get_koch_vertices(iteration)
                points = sample_koch_with_epsilon(verts, eps)
            else:
                tris = get_sierpinski_triangles(iteration)
                points = sample_sierpinski_with_epsilon(tris, eps)

            pts_norm = (points - points.min(axis=0)) / (
                points.max(axis=0) - points.min(axis=0)
            ).max()

            hit_boxes = set()
            for p in pts_norm:
                x_idx = int(p[0] / eps)
                y_idx = int(p[1] / eps)
                if x_idx == int(1 / eps):
                    x_idx -= 1
                if y_idx == int(1 / eps):
                    y_idx -= 1
                hit_boxes.add((x_idx, y_idx))

            # For visualization, we plot the points.
            # For Sierpinski, it's a cloud, for Koch it's a line.
            if fractal_name == "Koch Curve":
                ax.plot(pts_norm[:, 0], pts_norm[:, 1], "k-", linewidth=1.2, alpha=0.9)
            else:
                ax.scatter(pts_norm[:, 0], pts_norm[:, 1], c="k", s=1, alpha=0.5)

            ticks = np.arange(0, 1.001, eps)
            for t in ticks:
                ax.axvline(t, color="black", linestyle="-", linewidth=1.5, alpha=0.8)
                ax.axhline(t, color="black", linestyle="-", linewidth=1.5, alpha=0.8)

            for bx, by in hit_boxes:
                rect = patches.Rectangle(
                    (bx * eps, by * eps),
                    eps,
                    eps,
                    linewidth=0,
                    edgecolor="none",
                    facecolor="#e74c3c",
                    alpha=0.5,
                )
                ax.add_patch(rect)

            ax.set_title(
                f"$\\varepsilon = 1/{int(1/eps)}$\t$N(\\varepsilon) = {len(hit_boxes)}$",
                fontsize=30,
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.axis("off")

        fig.text(
            0.02,
            0.75 - row * 0.5,
            fractal_name,
            va="center",
            ha="center",
            rotation=90,
            fontsize=30,
        )

    fig.subplots_adjust(hspace=1, left=0.15)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    plot_combined_grid_overlay(output_dir / "combined_grid_overlay.png")
