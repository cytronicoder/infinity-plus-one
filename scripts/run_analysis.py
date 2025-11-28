"""End-to-end pipeline to reproduce IA figures and tables."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fractal_dimension.fractals import (
    DEFAULT_SPECS,
    generate_koch_curve,
    generate_sierpinski_triangle,
)
from fractal_dimension.pipeline import FractalExperiment
from fractal_dimension.regression import fit_scaling_relationship

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

sns.set_theme(style="ticks", context="paper", font_scale=1.2)
plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": True,
    }
)


def plot_log_log(counts: pd.DataFrame, title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        counts["log_epsilon"],
        counts["log_counts"],
        marker="o",
        linestyle="",
        color="black",
        markersize=4,
    )
    result = fit_scaling_relationship(
        counts["log_epsilon"].to_numpy(), counts["log_counts"].to_numpy()
    )
    log_eps_sorted = np.sort(counts["log_epsilon"])
    predicted = result.intercept + result.slope * log_eps_sorted
    ax.plot(
        log_eps_sorted,
        predicted,
        linestyle="-",
        color="red",
        linewidth=1.5,
        label=f"Best fit: m = {result.slope:.3g}\nD = -m = {-result.slope:.3g}",
    )
    ax.set_xlabel(r"$\ln(\varepsilon)$")
    ax.set_ylabel(r"$\ln N(\varepsilon)$")
    ax.set_title(f"Box-Counting Scaling Relationship for {title}")
    ax.legend(frameon=True, handlelength=1, handletextpad=0.5)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_residuals(counts: pd.DataFrame, output: Path) -> None:
    result = fit_scaling_relationship(
        counts["log_epsilon"].to_numpy(), counts["log_counts"].to_numpy()
    )
    predicted = result.intercept + result.slope * counts["log_epsilon"]
    residuals = counts["log_counts"] - predicted
    rmse = np.sqrt(np.mean(residuals**2))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.scatter(counts["log_epsilon"], residuals, color="black", s=20, label=f"Residuals (RMSE={rmse:.3g})")
    ax.set_xlabel(r"$\ln(\varepsilon)$")
    ax.set_ylabel("Residual")
    ax.set_title("Residual Analysis (Full Window)")
    ax.legend(frameon=True, handlelength=1, handletextpad=0.5)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_window_estimates(regressions: pd.DataFrame, output: Path) -> None:
    window_rows = regressions[regressions["window"].str.startswith("slide")].copy()
    window_rows["index"] = window_rows["window"].str.extract(r"(\d+)").astype(int)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(
        window_rows["index"],
        window_rows["d_est"],
        marker="o",
        linestyle="",
        label=r"$D_{est}$",
        color="black",
    )
    ax.set_xlabel("Sliding window index")
    ax.set_ylabel(r"$D_{est}$")
    ax.set_title("Sliding Window Slope Estimates (Width = 5)")
    ax.legend(frameon=True, handlelength=1, handletextpad=0.5)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_iteration_accuracy(summary: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.scatterplot(
        data=summary,
        x="iteration",
        y="abs_error",
        hue="fractal",
        marker="o",
        ax=ax,
        palette="viridis",
    )
    ax.set_ylabel(r"$E_{abs}$")
    ax.set_title("Iteration vs. Absolute Error (Fine Window)")
    ax.legend(frameon=True, handlelength=1, handletextpad=0.5)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_grid_overlay(
    fractal_name: str, generator_func, output: Path, iteration: int = 3
) -> None:
    """Visualize grid counting for a fractal at different scales."""
    if fractal_name == "koch":
        points = generator_func(iterations=iteration, samples_per_segment=100)
    else:
        points = generator_func(iterations=iteration, samples_per_triangle=100)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(
        f"Grid Overlay: {fractal_name.title()} (n={iteration})", fontsize=14, y=1.05
    )

    epsilons = [1 / 4, 1 / 16, 1 / 64]

    for ax, eps in zip(axes, epsilons):
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

        ax.plot(pts_norm[:, 0], pts_norm[:, 1], "k-", linewidth=0.8, alpha=0.8)

        ticks = np.arange(0, 1.001, eps)
        for t in ticks:
            ax.axvline(t, color="gray", linestyle=":", linewidth=0.5, alpha=0.3)
            ax.axhline(t, color="gray", linestyle=":", linewidth=0.5, alpha=0.3)

        for bx, by in hit_boxes:
            rect = patches.Rectangle(
                (bx * eps, by * eps),
                eps,
                eps,
                linewidth=0,
                edgecolor="none",
                facecolor="#e74c3c",
                alpha=0.4,
            )
            ax.add_patch(rect)

        ax.set_title(
            f"$\\varepsilon = 1/{int(1/eps)}$\n$N(\\varepsilon) = {len(hit_boxes)}$"
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_local_slopes(counts: pd.DataFrame, title: str, output: Path) -> None:
    """Plot discrete derivative of log-log curve."""
    log_eps = counts["log_epsilon"].to_numpy()
    log_N = counts["log_counts"].to_numpy()

    slopes = -np.diff(log_N) / np.diff(log_eps)
    mid_log_eps = (log_eps[:-1] + log_eps[1:]) / 2

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mid_log_eps, slopes, marker="o", linestyle="", color="black", markersize=4)
    ax.set_xlabel(r"$\ln(\varepsilon)$ (midpoint)")
    ax.set_ylabel(r"Local Dimension Estimate")
    ax.set_title(f"Local Slopes: {title}")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_multi_iteration_loglog(
    all_counts: list[pd.DataFrame], fractal_name: str, output: Path
) -> None:
    """Plot log-log curves for multiple iterations on one figure."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = sns.color_palette("viridis", len(all_counts))

    for i, counts in enumerate(all_counts):
        iteration = i + 1
        result = fit_scaling_relationship(
            counts["log_epsilon"].to_numpy(), counts["log_counts"].to_numpy()
        )
        log_eps_sorted = np.sort(counts["log_epsilon"])
        predicted = result.intercept + result.slope * log_eps_sorted
        ax.plot(
            log_eps_sorted,
            predicted,
            linestyle="-",
            color=colors[i],
            linewidth=1.5,
            label=f"Iteration {iteration}: m = {result.slope:.3g}\nD = -m = {-result.slope:.3g}",
            alpha=0.8,
        )
        ax.plot(
            counts["log_epsilon"],
            counts["log_counts"],
            marker="o",
            linestyle="",
            color=colors[i],
            markersize=3,
            alpha=0.8,
        )

    ax.set_xlabel(r"$\ln(\varepsilon)$")
    ax.set_ylabel(r"$\ln N(\varepsilon)$")
    ax.set_title(f"Convergence of Scaling Law for {fractal_name.title()}")
    ax.legend(frameon=True, loc="lower right", handlelength=1, handletextpad=0.5)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_error_heatmap(
    all_regressions: pd.DataFrame, fractal_name: str, output: Path
) -> None:
    """Plot heatmap of relative error vs iteration and window start."""
    df = all_regressions[all_regressions["window"].str.startswith("slide")].copy()
    df["window_start"] = df["window"].str.extract(r"slide_(\d+)_").astype(int)
    pivot = df.pivot(index="iteration", columns="window_start", values="rel_error")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1%",
        cmap="RdYlGn_r",
        ax=ax,
        cbar_kws={"label": "Relative Error"},
    )

    ax.set_title(f"Relative Error Heatmap: {fractal_name.title()}")
    ax.set_xlabel("Window Start Index (1=Largest Box)")
    ax.set_ylabel("Iteration Depth")

    fig.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(fig)


def run_pipeline(
    output_dir: Path,
    max_iter: int = 6,
    quick_density: bool = False,
    parallel: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating Grid Overlay Visualizations...")
    plot_grid_overlay(
        "koch", generate_koch_curve, output_dir / "viz_grid_overlay_koch.png"
    )
    plot_grid_overlay(
        "sierpinski",
        generate_sierpinski_triangle,
        output_dir / "viz_grid_overlay_sierpinski.png",
    )

    exp = FractalExperiment()
    summary_rows = []

    for spec in DEFAULT_SPECS:
        fractal_counts = []
        fractal_regressions = []

        for iteration in range(1, max_iter + 1):
            logger.info("Processing %s iteration %d...", spec.name, iteration)
            result = exp.run(spec.name, iterations=iteration)
            prefix = f"{spec.name}_n{iteration}"

            counts_path = output_dir / f"{prefix}_counts.csv"
            result.counts.to_csv(counts_path, index=False)
            fractal_counts.append(result.counts)

            regress_path = output_dir / f"{prefix}_regressions.csv"
            result.regressions["iteration"] = iteration
            result.regressions.to_csv(regress_path, index=False)
            fractal_regressions.append(result.regressions)

            logger.info("Running density check for %s n=%d...", spec.name, iteration)
            density_df = exp.run_density_check(
                spec.name,
                iterations=iteration,
                existing_result=result,
                quick=quick_density,
                parallel=parallel,
            )
            density_df.to_csv(output_dir / f"{prefix}_density_check.csv", index=False)

            plot_log_log(
                result.counts,
                f"{spec.name.title()} n={iteration}",
                output_dir / f"{prefix}_loglog.png",
            )
            plot_residuals(result.counts, output_dir / f"{prefix}_residuals.png")
            plot_window_estimates(
                result.regressions, output_dir / f"{prefix}_windows.png"
            )

            plot_local_slopes(
                result.counts,
                f"{spec.name.title()} n={iteration}",
                output_dir / f"{prefix}_local_slopes.png",
            )

            fine_row = result.regressions[result.regressions["window"] == "fine"].iloc[
                0
            ]
            
            # Calculate stability S (std dev of sliding window estimates)
            sliding_windows = result.regressions[
                result.regressions["window"].str.startswith("slide_")
            ]
            stability_s = sliding_windows["d_est"].std()

            summary_rows.append(
                {
                    "fractal": spec.name,
                    "iteration": iteration,
                    "d_est": fine_row.d_est,
                    "abs_error": fine_row.abs_error,
                    "rel_error": fine_row.rel_error,
                    "std_err": fine_row.std_err,
                    "stability_s": stability_s,
                }
            )

        plot_multi_iteration_loglog(
            fractal_counts,
            spec.name,
            output_dir / f"{spec.name}_convergence_loglog.png",
        )

        all_fractal_regs = pd.concat(fractal_regressions)
        plot_error_heatmap(
            all_fractal_regs, spec.name, output_dir / f"{spec.name}_error_heatmap.png"
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
    parser.add_argument(
        "--quick-density",
        action="store_true",
        help="Sample subset of epsilons for density check",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run density check in parallel",
    )
    args = parser.parse_args()
    run_pipeline(
        args.output,
        max_iter=args.max_iter,
        quick_density=args.quick_density,
        parallel=args.parallel,
    )
