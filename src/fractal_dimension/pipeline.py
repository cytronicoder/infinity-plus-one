"""Experiment orchestration for the IA case study."""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .boxcount import box_count
from .fractals import (
    DEFAULT_SPECS,
    FractalSpec,
    get_koch_vertices,
    sample_koch_with_epsilon,
    get_sierpinski_triangles,
    sample_sierpinski_with_epsilon,
)
from .regression import summarize_windows

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **_kwargs):
        """Small local fallback for tqdm when the package isn't available.

        This mirrors tqdm's signature minimally for internal loops during testing
        or CI where the dependency might be omitted.
        """
        return iterable


logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Holds experiment output: box counts and regression summaries.

    Attributes:
        counts: DataFrame of box counts for each epsilon.
        regressions: DataFrame of regression diagnostics for windows.
    """

    counts: pd.DataFrame
    regressions: pd.DataFrame


def _check_single_epsilon(args: tuple[str, int, float, FractalExperiment]) -> dict:
    """Worker function for parallel density check.

    Args:
        args (tuple): Tuple containing name, iterations, epsilon, and experiment instance.

    Returns:
        dict: Dictionary with epsilon, log_epsilon, counts, and duration.
    """
    name, iterations, eps, exp = args
    start_time = time.time()

    # pylint: disable=protected-access
    pts_base = exp._generate_points(name, iterations, eps, double_density=False)
    count_base = box_count(pts_base, np.array([eps])).iloc[0]["counts"]
    n_points_base = len(pts_base)

    pts_double = exp._generate_points(name, iterations, eps, double_density=True)
    count_double = box_count(pts_double, np.array([eps])).iloc[0]["counts"]

    delta = (
        abs(count_double - count_base) / count_double * 100 if count_double > 0 else 0.0
    )

    duration = time.time() - start_time

    return {
        "epsilon": eps,
        "log_epsilon": -math.log(eps),
        "N_base": count_base,
        "N_dense": count_double,
        "delta_percent": delta,
        "points_sampled": n_points_base,
        "duration": duration,
    }


class FractalExperiment:
    """Run box-counting experiments for named fractals and iteration depths.

    Manages the generation of fractal points, box-counting at various scales,
    and regression analysis to estimate fractal dimensions.

    Args:
        eps_powers (Iterable[int]): Powers of 2 for epsilon values.
    """

    def __init__(self, eps_powers: Iterable[int] = range(1, 11)) -> None:
        """Initialize the experiment with epsilon values.

        Args:
            eps_powers (Iterable[int]): Powers of 2 for epsilon values.
        """
        self.epsilons = np.array([2 ** (-i) for i in eps_powers], dtype=float)
        self.specs: Dict[str, FractalSpec] = {spec.name: spec for spec in DEFAULT_SPECS}

    def _generate_points(
        self, name: str, iterations: int, epsilon: float, double_density: bool = False
    ) -> np.ndarray:
        spec = self.specs.get(name)
        if spec is None:
            raise ValueError(f"Unknown fractal: {name}")

        eff_epsilon = epsilon / 2.0 if double_density else epsilon

        if spec.name == "koch":
            vertices = get_koch_vertices(iterations)
            return sample_koch_with_epsilon(vertices, eff_epsilon)

        if spec.name == "sierpinski":
            triangles = get_sierpinski_triangles(iterations)
            return sample_sierpinski_with_epsilon(triangles, eff_epsilon)

        raise ValueError(f"Unknown generator: {spec.generator}")

    def run(self, name: str, iterations: int) -> ExperimentResult:
        """
        Execute the box-counting experiment for a specific fractal and iteration.

        Args:
            name: Name of the fractal (e.g., 'koch').
            iterations: Iteration depth.

        Returns:
            ExperimentResult containing box counts and regression analysis.
        """
        spec = self.specs[name]
        all_counts = []

        for eps in self.epsilons:
            points = self._generate_points(name, iterations, eps)
            df = box_count(points, np.array([eps]))
            all_counts.append(df)

        counts_df = pd.concat(all_counts, ignore_index=True)
        counts_df = counts_df.sort_values("epsilon", ascending=False).reset_index(
            drop=True
        )

        regressions = summarize_windows(
            counts_df["log_epsilon"].to_numpy(),
            counts_df["log_counts"].to_numpy(),
            theoretical_dimension=spec.theoretical_dimension,
        )

        return ExperimentResult(counts=counts_df, regressions=regressions)

    def run_density_check(
        self,
        name: str,
        iterations: int,
        existing_result: ExperimentResult | None = None,
        quick: bool = False,
        parallel: bool = False,
    ) -> pd.DataFrame:
        """Verify sampling density sufficiency by comparing with double density.

        Args:
            name (str): Name of the fractal (e.g., 'koch').
            iterations (int): Iteration depth.
            existing_result (ExperimentResult | None): Optional existing experiment result to reuse cached counts.
            quick (bool): If True, check only a subset of epsilons.
            parallel (bool): If True, run checks in parallel using multiprocessing.

        Returns:
            pd.DataFrame: DataFrame containing density check results (delta percentages).
        """
        start_total = time.time()
        logger.info("Starting density check for %s n=%d...", name, iterations)

        epsilons = self.epsilons
        if quick:
            epsilons = epsilons[::2]
            logger.info("Quick mode enabled: sampling subset of epsilons.")

        results = []

        if parallel:
            logger.info("Running in parallel with %s cores.", os.cpu_count())
            with Pool() as pool:
                args = [(name, iterations, eps, self) for eps in epsilons]
                for res in tqdm(
                    pool.imap(_check_single_epsilon, args),
                    total=len(epsilons),
                    desc="Density Check",
                ):
                    results.append(res)
                    logger.info(
                        "Checked epsilon %.4g: %s pts. Delta: %.2f%%. Time: %.2fs",
                        res["epsilon"],
                        res["points_sampled"],
                        res["delta_percent"],
                        res["duration"],
                    )
        else:
            for eps in tqdm(epsilons, desc="Density Check"):
                start_eps = time.time()

                count_base = None
                n_points_base = 0

                if existing_result is not None:
                    match = existing_result.counts[
                        np.isclose(existing_result.counts["epsilon"], eps)
                    ]
                    if not match.empty:
                        count_base = match.iloc[0]["counts"]

                if count_base is None:
                    pts_base = self._generate_points(
                        name, iterations, eps, double_density=False
                    )
                    count_base = box_count(pts_base, np.array([eps])).iloc[0]["counts"]
                    n_points_base = len(pts_base)

                pts_double = self._generate_points(
                    name, iterations, eps, double_density=True
                )
                count_double = box_count(pts_double, np.array([eps])).iloc[0]["counts"]

                delta = (
                    abs(count_double - count_base) / count_double * 100
                    if count_double > 0
                    else 0.0
                )

                duration = time.time() - start_eps

                logger.info(
                    "Checked epsilon %.4g: %s pts. Delta: %.2f%%. Time: %.2fs",
                    eps,
                    n_points_base if n_points_base > 0 else "cached",
                    delta,
                    duration,
                )

                results.append(
                    {
                        "epsilon": eps,
                        "log_epsilon": -math.log(eps),
                        "N_base": count_base,
                        "N_dense": count_double,
                        "delta_percent": delta,
                    }
                )

        total_duration = time.time() - start_total
        logger.info(
            "Density check completed for %s n=%d in %.2fs",
            name,
            iterations,
            total_duration,
        )

        return pd.DataFrame(results)
