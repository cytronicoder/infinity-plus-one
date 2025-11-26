"""Experiment orchestration for the IA case study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .boxcount import box_count
from .fractals import (
    DEFAULT_SPECS,
    FractalSpec,
    generate_koch_curve,
    generate_sierpinski_triangle,
)
from .regression import summarize_windows


import math


@dataclass
class ExperimentResult:
    counts: pd.DataFrame
    regressions: pd.DataFrame


class FractalExperiment:
    """Run box-counting experiments for named fractals and iteration depths."""

    def __init__(self, eps_powers: Iterable[int] = range(1, 11)) -> None:
        self.epsilons = np.array([2 ** (-i) for i in eps_powers], dtype=float)
        self.specs: Dict[str, FractalSpec] = {spec.name: spec for spec in DEFAULT_SPECS}

    def _generate_points(
        self, name: str, iterations: int, sample_density: int
    ) -> np.ndarray:
        spec = self.specs.get(name)
        if spec is None:
            raise ValueError(f"Unknown fractal: {name}")

        if spec.generator == "generate_koch_curve":
            return generate_koch_curve(
                iterations=iterations, samples_per_segment=sample_density
            )

        if spec.generator == "generate_sierpinski_triangle":
            return generate_sierpinski_triangle(
                iterations=iterations, samples_per_triangle=sample_density
            )

        raise ValueError(f"Unknown generator: {spec.generator}")

    def _get_density(
        self, name: str, iterations: int, eps: float, double_density: bool = False
    ) -> int:
        """Calculate sampling density based on fractal type and epsilon."""
        factor = 2 if double_density else 1

        if name == "koch":
            return math.ceil(10 * factor / eps)

        if name == "sierpinski":
            side_len = 0.5**iterations
            spacing = eps / (5.0 * factor)
            points_per_side = side_len / spacing
            density = math.ceil(points_per_side**2)
            return max(1, density)

        return 120 * factor

    def run(self, name: str, iterations: int) -> ExperimentResult:
        spec = self.specs[name]
        all_counts = []

        for eps in self.epsilons:
            density = self._get_density(name, iterations, eps)
            points = self._generate_points(name, iterations, density)
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

    def run_density_check(self, name: str, iterations: int) -> pd.DataFrame:
        """Verify sampling density sufficiency by comparing with double density."""
        results = []
        for eps in self.epsilons:
            dens_base = self._get_density(name, iterations, eps, double_density=False)
            pts_base = self._generate_points(name, iterations, dens_base)
            count_base = box_count(pts_base, np.array([eps])).iloc[0]["counts"]

            dens_double = self._get_density(name, iterations, eps, double_density=True)
            pts_double = self._generate_points(name, iterations, dens_double)
            count_double = box_count(pts_double, np.array([eps])).iloc[0]["counts"]

            delta = (
                abs(count_double - count_base) / count_double * 100
                if count_double > 0
                else 0.0
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
        return pd.DataFrame(results)
