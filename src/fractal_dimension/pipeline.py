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


@dataclass
class ExperimentResult:
    points: np.ndarray
    counts: pd.DataFrame
    regressions: pd.DataFrame


class FractalExperiment:
    """Run box-counting experiments for named fractals and iteration depths."""

    def __init__(self, eps_powers: Iterable[int] = range(1, 11)) -> None:
        self.epsilons = np.array([2 ** (-i) for i in eps_powers], dtype=float)
        self.specs: Dict[str, FractalSpec] = {spec.name: spec for spec in DEFAULT_SPECS}

    def _generate_points(
        self, name: str, iterations: int, sample_density: int = 120
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

    def run(
        self, name: str, iterations: int, sample_density: int = 120
    ) -> ExperimentResult:
        points = self._generate_points(name, iterations, sample_density)
        counts = box_count(points, self.epsilons)
        spec = self.specs[name]
        regressions = summarize_windows(
            counts, theoretical_dimension=spec.theoretical_dimension
        )
        return ExperimentResult(points=points, counts=counts, regressions=regressions)
