import math

import numpy as np

from fractal_dimension.boxcount import box_count
from fractal_dimension.fractals import (
    DEFAULT_SPECS,
    generate_koch_curve,
)
from fractal_dimension.pipeline import FractalExperiment


def test_box_counts_monotone():
    points = generate_koch_curve(iterations=2, samples_per_segment=40)
    eps = np.array([2 ** (-i) for i in range(1, 8)], dtype=float)
    counts = box_count(points, eps)
    assert counts["counts"].is_monotonic_increasing


def test_dimension_trends_toward_theory():
    exp = FractalExperiment()
    prior_error = math.inf
    for n in range(2, 5):
        result = exp.run("koch", iterations=n, sample_density=60)
        d_est = result.regressions.loc[
            result.regressions["window"] == "fine", "d_est"
        ].iloc[0]
        error = abs(d_est - DEFAULT_SPECS[0].theoretical_dimension)
        assert error < prior_error
        prior_error = error


def test_resampling_stability():
    exp = FractalExperiment()
    low_density = exp.run("sierpinski", iterations=3, sample_density=240)
    high_density = exp.run("sierpinski", iterations=3, sample_density=300)
    low_est = low_density.regressions.loc[
        low_density.regressions["window"] == "fine", "d_est"
    ].iloc[0]
    high_est = high_density.regressions.loc[
        high_density.regressions["window"] == "fine", "d_est"
    ].iloc[0]
    assert abs(low_est - high_est) < 0.05
