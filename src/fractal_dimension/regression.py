"""Regression helpers for estimating box-counting dimensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class RegressionResult:
    slope: float
    intercept: float
    r2: float
    residuals: np.ndarray
    rmse: float
    max_residual: float
    d_est: float
    abs_error: float
    rel_error: float
    std_err: float


WINDOW_PRESETS = {
    "full": slice(None),
    "coarse": slice(0, 5),
    "fine": slice(5, 10),
}


def fit_scaling_relationship(
    log_eps: np.ndarray,
    log_counts: np.ndarray,
    theoretical_dimension: float | None = None,
) -> RegressionResult:
    """Fit ``log N`` against ``log epsilon`` and return diagnostic metrics."""

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_eps, log_counts)
    predicted = intercept + slope * log_eps
    residuals = log_counts - predicted
    rmse = np.sqrt(np.mean(residuals**2))
    max_residual = np.max(np.abs(residuals))
    d_est = -slope
    abs_error = (
        float("nan")
        if theoretical_dimension is None
        else abs(d_est - theoretical_dimension)
    )
    rel_error = (
        float("nan")
        if theoretical_dimension is None or theoretical_dimension == 0
        else abs_error / theoretical_dimension
    )
    return RegressionResult(
        slope=slope,
        intercept=intercept,
        r2=r_value**2,
        residuals=residuals,
        rmse=rmse,
        max_residual=max_residual,
        d_est=d_est,
        abs_error=abs_error,
        rel_error=rel_error,
        std_err=std_err,
    )


def summarize_windows(
    log_eps: np.ndarray,
    log_counts: np.ndarray,
    theoretical_dimension: float | None = None,
) -> pd.DataFrame:
    """Run regressions on all preset and sliding windows."""
    results = []

    for name, sl in WINDOW_PRESETS.items():
        if sl.stop is not None and sl.stop > len(log_eps):
            continue

        subset_eps = log_eps[sl]
        subset_counts = log_counts[sl]

        if len(subset_eps) < 2:
            continue

        res = fit_scaling_relationship(subset_eps, subset_counts, theoretical_dimension)
        results.append(
            {
                "window": name,
                "slope": res.slope,
                "intercept": res.intercept,
                "r2": res.r2,
                "rmse": res.rmse,
                "max_residual": res.max_residual,
                "d_est": res.d_est,
                "abs_error": res.abs_error,
                "rel_error": res.rel_error,
                "std_err": res.std_err,
            }
        )

    window_width = 5
    n_points = len(log_eps)

    for i in range(n_points - window_width + 1):
        sl = slice(i, i + window_width)
        subset_eps = log_eps[sl]
        subset_counts = log_counts[sl]

        res = fit_scaling_relationship(subset_eps, subset_counts, theoretical_dimension)
        window_name = f"slide_{i+1}_to_{i+window_width}"

        results.append(
            {
                "window": window_name,
                "slope": res.slope,
                "intercept": res.intercept,
                "r2": res.r2,
                "rmse": res.rmse,
                "max_residual": res.max_residual,
                "d_est": res.d_est,
                "abs_error": res.abs_error,
                "rel_error": res.rel_error,
                "std_err": res.std_err,
            }
        )

    return pd.DataFrame(results)
