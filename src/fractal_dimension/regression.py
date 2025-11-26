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
    d_est: float
    abs_error: float
    rel_error: float


WINDOW_PRESETS = {
    "full": slice(None),
    "coarse": slice(0, 5),
    "fine": slice(2, 7),
}


def fit_scaling_relationship(
    log_eps: np.ndarray,
    log_counts: np.ndarray,
    theoretical_dimension: float | None = None,
) -> RegressionResult:
    """Fit ``log N`` against ``log epsilon`` and return diagnostic metrics."""

    slope, intercept, r_value, _, _ = stats.linregress(log_eps, log_counts)
    predicted = intercept + slope * log_eps
    residuals = log_counts - predicted
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
        d_est=d_est,
        abs_error=abs_error,
        rel_error=rel_error,
    )


def sliding_windows(length: int, window: int = 5) -> List[slice]:
    """Return sliding window slices of a given width."""

    if window <= 0:
        raise ValueError("window must be positive")
    return [
        slice(start, start + window) for start in range(0, max(1, length - window + 1))
    ]


def summarize_windows(
    df: pd.DataFrame, theoretical_dimension: float | None = None
) -> pd.DataFrame:
    """Compute regression metrics across preset and sliding windows."""

    records = []
    for name, sl in WINDOW_PRESETS.items():
        subset = df.iloc[sl]
        result = fit_scaling_relationship(
            subset["log_epsilon"].to_numpy(),
            subset["log_counts"].to_numpy(),
            theoretical_dimension,
        )
        records.append(
            {
                "window": name,
                "start": int(sl.start or 0),
                "end": int(len(df) if sl.stop is None else sl.stop),
                "d_est": result.d_est,
                "r2": result.r2,
                "abs_error": result.abs_error,
                "rel_error": result.rel_error,
            }
        )

    for idx, sl in enumerate(sliding_windows(len(df), window=5)):
        subset = df.iloc[sl]
        result = fit_scaling_relationship(
            subset["log_epsilon"].to_numpy(),
            subset["log_counts"].to_numpy(),
            theoretical_dimension,
        )
        records.append(
            {
                "window": f"slide_{idx}",
                "start": int(sl.start),
                "end": int(sl.stop),
                "d_est": result.d_est,
                "r2": result.r2,
                "abs_error": result.abs_error,
                "rel_error": result.rel_error,
            }
        )
    return pd.DataFrame.from_records(records)
