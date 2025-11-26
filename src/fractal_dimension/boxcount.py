"""Box-counting utilities anchored at the origin for unit-square data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def box_count(points: np.ndarray, epsilons: np.ndarray) -> pd.DataFrame:
    """Compute box counts for a collection of ``epsilon`` scales.

    Parameters
    ----------
    points:
        Array of shape (N, 2) in ``[0, 1]^2``.
    epsilons:
        One-dimensional array of box widths to evaluate.

    Returns
    -------
    pd.DataFrame
        Columns ``epsilon``, ``log_epsilon``, ``counts`` (``N(ε)``), and
        ``log_counts``.
    """

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    epsilons = np.asarray(epsilons, dtype=float)
    if np.any(epsilons <= 0):
        raise ValueError("epsilons must be positive")

    clipped = np.clip(points, 0.0, 1.0)
    records = []
    for eps in epsilons:
        indices = np.floor(clipped / eps).astype(int)
        unique_boxes = np.unique(indices, axis=0).shape[0]
        records.append(
            {
                "epsilon": eps,
                "log_epsilon": np.log(eps),
                "counts": int(unique_boxes),
                "log_counts": np.log(unique_boxes),
            }
        )
    df = pd.DataFrame.from_records(records).sort_values("epsilon", ascending=False)
    return df.reset_index(drop=True)
