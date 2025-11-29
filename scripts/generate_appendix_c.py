"""
Generate Appendix C data for math paper.

Includes:
1. Sampling validation table (original vs double density) for each iteration n=1 to n=6.
2. Grid shift test with multiple offsets for each iteration n=1 to n=6.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent / "src"))

from fractal_dimension.fractals import generate_koch_curve, generate_sierpinski_triangle
from fractal_dimension.regression import fit_scaling_relationship


def count_boxes_offset(
    points: np.ndarray, epsilons: np.ndarray, offset: tuple[float, float]
) -> pd.DataFrame:
    """Compute box counts with a grid offset."""

    shifted_points = points - np.array(offset)

    records = []
    for eps in epsilons:
        indices = np.floor(shifted_points / eps).astype(int)
        unique_boxes = np.unique(indices, axis=0).shape[0]
        records.append(
            {
                "epsilon": eps,
                "log_epsilon": np.log(eps),
                "counts": int(unique_boxes),
                "log_counts": np.log(unique_boxes),
            }
        )
    return pd.DataFrame.from_records(records)


def main():
    epsilons = 2.0 ** -np.arange(1, 11)
    offsets = [(0.0, 0.0), (0.001, 0.001), (0.01, 0.01), (0.1, 0.1)]

    # Process each iteration
    for n in range(1, 7):
        print("=" * 80)
        print(f"Iteration n={n}")
        print("=" * 80)

        print("1. Sampling Validation (Koch)")
        print("-" * 40)

        try:
            density_data = pd.read_csv(f"results/koch_n{n}_density_check.csv")
            subset = density_data.head(5).copy()

            subset["delta_N"] = subset["N_dense"] - subset["N_base"]

            display_table = subset[["epsilon", "N_base", "N_dense", "delta_N"]].rename(
                columns={
                    "epsilon": "eps_i",
                    "N_base": "N_i",
                    "N_dense": "N_i*",
                    "delta_N": "ΔN_i",
                }
            )

            print(display_table.to_string(index=False))
            display_table.to_csv(f"results/appendix_c_sampling_n{n}.csv", index=False)

        except FileNotFoundError:
            print(
                f"Error: results/koch_n{n}_density_check.csv not found. Run analysis first."
            )

        print("\n")

        print("2. Grid Shift Test (Koch)")
        print("-" * 40)

        points = generate_koch_curve(iterations=n)

        results = []
        for offset in offsets:
            df = count_boxes_offset(points, epsilons, offset)
            res = fit_scaling_relationship(
                df["log_epsilon"].to_numpy(), df["log_counts"].to_numpy()
            )
            d_est = -res.slope
            results.append(
                {"Grid Anchor": f"({offset[0]}, {offset[1]})", "D_est": f"{d_est:.4f}"}
            )

        d_base = float(results[0]["D_est"])
        for i in range(1, len(offsets)):
            d_current = float(results[i]["D_est"])
            diff = abs(d_base - d_current)
            results.append(
                {
                    "Grid Anchor": f"Difference from (0,0) for ({offsets[i][0]}, {offsets[i][1]})",
                    "D_est": f"{diff:.4f}",
                }
            )

        grid_df = pd.DataFrame(results)
        print(grid_df.to_string(index=False))
        grid_df.to_csv(f"results/appendix_c_grid_shift_n{n}.csv", index=False)

        print("\n")


if __name__ == "__main__":
    main()
