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

from fractal_dimension.fractals import (
    get_koch_vertices,
    sample_koch_with_epsilon,
    get_sierpinski_triangles,
    sample_sierpinski_with_epsilon,
)
from fractal_dimension.regression import fit_scaling_relationship


def calculate_counts_with_offset(
    fractal_type: str,
    iterations: int,
    epsilons: np.ndarray,
    offset: tuple[float, float],
) -> pd.DataFrame:
    """Compute box counts with a grid offset using adaptive sampling."""
    records = []
    for eps in epsilons:
        # Generate points for this epsilon
        if fractal_type == "koch":
            verts = get_koch_vertices(iterations)
            points = sample_koch_with_epsilon(verts, eps)
        elif fractal_type == "sierpinski":
            tris = get_sierpinski_triangles(iterations)
            points = sample_sierpinski_with_epsilon(tris, eps)
        else:
            raise ValueError(f"Unknown fractal type: {fractal_type}")

        # Shift
        shifted_points = points - np.array(offset)

        # Count
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
            display_table.to_csv(
                f"results/appendix_c_sampling_koch_n{n}.csv", index=False
            )

        except FileNotFoundError:
            print(
                f"Error: results/koch_n{n}_density_check.csv not found. Run analysis first."
            )

        print("\n")

        print("2. Grid Shift Test (Koch)")
        print("-" * 40)

        results = []
        for offset in offsets:
            df = calculate_counts_with_offset("koch", n, epsilons, offset)
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
            results[i]["Diff"] = f"{diff:.4f}"
            # results[i]["Pass"] = "Yes" if diff < 0.01 else "No" # User said ignore criteria

        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        results_df.to_csv(f"results/appendix_c_grid_shift_koch_n{n}.csv", index=False)
        print("\n")

        print("3. Sampling Validation (Sierpiński)")
        print("-" * 40)

        try:
            density_data = pd.read_csv(f"results/sierpinski_n{n}_density_check.csv")
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
            display_table.to_csv(
                f"results/appendix_c_sampling_sierpinski_n{n}.csv", index=False
            )

        except FileNotFoundError:
            print(
                f"Error: results/sierpinski_n{n}_density_check.csv not found. Run analysis first."
            )

        print("\n")

        print("4. Grid Shift Test (Sierpiński)")
        print("-" * 40)

        results = []
        for offset in offsets:
            df = calculate_counts_with_offset("sierpinski", n, epsilons, offset)
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
        grid_df.to_csv(
            f"results/appendix_c_grid_shift_sierpinski_n{n}.csv", index=False
        )

        print("\n")


if __name__ == "__main__":
    main()
