"""
Generate Appendix B data for math paper.

Extracts box-counting data for all iterations (n=1 to n=6)
for both Koch curve and Sierpiński triangle.
"""

import pandas as pd

for n in range(1, 7):
    print("=" * 60)
    print(f"Koch Curve (n={n})")
    print("=" * 60)

    koch_data = pd.read_csv(f"results/koch_n{n}_counts.csv")
    koch_data = koch_data.rename(
        columns={
            "epsilon": "eps",
            "counts": "N(eps)",
            "log_epsilon": "ln eps",
            "log_counts": "ln N",
        }
    )

    print(koch_data.to_string(index=False))
    print("\n")

    print("=" * 60)
    print(f"Sierpiński Triangle (n={n})")
    print("=" * 60)

    sierpinski_data = pd.read_csv(f"results/sierpinski_n{n}_counts.csv")
    sierpinski_data = sierpinski_data.rename(
        columns={
            "epsilon": "eps",
            "counts": "N(eps)",
            "log_epsilon": "ln eps",
            "log_counts": "ln N",
        }
    )

    print(sierpinski_data.to_string(index=False))
    print("\n")

    koch_data.to_csv(f"results/koch_n{n}_appendix.csv", index=False)
    sierpinski_data.to_csv(f"results/sierpinski_n{n}_appendix.csv", index=False)
