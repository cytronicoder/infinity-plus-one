# Evaluating Fractal Dimension Estimates Across Scale Ranges

## Introduction
This Internal Assessment studies how the box-counting dimension estimate depends on the choice of scale window. The focus is on deterministic, finite-iteration versions (\(n = 1..6\)) of the Koch curve and Sierpiński triangle, using consistent anchoring at \((0, 0)\) and normalization to \([0, 1]^2\). The guiding question is: **How does the box-size window \([\varepsilon_{\min},\varepsilon_{\max}]\) affect \(D_{\text{est}}\) for these fractals?**

## Theory
For a set \(S \subset \mathbb{R}^2\), the box-counting dimension is defined by
\[
D = \lim_{\varepsilon \to 0} \frac{\ln N(\varepsilon)}{-\ln \varepsilon},
\]
where \(N(\varepsilon)\) counts the number of \(\varepsilon \times \varepsilon\) boxes intersecting \(S\). Benchmark dimensions are \(D_{\text{Koch}} = \ln 4 / \ln 3\) and \(D_{\text{Sier}} = \ln 3 / \ln 2\). Finite approximations exhibit scaling only across a finite window, motivating the window-sensitivity analysis.

## Method
- **Fractal generation:** Iterative construction up to \(n = 6\) with adjustable sampling density (per segment for Koch, per triangle for Sierpiński) to ensure stable box occupancy.
- **Box-counting:** Grids anchored at \((0, 0)\) over \([0, 1]^2\) with scales \(\varepsilon_i = 2^{-i}\) for \(i = 1..10\). Counts and logs are recorded for regression.
- **Regression windows:** Full (all scales), coarse (largest five scales), fine (mid-scale indices \(i = 3..7\) to avoid small-scale saturation), and width-5 sliding windows. Linear regression fits \(\ln N(\varepsilon) = m \ln \varepsilon + b\); the dimension estimate is \(D_{\text{est}} = -m\).
- **Diagnostics:** Report \(R^2\), residuals, absolute error \(E_{\text{abs}} = |D_{\text{est}} - D_{\text{theory}}|\), and relative error \(E_{\text{rel}}\).

## Analysis
Running the pipeline for \(n = 1..6\) shows that fine-window estimates converge toward theoretical dimensions as \(n\) increases. Sliding windows reveal drift at coarser scales where finite construction artifacts dominate. Residual plots remain small and structureless for mid-scale windows, supporting the linear model assumption in those ranges.

## Discussion
The choice of \([\varepsilon_{\min},\varepsilon_{\max}]\) has a measurable impact: coarse windows under-estimate dimension for low \(n\), while fine windows stabilize once the geometry is sufficiently resolved. Increasing sampling density per iteration produces negligible changes in \(D_{\text{est}}\) within a tolerance of \(\pm 0.05\), indicating robustness to point-cloud representation.

## Conclusion
For IA-scale experiments, selecting a fine or mid-scale window (e.g., width-5 sliding windows focused on \(\varepsilon \le 2^{-5}\)) yields accurate, reproducible estimates for both fractals by \(n \ge 4\). The accompanying codebase automates generation, counting, regression, diagnostics, and figure/table export to support transparent, reproducible reporting.
