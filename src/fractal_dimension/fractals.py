"""Fractal generators for Koch curve and Sierpiński triangle."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


def _sample_polyline(vertices: np.ndarray, samples_per_segment: int) -> np.ndarray:
    """Return evenly spaced points along a polyline defined by ``vertices``.

    The sampler includes the starting vertex of each segment but drops interior
    duplicates to preserve deterministic ordering and avoid overlapping points.
    """

    points: List[np.ndarray] = []
    for start, end in zip(vertices[:-1], vertices[1:]):
        segment = end - start
        for i in range(samples_per_segment):
            t = i / samples_per_segment
            points.append(start + t * segment)
    points.append(vertices[-1])
    return np.vstack(points)


def _koch_iteration(vertices: np.ndarray) -> np.ndarray:
    """Apply a single Koch iteration to a list of vertices."""

    new_vertices: List[np.ndarray] = []
    rotation = np.array(
        [
            [math.cos(math.pi / 3), -math.sin(math.pi / 3)],
            [math.sin(math.pi / 3), math.cos(math.pi / 3)],
        ]
    )
    for start, end in zip(vertices[:-1], vertices[1:]):
        segment = end - start
        a = start + segment / 3
        c = start + 2 * segment / 3
        b = a + rotation @ (segment / 3)
        new_vertices.extend([start, a, b, c])
    new_vertices.append(vertices[-1])
    return np.vstack(new_vertices)


def generate_koch_curve(
    *, iterations: int = 4, samples_per_segment: int = 120
) -> np.ndarray:
    """Generate a normalized polyline approximation of the Koch curve.

    Args:
        iterations (int): Number of Koch iterations to apply (1 <= n <= 6).
        samples_per_segment (int): Interpolation points per segment for stable box counts.

    Returns:
        np.ndarray: Array of shape (N, 2) of points in [0, 1]^2 anchored at the origin.
    """

    if not 1 <= iterations <= 6:
        raise ValueError("iterations must be between 1 and 6")
    vertices = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    for _ in range(iterations):
        vertices = _koch_iteration(vertices)
    return _sample_polyline(vertices, samples_per_segment)


def _subdivide_triangle(triangle: np.ndarray) -> List[np.ndarray]:
    """Return the three corner triangles after removing the central hole.

    Args:
        triangle (np.ndarray): Array of shape (3, 2) representing the vertices of the triangle.

    Returns:
        List[np.ndarray]: List of three arrays, each of shape (3, 2), representing the sub-triangles.
    """

    a, b, c = triangle
    ab = (a + b) / 2
    ac = (a + c) / 2
    bc = (b + c) / 2
    return [
        np.vstack([a, ab, ac]),
        np.vstack([ab, b, bc]),
        np.vstack([ac, bc, c]),
    ]


def _triangle_grid(triangle: np.ndarray, samples: int) -> np.ndarray:
    """Create deterministic barycentric grid samples inside a triangle.

    Args:
        triangle (np.ndarray): Array of shape (3, 2) representing the vertices of the triangle.
        samples (int): Approximate number of points to generate.

    Returns:
        np.ndarray: Array of shape (N, 2) containing the sampled points.
    """

    base = int(math.sqrt(samples)) or 1
    us = np.linspace(0, 1, base, endpoint=False)
    vs = np.linspace(0, 1, base, endpoint=False)
    points: List[np.ndarray] = []
    a, b, c = triangle
    for u in us:
        for v in vs:
            if u + v < 1:
                w = 1 - u - v
                points.append(u * a + v * b + w * c)
    points.extend([a, b, c])
    return np.vstack(points)


def generate_sierpinski_triangle(
    *, iterations: int = 4, samples_per_triangle: int = 120
) -> np.ndarray:
    """Generate a point cloud approximation of the Sierpiński triangle.

    Args:
        iterations (int): Number of recursive subdivisions to apply (1 <= n <= 6).
        samples_per_triangle (int): Points to sample per smallest triangle.

    Returns:
        np.ndarray: Array of shape (N, 2) of points in [0, 1]^2.
    """

    if not 1 <= iterations <= 6:
        raise ValueError("iterations must be between 1 and 6")
    base_height = math.sqrt(3) / 2
    triangles: List[np.ndarray] = [
        np.array([[0.0, 0.0], [1.0, 0.0], [0.5, base_height]], dtype=float)
    ]
    for _ in range(iterations):
        refined: List[np.ndarray] = []
        for tri in triangles:
            refined.extend(_subdivide_triangle(tri))
        triangles = refined

    samples = [_triangle_grid(tri, samples_per_triangle) for tri in triangles]
    return np.vstack(samples)


@dataclass(frozen=True)
class FractalSpec:
    """Specification for a fractal instance used in experiments."""

    name: str
    generator: str
    theoretical_dimension: float


DEFAULT_SPECS: Sequence[FractalSpec] = (
    FractalSpec("koch", "generate_koch_curve", math.log(4) / math.log(3)),
    FractalSpec(
        "sierpinski", "generate_sierpinski_triangle", math.log(3) / math.log(2)
    ),
)
