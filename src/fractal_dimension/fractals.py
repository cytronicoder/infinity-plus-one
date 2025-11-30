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


def get_koch_vertices(iterations: int) -> np.ndarray:
    """Generate the vertices of the Koch curve at a given iteration.

    Args:
        iterations (int): Number of Koch iterations (1 <= n <= 6).

    Returns:
        np.ndarray: Array of shape (N, 2) representing the vertices.
    """
    if not 1 <= iterations <= 6:
        raise ValueError("iterations must be between 1 and 6")
    vertices = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    for _ in range(iterations):
        vertices = _koch_iteration(vertices)
    return vertices


def sample_koch_with_epsilon(vertices: np.ndarray, epsilon: float) -> np.ndarray:
    """Sample points along the Koch curve segments with spacing epsilon/4.

    Args:
        vertices (np.ndarray): Vertices of the Koch curve.
        epsilon (float): Box size epsilon.

    Returns:
        np.ndarray: Array of sampled points.
    """
    # Calculate segment length (all segments are equal length)
    p1 = vertices[0]
    p2 = vertices[1]
    seg_len = np.linalg.norm(p2 - p1)

    spacing = epsilon / 4.0
    # p = ceil(len / spacing) + 1 points per segment (including endpoints)
    # _sample_polyline generates N points per segment + 1 at end.
    # So we want samples_per_segment = ceil(len / spacing).
    samples_per_segment = math.ceil(seg_len / spacing)

    return _sample_polyline(vertices, samples_per_segment)


def get_sierpinski_triangles(iterations: int) -> List[np.ndarray]:
    """Generate the list of filled triangles for the Sierpiński triangle.

    Args:
        iterations (int): Number of recursive subdivisions (1 <= n <= 6).

    Returns:
        List[np.ndarray]: List of triangles, each shape (3, 2).
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
    return triangles


def _is_point_in_triangle_vectorized(pts: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Check if points are inside a triangle using barycentric coordinates (vectorized)."""
    a, b, c = tri
    v0 = c - a
    v1 = b - a
    v2 = pts - a

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    # v0 is (2,), v2 is (N, 2). We want dot product for each point.
    dot02 = np.sum(v0 * v2, axis=1)
    dot11 = np.dot(v1, v1)
    dot12 = np.sum(v1 * v2, axis=1)

    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    return (u >= 0) & (v >= 0) & (u + v <= 1)


def sample_sierpinski_with_epsilon(
    triangles: List[np.ndarray], epsilon: float
) -> np.ndarray:
    """Sample points using a Cartesian grid with spacing epsilon/3.

    Args:
        triangles (List[np.ndarray]): List of filled triangles.
        epsilon (float): Box size epsilon.

    Returns:
        np.ndarray: Array of sampled points inside the triangles.
    """
    spacing = epsilon / 3.0
    points_list = []

    # Optimization: Iterate triangles and find grid points in their bounding box
    for tri in triangles:
        min_x, max_x = np.min(tri[:, 0]), np.max(tri[:, 0])
        min_y, max_y = np.min(tri[:, 1]), np.max(tri[:, 1])

        # Grid indices
        i_start = math.floor(min_x / spacing)
        i_end = math.ceil(max_x / spacing)
        j_start = math.floor(min_y / spacing)
        j_end = math.ceil(max_y / spacing)

        # Vectorized grid generation
        i = np.arange(i_start, i_end + 1)
        j = np.arange(j_start, j_end + 1)
        if len(i) == 0 or len(j) == 0:
            continue

        xx, yy = np.meshgrid(i * spacing, j * spacing)
        pts = np.column_stack((xx.ravel(), yy.ravel()))

        if len(pts) > 0:
            mask = _is_point_in_triangle_vectorized(pts, tri)
            points_list.append(pts[mask])

    if not points_list:
        return np.zeros((0, 2))

    # Remove duplicates (triangles touch at vertices)
    all_points = np.vstack(points_list)
    return np.unique(all_points, axis=0)


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
