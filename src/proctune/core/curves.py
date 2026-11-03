r"""Curves as parameters: how strongly one structure responds to a driver (growth, adiposity, age), drawn by hand.

A `Curve` is a few control points $(x_i, y_i)$ on $x \in [0, 1]$ through which a monotone piecewise cubic (PCHIP,
`scipy.interpolate.PchipInterpolator`) is drawn: it passes through every point and never overshoots between them, so
a curve drawn rising stays rising. Outside $[0, 1]$ it holds its end values. The same evaluation is used in Blender
and headless, so a project gives the same model everywhere; Blender's curve widget only edits the points.

`allometric` is the standard form for one structure growing relative to another, $y = a\,x^{b}$: $b = 1$ grows in
proportion, $b > 1$ faster, $b < 1$ slower.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

type F64 = NDArray[np.float64]


@dataclass(frozen=True)
class Curve:
    points: tuple[tuple[float, float], ...] = ((0.0, 0.0), (1.0, 1.0))
    """Control points (x, y), x in [0, 1]"""
    lo: float = 0.0
    hi: float = 1.0
    """The range of y the curve editor allows"""

    def __post_init__(self) -> None:
        xs = [p[0] for p in self.points]
        if len(xs) < 2 or any(b <= a for a, b in itertools.pairwise(xs)):
            raise ValueError(f"curve needs two or more points with increasing x, got {self.points}")

    def __call__(self, x: float) -> float:
        return float(self.evaluate(np.asarray([x], np.float64))[0])

    def evaluate(self, x: F64) -> F64:
        pts = np.asarray(self.points, np.float64)
        xs, ys = pts[:, 0], pts[:, 1]
        inside = np.clip(x, xs[0], xs[-1])
        if len(pts) == 2:
            return np.interp(inside, xs, ys)
        # imported here so `import proctune` works without scipy and can say what is missing (proctune.deps)
        from scipy.interpolate import PchipInterpolator  # noqa: PLC0415

        return np.asarray(PchipInterpolator(xs, ys, extrapolate=False)(inside), np.float64)

    @staticmethod
    def through(*points: tuple[float, float], lo: float = 0.0, hi: float = 1.0) -> Curve:
        return Curve(tuple(points), lo, hi)


def allometric(x: float, b: float, a: float = 1.0) -> float:
    r"""$y = a\,x^{b}$, for $x > 0$."""
    return a * x**b
