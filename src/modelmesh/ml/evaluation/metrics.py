"""Router evaluation metrics (ADR-0004): quality-cost hull, AIQ, Oracle, regret, calibration, bootstrap CIs.

Conventions, fixed here so every router is measured the same way:
- A router's operating points are (mean realised cost, mean realised acceptance) over the evaluation items, one
  point per threshold of its sweep.
- Its hull is the non-decreasing upper concave hull of those points **plus the Cheapest-Single point**. Mixing
  any router with "always use the cheapest model" is achievable by randomising per request, which is what the
  convex hull represents (RouterBench). Right of its last vertex the hull is flat.
- AIQ is the hull's mean value over the shared cost range [Cheapest-Single cost, most expensive single model's
  cost] on the same items.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Floats = npt.NDArray[np.float64]
Ints = npt.NDArray[np.int64]


def oracle_choice(acc: Floats, cost: Floats) -> Ints:
    """Per item, the cheapest accepted model; the cheapest model when none is accepted. Ties take the lower index."""
    choices = np.empty(acc.shape[0], dtype=np.int64)
    for i in range(acc.shape[0]):
        accepted = np.flatnonzero(acc[i] >= 0.5)
        pool = accepted if accepted.size else np.arange(acc.shape[1])
        choices[i] = pool[np.argmin(cost[i, pool])]  # argmin returns the first minimum
    return choices


def route_cost_optimized(qhat: Floats, cost_pred: Floats, threshold: float) -> Ints:
    """ADR-0004 cost_optimized: cheapest predicted cost with Q-hat >= threshold, tie-broken on higher Q-hat.

    When no model meets the threshold the highest Q-hat is taken (tie-broken on cost), so a sweep stays defined.
    """
    choices = np.empty(qhat.shape[0], dtype=np.int64)
    for i in range(qhat.shape[0]):
        eligible = np.flatnonzero(qhat[i] >= threshold)
        if eligible.size:
            order = np.lexsort((-qhat[i, eligible], cost_pred[i, eligible]))
            choices[i] = eligible[order[0]]
        else:
            order = np.lexsort((cost_pred[i], -qhat[i]))
            choices[i] = order[0]
    return choices


def picked(matrix: Floats, choices: Ints) -> Floats:
    """matrix[i, choices[..., i]] for a (n,) or (T, n) choice array."""
    rows = np.arange(matrix.shape[0])
    return matrix[rows, choices]


def weighted_mean(values: Floats, weights: Floats | None) -> float:
    if weights is None:
        return float(np.mean(values))
    return float(np.sum(values * weights) / np.sum(weights))


def operating_points(choices: Ints, acc: Floats, cost: Floats, weights: Floats | None = None) -> tuple[Floats, Floats]:
    """Mean cost and mean acceptance per threshold. choices has shape (T, n)."""
    costs = np.array([weighted_mean(picked(cost, row), weights) for row in choices])
    quals = np.array([weighted_mean(picked(acc, row), weights) for row in choices])
    return costs, quals


def upper_hull(costs: Floats, quals: Floats) -> tuple[Floats, Floats]:
    """Non-decreasing upper concave hull of the points, as vertex arrays sorted by cost."""
    order = np.lexsort((-quals, costs))
    frontier: list[tuple[float, float]] = []
    for c, q in zip(costs[order], quals[order], strict=True):
        if frontier and q <= frontier[-1][1]:
            continue  # dominated: costs at least as much for no more quality
        if frontier and c == frontier[-1][0]:
            frontier.pop()
        frontier.append((float(c), float(q)))
    hull: list[tuple[float, float]] = []
    for point in frontier:
        while len(hull) >= 2:
            (ax, ay), (bx, by) = hull[-2], hull[-1]
            cross = (bx - ax) * (point[1] - ay) - (by - ay) * (point[0] - ax)
            if cross >= 0:  # the middle vertex lies on or below the chord, so it is not on the hull
                hull.pop()
            else:
                break
        hull.append(point)
    return np.array([p[0] for p in hull]), np.array([p[1] for p in hull])


def hull_at(hull: tuple[Floats, Floats], cost: float) -> float:
    return float(np.interp(cost, hull[0], hull[1]))


def aiq(hull: tuple[Floats, Floats], c_min: float, c_max: float) -> float:
    """Mean of the hull over [c_min, c_max]. Exact, since the hull is piecewise linear."""
    if c_max <= c_min:
        return hull_at(hull, c_min)
    inner = hull[0][(hull[0] > c_min) & (hull[0] < c_max)]
    xs = np.concatenate(([c_min], inner, [c_max]))
    ys = np.interp(xs, hull[0], hull[1])
    return float(np.sum((xs[1:] - xs[:-1]) * (ys[1:] + ys[:-1]) / 2) / (c_max - c_min))


def min_cost_for_quality(hull: tuple[Floats, Floats], target: float) -> float:
    """Smallest cost at which the hull reaches the target quality; NaN when it never does."""
    xs, ys = hull
    if target > ys[-1] + 1e-12:
        return float("nan")
    if target <= ys[0]:
        return float(xs[0])
    j = int(np.flatnonzero(ys >= target - 1e-12)[0])
    x0, x1, y0, y1 = xs[j - 1], xs[j], ys[j - 1], ys[j]
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))


@dataclass(frozen=True)
class EvalSet:
    """Realised outcomes on evaluation items: acceptance and cost per (item, model)."""

    acc: Floats
    cost: Floats
    cheapest: int  # index of the Cheapest-Single model
    priciest: int  # index of the most expensive single model (upper end of the AIQ range)
    weights: Floats | None = None

    def subset(self, idx: Ints) -> EvalSet:
        weights = None if self.weights is None else self.weights[idx]
        return EvalSet(self.acc[idx], self.cost[idx], self.cheapest, self.priciest, weights)

    def single(self, model: int) -> tuple[float, float]:
        return weighted_mean(self.cost[:, model], self.weights), weighted_mean(self.acc[:, model], self.weights)

    def cost_range(self) -> tuple[float, float]:
        return self.single(self.cheapest)[0], self.single(self.priciest)[0]

    def router_hull(self, choices: Ints) -> tuple[Floats, Floats]:
        costs, quals = operating_points(np.atleast_2d(choices), self.acc, self.cost, self.weights)
        anchor_c, anchor_q = self.single(self.cheapest)
        return upper_hull(np.append(costs, anchor_c), np.append(quals, anchor_q))

    def router_aiq(self, choices: Ints) -> float:
        return aiq(self.router_hull(choices), *self.cost_range())

    def zero_router_hull(self) -> tuple[Floats, Floats]:
        points = [self.single(m) for m in range(self.acc.shape[1])]
        return upper_hull(np.array([p[0] for p in points]), np.array([p[1] for p in points]))


def ece(prob: Floats, label: Floats, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins, weighted by bin size."""
    if prob.size == 0:
        return float("nan")
    index = np.minimum((prob * bins).astype(np.int64), bins - 1)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            total += mask.sum() * abs(float(prob[mask].mean()) - float(label[mask].mean()))
    return total / prob.size


def brier(prob: Floats, label: Floats) -> float:
    return float(np.mean((prob - label) ** 2)) if prob.size else float("nan")


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float

    def as_dict(self) -> dict[str, float]:
        return {"point": self.point, "low": self.low, "high": self.high}


def bootstrap(
    statistic: Callable[[Ints], Floats], n: int, *, resamples: int = 2000, seed: int = 0, level: float = 0.95
) -> list[Interval]:
    """Percentile bootstrap over items. statistic(idx) returns a vector; differences inside it are paired."""
    rng = np.random.default_rng(seed)
    point = np.asarray(statistic(np.arange(n)), dtype=np.float64)
    draws = np.empty((resamples, point.size))
    for r in range(resamples):
        draws[r] = statistic(rng.integers(0, n, n))
    alpha = (1 - level) / 2
    low = np.nanquantile(draws, alpha, axis=0)
    high = np.nanquantile(draws, 1 - alpha, axis=0)
    return [Interval(float(p), float(lo), float(hi)) for p, lo, hi in zip(point, low, high, strict=True)]


@dataclass(frozen=True)
class Outcomes:
    """Realised acceptance and cost of one router's choices, per threshold (rows) and item (columns)."""

    acc: Floats
    cost: Floats

    @classmethod
    def from_choices(cls, choices: Ints, acc: Floats, cost: Floats) -> Outcomes:
        rows = np.atleast_2d(choices)
        return cls(picked(acc, rows), picked(cost, rows))

    def points(self, idx: Ints, weights: Floats | None = None) -> tuple[Floats, Floats]:
        a, c = self.acc[:, idx], self.cost[:, idx]
        if weights is None:
            return c.mean(axis=1), a.mean(axis=1)
        w = weights[idx]
        return (c * w).sum(axis=1) / w.sum(), (a * w).sum(axis=1) / w.sum()

    def hull(self, idx: Ints, anchor: tuple[float, float], weights: Floats | None = None) -> tuple[Floats, Floats]:
        costs, quals = self.points(idx, weights)
        return upper_hull(np.append(costs, anchor[0]), np.append(quals, anchor[1]))
