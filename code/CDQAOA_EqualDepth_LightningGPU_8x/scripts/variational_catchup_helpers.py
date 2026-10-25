"""Small pure helpers for the variational catch-up pilot."""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def _finite_values(values: Iterable[float]) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError("values must not be empty")
    if not all(math.isfinite(value) for value in result):
        raise ValueError("values must all be finite")
    return result


def cumulative_best(values: Iterable[float]) -> list[float]:
    """Return the cumulative maximum of a finite, nonempty sequence."""
    raw = _finite_values(values)
    best = raw[0]
    result = []
    for value in raw:
        best = max(best, value)
        result.append(best)
    return result


def pad_best(values: Iterable[float], length: int) -> list[float]:
    """Pad a best-so-far trajectory by carrying its final value forward."""
    best = cumulative_best(values)
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError("length must be a positive integer")
    if len(best) > length:
        raise ValueError("trajectory is longer than requested length")
    return best + [best[-1]] * (length - len(best))


def mean_trajectory(traces: Sequence[Sequence[float]]) -> list[float]:
    """Average aligned trajectories, requiring equal nonzero lengths."""
    if not traces:
        raise ValueError("traces must not be empty")
    clean = [_finite_values(trace) for trace in traces]
    lengths = {len(trace) for trace in clean}
    if len(lengths) != 1:
        raise ValueError("trajectories must have equal length")
    return [sum(values) / len(clean) for values in zip(*clean)]


def first_crossing(mean: Iterable[float], target: float) -> int | None:
    """Return the one-based first evaluation meeting or exceeding target."""
    values = _finite_values(mean)
    threshold = float(target)
    if not math.isfinite(threshold):
        raise ValueError("target must be finite")
    for evaluation, value in enumerate(values, start=1):
        if value >= threshold:
            return evaluation
    return None
