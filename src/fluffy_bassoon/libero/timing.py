"""Convention-explicit command-motion lag estimation for LIBERO rollouts."""

import numpy as np


def _normalized_cross_correlation(commands, displacements):
    """Return zero-mean vector cosine similarity, or None when degenerate."""
    xc = commands - commands.mean(axis=0)
    yc = displacements - displacements.mean(axis=0)
    denom = float(np.linalg.norm(xc) * np.linalg.norm(yc))
    if denom <= 0.0:
        return None
    return float(np.sum(xc * yc) / denom)


def estimate_command_lag(
    commands,
    positions,
    *,
    positions_timing,
    max_lag=20,
    prefix=80,
    fixed_window=True,
    min_overlap=5,
):
    """Estimate discrete command-to-displacement lag in controls.

    Args:
        commands: Sequence/array of issued commands with shape ``[T, >=3]``.
        positions: Sequence/array of end-effector positions with shape ``[T, >=3]``.
        positions_timing: ``"before"`` when ``positions[t]`` is observed before
            ``commands[t]`` executes (aligning ``dy[t]``
            with ``commands[t - lag]``), or ``"after"`` when ``positions[t]``
            is recorded after ``step(commands[t])`` (live ``state_after``
            convention, aligning ``dy[t]`` with ``commands[t + 1 - lag]``).
        max_lag: Maximum integer lag (inclusive) to evaluate.
        prefix: Optional prefix cap applied to both arrays before analysis.
        fixed_window: When True, compare every candidate lag on the identical
            displacement window ``[max_lag, n - 1)``. When False, evaluate each
            lag on its maximal valid overlap without zero-padding.
        min_overlap: Minimum number of displacement steps required (>= 2).

    Returns:
        Unique argmax integer lag, or ``None`` if overlap is insufficient,
        signals have zero variance, or multiple lags tie for the maximum.
    """
    if positions_timing not in ("before", "after"):
        raise ValueError(f"positions_timing must be 'before' or 'after', got {positions_timing!r}")
    if isinstance(max_lag, bool) or max_lag < 0:
        raise ValueError(f"max_lag must be a non-negative integer, got {max_lag!r}")
    if isinstance(min_overlap, bool) or min_overlap < 2:
        raise ValueError(f"min_overlap must be an integer >= 2, got {min_overlap!r}")
    if prefix is not None and (isinstance(prefix, bool) or prefix < 2):
        raise ValueError(f"prefix must be None or an integer >= 2, got {prefix!r}")

    cmd = np.asarray(commands, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    if cmd.ndim != 2 or cmd.shape[1] < 3 or pos.ndim != 2 or pos.shape[1] < 3:
        raise ValueError("commands and positions must be 2D arrays with at least 3 coordinates")
    if prefix is not None:
        cmd = cmd[:prefix]
        pos = pos[:prefix]
    n = min(len(cmd), len(pos))
    if n < 2:
        return None
    cmd = cmd[:n, :3]
    pos = pos[:n, :3]
    if not (np.isfinite(cmd).all() and np.isfinite(pos).all()):
        raise ValueError("Require finite commands and positions")

    shift = 0 if positions_timing == "before" else 1
    dy = np.diff(pos, axis=0)
    if fixed_window and len(dy) - max_lag < min_overlap:
        return None

    scores = {}
    for lag in range(max_lag + 1):
        if fixed_window:
            lo, hi = max_lag, len(dy)
        else:
            lo = max(0, lag - shift)
            hi = min(len(dy), len(cmd) - shift + lag)
            if hi - lo < min_overlap:
                continue
        score = _normalized_cross_correlation(cmd[lo + shift - lag : hi + shift - lag], dy[lo:hi])
        if score is not None:
            scores[lag] = score

    if not scores:
        return None
    best = max(scores.values())
    winners = [lag for lag, score in scores.items() if score == best]
    return winners[0] if len(winners) == 1 else None
