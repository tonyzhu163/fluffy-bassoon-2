"""Causal Flora success decisions and simulator-only offline qualification.

Terminal grades cannot supply a first-success timestamp. Temporal callers must
score only video prefixes (or chunks ending at the decision), in control units.
This module never passes simulator truth to the decision rule.
"""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class FloraSuccessRule:
    minimum_reward: float = .95
    minimum_p5: float = .95
    consecutive: int = 1
    stride: int = 50

    def __post_init__(self):
        if (not 0 <= self.minimum_reward <= 1 or not 0 <= self.minimum_p5 <= 1
                or type(self.consecutive) is not int or self.consecutive < 1
                or type(self.stride) is not int or self.stride < 1):
            raise ValueError("Finite probability thresholds and positive hold/stride required")

    def record(self):
        return asdict(self)


def first_success(scores, rule):
    """Return confirmation time, never backdate to the start of the hold.

    Every row binds the last visible physical control. Missing observations
    break a hold; future frames, repeated or unordered timestamps are errors.
    consecutive=3,stride=50 means three chunk decisions, not three RGB frames.
    """
    previous, held, first = 0, 0, None
    for row in scores:
        step = row["step"]
        reward, p5 = float(row["reward"]), float(row["p5"])
        if (type(step) is not int or step <= previous or row["observed_through"] != step
                or not math.isfinite(reward) or not math.isfinite(p5)
                or not 0 <= reward <= 1 or not 0 <= p5 <= 1):
            raise ValueError("Require ordered causal prefix scores in physical control units")
        if step - previous != rule.stride:
            held = 0
        held = held + 1 if reward >= rule.minimum_reward and p5 >= rule.minimum_p5 else 0
        if first is None and held >= rule.consecutive:
            first = step
        previous = step
    return first


def timing_metrics(predicted, simulator, *, tolerance=50):
    """Align one confirmation/first-native-success pair per complete episode.

    Early detections outside tolerance count as timing false positives, even
    if that episode eventually succeeds. Report those separately from native
    never-success episodes. No terminal-success proxy is accepted here.
    """
    if len(predicted) != len(simulator) or not predicted or type(tolerance) is not int or tolerance < 0:
        raise ValueError("Require aligned nonempty episodes and nonnegative timing tolerance")
    for value in (*predicted, *simulator):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError("First-success controls must be positive integers or None")
    detections = sum(x is not None for x in predicted)
    positives = sum(x is not None for x in simulator)
    never = sum(p is not None and s is None for p, s in zip(predicted, simulator, strict=True))
    early = sum(p is not None and s is not None and p < s - tolerance
                for p, s in zip(predicted, simulator, strict=True))
    valid = sum(p is not None and s is not None and p >= s - tolerance
                for p, s in zip(predicted, simulator, strict=True))
    delays = sorted(p - s for p, s in zip(predicted, simulator, strict=True)
                    if p is not None and s is not None)
    return dict(episodes=len(predicted), detections=detections, native_successes=positives,
                false_on_never_success=never, prematurely_detected=early,
                first_success_precision=valid / detections if detections else None,
                recall=valid / positives if positives else None,
                timing_error_controls=delays, tolerance_controls=tolerance)
