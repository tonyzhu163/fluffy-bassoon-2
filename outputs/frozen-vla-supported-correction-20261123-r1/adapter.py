"""Row-vector translation calibration and native-box application."""
import numpy as np


def spectrum(prior, trace):
    x = trace["inputs"]
    u, s, vt = np.linalg.svd(x[:, :3], full_matrices=False)
    if not s[0] > 0:
        raise ValueError("Passive support has no translation excitation")
    return u, s, vt, trace["response"] - x @ prior


def spectral_fit(prior, spec, strength, component=None):
    u, s, vt, residual = spec
    factors = s / (s**2 + strength)
    if component is not None:
        factors = factors * (np.arange(3) == component)
    target = prior.copy()
    target[:3] += (vt.T * factors) @ (u.T @ residual)
    return target, float(s @ factors)


def matched_strength(s, df):
    """Choose isotropic ridge giving the specified hat-matrix trace per output."""
    if df == 0:
        return np.inf
    powers = s**2
    low, high = 0.0, powers.sum() / df
    for _ in range(80):
        mid = (low + high) / 2
        if np.sum(powers / (powers + mid)) > df:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def line_inverse(prior, target, noise, v):
    """Optimize a scalar command gain along v, preserving both null-space actions and components."""
    ref = v @ (prior[:3] + prior[3:])
    body = v @ (target[:3] + target[3:])
    gain_change = (ref - body) @ body / (body @ body + noise)
    return np.eye(3) + gain_change * np.outer(v, v)


def issue(action, matrix, direction=None):
    """Native row-vector map. A line adapter scales the whole increment at the first box face."""
    a = np.asarray(action)
    if a.shape[-1] != 7 or not np.isfinite(a).all() or np.any(np.abs(a) > 1):
        raise ValueError("Expected bounded seven-component native actions")
    x = np.asarray(a[..., :3], dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Invalid translation matrix")
    if direction is None:
        y = np.clip(x @ matrix, -1, 1)
    else:
        v = np.asarray(direction, dtype=float)
        delta = ((x @ v) * (v @ (matrix - np.eye(3)) @ v))[..., None] * v
        room = np.where(delta >= 0, 1 - x, 1 + x)
        feasible = np.divide(room, abs(delta), out=np.full_like(delta, np.inf), where=delta != 0)
        y = x + np.minimum(1, feasible.min(axis=-1))[..., None] * delta
    out = np.array(a, copy=True)
    out[..., :3] = y
    if not np.isfinite(out).all() or np.any(abs(out) > 1):
        raise ValueError("Native command escaped its bounds")
    return out
