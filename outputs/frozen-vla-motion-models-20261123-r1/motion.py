"""Few-shot native Cartesian response matching, using row-vector action convention."""
import numpy as np


def fit_source(traces):
    inputs = np.concatenate([t["inputs"] for t in traces])
    outputs = np.concatenate([t["response"] for t in traces])
    coef, _, rank, _ = np.linalg.lstsq(inputs, outputs, rcond=None)
    if rank != 6:
        raise ValueError("Source traces do not span current and preceding actions")
    noise = np.sum((outputs - inputs @ coef) ** 2) / np.sum(inputs[:, :3] ** 2)
    return coef, noise


def ridge(inputs, outputs, strength):
    return np.linalg.solve(inputs.T @ inputs + np.eye(inputs.shape[1]) * strength, inputs.T @ outputs)


def fit_target(method, source, trace, strength):
    x, y = trace["inputs"], trace["response"]
    if method == "source":
        return source.copy()
    if method == "target_ridge":
        return ridge(x, y, strength)
    residual = y - x @ source
    result = source.copy()
    if method == "jacobian":
        result[:3] += ridge(x[:, :3], residual, strength)
    elif method == "planar":
        u, v = x[:, 0], x[:, 1]
        basis = np.stack((np.column_stack((u, -v)), np.column_stack((v, u))), axis=1)
        a, b = ridge(basis.reshape(-1, 2), residual[:, :2].reshape(-1), strength)
        result[:2, :2] += [[a, b], [-b, a]]
    else:
        raise ValueError(method)
    return result


def correction_matrix(source, target, noise):
    # At constant velocity, the preceding and current command have equal value.
    # Weight identity by the source residual power per unit command power.
    ref = source[:3] + source[3:]
    body = target[:3] + target[3:]
    return np.eye(3) + np.linalg.solve(
        body @ body.T + noise * np.eye(3), body @ (ref - body).T
    ).T


def map_action(action, matrix):
    result = np.array(action, dtype=np.float32, copy=True)
    matrix = np.asarray(matrix, dtype=float)
    if result.shape[-1] != 7 or matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Expected finite 3x3 body correction and native seven-component actions")
    result[..., :3] = np.clip(result[..., :3] @ matrix, -1, 1)
    return result
