"""One-step bounded command choices, using only end-of-support evidence for inference."""
from itertools import product
from pathlib import Path
import sys

import numpy as np
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "symmeta-active-map-reader-20261124-r1"))
from observations import correct, features, ridge, rms

TEMPS = 2.0 ** np.arange(-4, 5)
SHRINK = (0., .25, .5, 1., 2., 4., 8.)


def box(Q, v):
    """Global minimum over a 2D box for every row v. A singular flat face is valid."""
    Q = (Q + Q.T) / 2
    interior = v @ np.linalg.pinv(Q, hermitian=True)
    candidates = [np.clip(interior, -1, 1)]
    for i, sign in product((0, 1), (-1., 1.)):
        j = 1 - i
        free = v[:, j] - sign * Q[i, j]
        u = np.empty_like(v)
        u[:, i] = sign
        u[:, j] = np.clip(free / Q[j, j], -1, 1) if Q[j, j] > 0 else np.sign(free)
        candidates.append(u)
    u = np.stack(candidates, axis=1)
    cost = ((u @ Q - 2 * v[:, None, :]) * u).sum(axis=2)
    return u[np.arange(len(v)), cost.argmin(axis=1)]


def point(A, q):
    return box(A.T.astype(float) @ A.astype(float), q @ A.astype(float))


def support(issued, fk, clock, law, noise):
    _, x, y = features(issued.astype(float), fk, clock, law, noise)
    return x, y


class Templates:
    def __init__(self, maps, prior):
        self.maps = maps
        self.flat = maps.reshape(-1, 4)
        self.gram = maps.transpose(0, 2, 1) @ maps
        self.positive = np.linalg.det(maps) > 0
        self.prior = prior
        self.logprior = np.log(prior)
        np.testing.assert_allclose(prior.sum(), 1., atol=1e-12, rtol=0)

    def evidence(self, x, y):
        residuals = self.flat @ x.T - y
        return np.square(residuals).sum(axis=1)

    def logweights(self, sse, var):
        logw = self.logprior - .5 * sse / var
        return logw - logsumexp(logw)

    def moments(self, w):
        mean = w @ self.flat
        second = (w @ self.gram.reshape(-1, 4)).reshape(2, 2)
        return mean.reshape(2, 2), second

    def diagnose(self, w, A):
        mean, Q = self.moments(w)
        C = Q - mean.T @ mean
        d = np.linalg.norm(self.flat - A.ravel(), axis=1)
        near = np.isclose(d, d.min(), rtol=0, atol=2e-12)
        prob = float(w[self.positive].sum())
        return dict(ess=float(1 / (w @ w)), top=float(w.max()), hand_positive=prob,
                    true_hand=prob if np.linalg.det(A) > 0 else 1-prob,
                    hand_brier=(float(np.linalg.det(A) > 0)-prob)**2,
                    exact_mass=float(w[d < 2e-12].sum()), nearest_dist=float(d.min()),
                    nearest_mass=float(w[near].sum()), cov_eigen=np.linalg.eigvalsh(C).tolist(),
                    cov_trace=float(np.trace(C)), mean=mean.tolist())


def actions(q, mean, Q, lam, R, D=None):
    gram = mean.T @ mean
    scalar = max(0., np.trace(Q - gram) / 2)
    v = q @ mean
    arms = dict(risk=box(Q, v), mean_box=box(gram, v), mean_clip=correct(q, mean),
                isotropic=box(gram + lam * scalar * np.eye(2), v),
                identity=q.copy(), zero=np.zeros_like(q), R_box=point(R, q), R_clip=correct(q, R))
    for seed, estimate in (D or {}).items():
        arms[f"D{seed}_clip"] = correct(q, estimate)
        arms[f"D{seed}_box"] = point(estimate, q)
    return arms


def outcomes(u, q, A, oracle, clipped, optimum):
    world = u @ A.T
    norms = np.linalg.norm(q, axis=1)
    unit = np.divide(q, norms[:, None], out=np.zeros_like(q), where=norms[:, None] > 0)
    along = (world * unit).sum(axis=1)
    err = 50 * rms(world - q)
    assert err >= optimum - 1e-8 and np.abs(u).max() <= 1+1e-12
    return dict(setpoint=err, oracle_gap=err-optimum, issued_to_box=50*rms(u-oracle),
                issued_to_clip=50*rms(u-clipped), progress=50*float(along.mean()),
                desired=50*float(norms.mean()), shortfall=50*float((norms-along).mean()),
                orthogonal=50*rms(world-along[:, None]*unit), world_norm=50*rms(world),
                effort=rms(u), saturation=float((abs(u) >= 1-1e-12).any(axis=1).mean()),
                reverse_share=float((along < 0).mean()))


def qualify():
    from scipy.optimize import lsq_linear
    rng = np.random.default_rng(24112449)
    q = np.r_[rng.uniform(-1, 1, (12, 2)), [[.8, -.9], [0., 0.], [1., 1.]]]
    a = np.array([[.36, .63], [.11, -.03]])
    b = np.array([[-.42, .54], [.39, .47]])
    r = np.array([[.3, -.6], [-.2, .4]])
    maximum = 0.
    corners = 0
    mismatch = 0.
    refs = []
    for name, maps in dict(zero=[0*a], rank_one=[r], reflected=[a], positive=[-b.T @ b],
                           identical=[a, a], signed_pair=[a, b], cancelling=[a, -a],
                           common_null=[r, r/2], anisotropic=[a, b, r]).items():
        maps = np.asarray(maps)
        w = np.arange(1, 1+len(maps), dtype=float)
        w /= w.sum()
        M, Q = Templates(maps, w).moments(w)
        u = box(Q, q @ M)
        design = (np.sqrt(w)[:, None, None]*maps).reshape(-1, 2)
        costs = []
        for proposal, cmd in zip(q, u, strict=True):
            y = (np.sqrt(w)[:, None]*proposal).ravel()
            ref = lsq_linear(design, y, bounds=(-1, 1), method="bvls", tol=1e-13)
            assert ref.success
            cost = np.sum(w * np.sum((maps @ cmd - proposal)**2, axis=1))
            costs.append(abs(cost - 2*ref.cost))
        if np.all(maps == maps[0]):
            np.testing.assert_allclose(u @ M.T, point(M, q) @ M.T, atol=1e-10, rtol=0)
        maximum = max(maximum, max(costs))
        corners += int(np.all(abs(u)==1., axis=1).sum())
        mismatch = max(mismatch, float(abs(point(M, q)-correct(q, M)).max()))
        refs.append(dict(case=name, max_objective_error=max(costs)))
    assert maximum < 2e-10 and corners > 0 and mismatch > 0.
    return dict(reference="scipy BVLS on stacked sqrt(weight)*A and repeated q; explicit mixture costs",
                test_cases=refs, queries_per_case=len(q), corners=corners, worst_error=maximum)
