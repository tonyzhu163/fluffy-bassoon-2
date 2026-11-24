"""Map readouts. Only contemporaneous support factors and frozen predictions enter."""
import numpy as np

from common import SCALE, ridge

MULTIPLIERS = [0.] + [10.**(j/2) for j in range(-6, 7)] + ["direct"]
ALPHAS = np.arange(1001) / 1000


class Support:
    def __init__(self, x, y, residual):
        self.r = ridge(x, y, residual)
        self.k = residual / SCALE**2
        self.gram = x.T @ x
        g, self.v = np.linalg.eigh(self.gram)
        # A Gram matrix is positive semidefinite. Remove only signed roundoff at zero.
        if g[0] < -16*np.finfo(float).eps*max(1., g[-1]):
            raise ValueError("Indefinite design Gram")
        self.g = np.maximum(g, 0.)

    def weights(self, multiplier):
        if multiplier == "direct":
            return np.ones(4)
        if multiplier == 0:
            return np.zeros(4)
        lam = multiplier*self.k
        return lam/(lam+self.g)

    def blended(self, direct, multiplier, variant="directional"):
        if multiplier == "direct":
            return direct.copy()
        direct = np.asarray(direct, dtype=float)
        if multiplier == 0:
            return self.r.copy()
        weights = self.weights(multiplier)
        if variant == "trace":
            return scalar(self.r, direct, weights.mean())
        if variant == "reversed":
            weights = weights[::-1]
        elif variant != "directional":
            raise ValueError(variant)
        delta = (direct-self.r).reshape(4)
        return self.r + (self.v @ (weights*(self.v.T @ delta))).reshape(2, 2)


def scalar(r, d, alpha):
    if alpha == 0:
        return r.copy()
    if alpha == 1:
        return d.copy()
    return r + alpha*(d-r)
