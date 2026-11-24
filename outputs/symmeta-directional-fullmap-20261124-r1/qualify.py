"""Single-core numerical convention check on one existing training acquisition."""
from pathlib import Path
import sys

import numpy as np
from algebra import Support, scalar
from select_gate import grid_cost
from common import (HERE, READER_PIN, REGISTRATION, SCALE, case_rows, correct, historical, law,
                    metrics, write)


def main(out, bank):
    fit = historical()
    prior, residual = np.array(fit["prior"]), fit["residual_power"]
    rows = case_rows(bank / "Panda-libero_10-1", prior, residual)
    assert len(rows) == 40 and all(r["arm"] == "Panda" and r["split"] == "train" for r in rows)
    direct = np.array([[.7, -.61], [.39, .25]])   # Deliberately nonsymmetric fixture, independent of any fit.
    maxerr = 0.
    for r in rows:
        s = Support(r["x"], r["y"], residual)
        np.testing.assert_array_equal(s.blended(direct, "direct"), direct)
        np.testing.assert_array_equal(s.blended(direct, 0), s.r)
        np.testing.assert_array_equal(scalar(s.r, direct, 0), s.r)
        np.testing.assert_array_equal(scalar(s.r, direct, 1), direct)
        result = s.blended(direct, 3.)
        expected = s.r + np.linalg.solve(s.gram+3*s.k*np.eye(4), (direct-s.r).ravel()*3*s.k).reshape(2, 2)
        maxerr = max(maxerr, float(abs(result-expected).max()))
        np.testing.assert_allclose(result, expected, atol=2e-13, rtol=0)
        np.testing.assert_allclose(s.blended(direct, 3., "trace"), scalar(s.r, direct, s.weights(3.).mean()))
        assert np.all(np.diff(s.weights(3.)) <= 0)
        cmds = r["issued"].reshape(3,5,7).mean(1)[:,:3]
        np.testing.assert_allclose((r["x"]@direct.ravel()).reshape(3,2)*SCALE,
                                   law(cmds, direct, prior)-law(cmds, np.zeros((2,2)), prior), atol=1e-15)
        cost = grid_cost(r, np.stack([s.r, result]), prior)
        for j, op in enumerate((s.r, result)):
            original = metrics(r, op, prior)
            np.testing.assert_allclose(cost[j,0], original["command"], atol=1e-12)
            if original["motion"] is not None:
                np.testing.assert_allclose(cost[j,1], original["motion"], atol=1e-12)
        # Token layout copied from the original 100-channel feature builder.
        np.testing.assert_allclose(r["tok"][:,67:83], np.broadcast_to(s.gram.ravel(), (3,16)),
                                   atol=2e-7, rtol=1e-6)
    x = np.diag([0., 1., 2., 4.])
    s = Support(x, np.ones(4), residual)
    w = s.weights(1.)
    np.testing.assert_allclose(s.blended(direct, 1.).ravel(), s.r.ravel()+w*(direct-s.r).ravel())
    np.testing.assert_allclose(s.blended(direct, 1., "reversed").ravel(), s.r.ravel()+w[::-1]*(direct-s.r).ravel())
    assert w[0] == 1 and s.weights(0)[0] == 0
    write(out/"qualification.json", dict(registration=REGISTRATION, reader_pin=READER_PIN,
          training_fixture=str(bank / "Panda-libero_10-1"), supports=len(rows), solve_max_abs=maxerr,
          result="passed", checks=["C-order", "exact endpoints", "null axis", "reversed weights",
          "trace scalar", "inverse vs eigensolve", "Gram available in original end-of-support tokens"]))


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:]))
