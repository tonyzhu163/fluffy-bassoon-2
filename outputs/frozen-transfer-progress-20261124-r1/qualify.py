"""Numerical edge cases + scripted, non-model proposals for native Linux CPU qualification."""
from types import SimpleNamespace

import numpy as np

from contract import IDENTITY, require
from intervention import Phase, Policy, alignment, estimated_source, issued
from readback import native_commands


class Fixture:
    _model = SimpleNamespace(action_horizon=13, action_dim=32)

    def __init__(self):
        self.calls = 0

    def infer(self, item, *, noise):
        require(item["observation/image"].shape == item["observation/wrist_image"].shape == (224, 224, 3)
                and item["observation/state"].shape == (8,), "native OpenPI observation")
        t = np.arange(13, dtype=np.float32)[:, None]
        base = np.array([.24, -.15, .09, .03, -.04, .06, -.3], np.float32)
        slope = np.array([-.05, .023, -.02, .01, .01, -.02, .04], np.float32)
        original = base + slope*t + np.float32(.03*self.calls) + np.float32(.003)*noise[:, :7]
        self.calls += 1
        return {"actions": original}


def numerical():
    eye = np.vstack((np.eye(3), np.zeros((3, 3))))
    donor = np.zeros((13, 7), np.float32)
    donor[:, 0] = 1.
    carry = np.zeros((5, 7), np.float32)
    start = np.array([40., -72., 83.])
    for j in range(6):
        expected = np.array([j/5., 0., 0.])
        index, _ = alignment(donor, start, start+expected, carry, IDENTITY, eye)
        require(index == j, "failed translated-origin phase index")
    require(alignment(0*donor, start, start, carry, IDENTITY, eye)[0] == 5, "stationary tie")
    repeated = donor.copy()
    repeated[2:, 0] = 0
    require(alignment(repeated, start, start+[.4,0,0], carry, IDENTITY, eye)[0] == 5, "repeated waypoint tie")
    # Compare gain, yaw direction, XYZ and final clip against native inverse and an analytic quarter turn.
    c = dict(yaw_deg=90., xy_scale=.5, gripper_sign=-1)
    proposal = np.tile([1.7, -.75, 1.2, -.6, .4, -.3, .7], (13, 1)).astype("f4")
    np.testing.assert_array_equal(issued(proposal, c), native_commands(proposal, c))
    np.testing.assert_allclose(estimated_source(issued(proposal, c), c)[0], [.5,-.5,.5], atol=1e-7)
    for mode in ("fixed", "progress"):
        phase = Phase(mode, c, eye)
        first, j, _ = phase.propose(proposal)
        require(j is None and np.array_equal(first, proposal), "opening must retain unclipped raw proposal")
        phase.observe(start, start, issued(first[:5], c))
        nxt = -proposal
        second, j, _ = phase.propose(nxt)
        require(j == (5 if mode == "fixed" else 0), "unexpected donor boundary")
        require(np.all(second[:5,:3] == 0) and np.array_equal(second[:,3:], nxt[:,3:])
                and np.array_equal(second[5:], nxt[5:]), "clipped mean/wrist/tail")
        phase.observe(start, start, issued(second[:2], c))
        try:
            phase.propose(nxt)
        except ValueError:
            pass
        else:
            raise ValueError("unexpected replan after partial execution")
    for h in (9, 10, 13):
        stub = SimpleNamespace(_model=SimpleNamespace(action_horizon=h, action_dim=32))
        try:
            Policy(stub, Phase("raw", IDENTITY, eye))
        except ValueError:
            require(h == 9, "valid horizon rejected")
        else:
            require(h >= 10, "truncated donor horizon accepted")
    # All three tail variants must ignore future controls when inferring j.
    altered = donor.copy()
    altered[5:] *= -1.
    require(alignment(donor, start, start, carry, c, eye) == alignment(altered, start, start, carry, c, eye),
            "future predicted suffix entered detector")
