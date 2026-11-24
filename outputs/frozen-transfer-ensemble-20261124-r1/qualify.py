"""Exact regression cases for branch cuts, nonconvex losses, symmetry, missing angles and tie policy."""
from fractions import Fraction as F

import numpy as np

from calibrate import knots, optimum
from score import columns, predict

def check(rows, expected):
    records = [dict(task="fixture", y=y,t=t,tt=h,tv=True,ttv=valid) for y,t,h,valid in rows]
    tasks,table = knots(records,"tt")
    assert tasks == ["fixture"] and optimum(table,(1,))[0] == expected
    for r in records:
        r.update(tg=r["tt"],tgv=r["ttv"])
    d = columns(records)
    yaws = predict(d,(float(expected),float(expected)))
    for n,r in enumerate(records):
        if not r["ttv"] or abs((r["tt"]-r["t"]+180)%360-180)==180:
            np.testing.assert_array_equal(yaws[n,:], (r["t"]+180)%360-180)

check([(180,170,-160,True)], F(1,3))
check([(0,170,-170,True)], F(0))  # Antipodal truth halfway along arc; both endpoints minimize loss.
check([(5,0,0,True),(8,4,-176,True),(6,3,100,False)], F(1,2))
check([(5,0,20,True),(15,0,20,True)], F(1,2))  # Flat minimum from .25 through .75.
check([(0,0,20,True),(5,0,20,True)], F(1,4))  # Flat minimum to the left of .5.
print("Passed exact wrap, validity, disconnected and flat tie fixtures.")
