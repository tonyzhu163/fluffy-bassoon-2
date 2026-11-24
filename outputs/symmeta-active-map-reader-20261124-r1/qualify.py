"""Linux numerical fixture on a single completed SOURCE case, before consuming GPU allocation."""
import json
from pathlib import Path
import sys

import numpy as np

from observations import (BASIS, IDENTITY, case_rows, correct, features, metrics, model_rows,
                          ridge, save, train_views)


def main():
    bank,output=map(Path,sys.argv[1:])
    rt=json.loads((bank/"runtime-Panda.json").read_text())
    prior,residual=np.asarray(rt["prior"]),rt["residual"]
    rows=case_rows(bank/"Panda-libero_10-1",prior,residual)
    assert len(rows)==40 and all(r["arm"]=="Panda" and r["split"]=="train" for r in rows)
    # Explicit nonsymmetric, reflecting matrix detects both a transposed map and reversed inverse.
    a=np.array([[.12,.59],[.81,-.06]])
    q=np.array([[.15,-.2],[-.75,.9],[.5,.1]])
    exact=correct(q,a)
    np.testing.assert_allclose(exact,np.clip(np.linalg.solve(a,q.T).T,-1,1),atol=1e-14)
    assert np.any(abs(exact)==1) and not np.allclose(exact,correct(q,a.T))
    rng=np.random.default_rng(24112431)
    count=0
    oracle_max=0.
    for r in rows:
        for tok,x,y,op in train_views(r,prior,residual,rng):
            assert tok.shape==(3,100) and x.shape==(6,4) and y.shape==(6,)
            count+=1
        if np.linalg.matrix_rank(r["x"])==4:
            clean=np.linalg.lstsq(r["x"],r["x"]@r["a"].ravel(),rcond=None)[0].reshape(2,2)
            np.testing.assert_allclose(clean,r["a"],atol=1e-9)
            oracle_max=max(oracle_max,metrics(r,r["ideal"],prior)["command"])
    scores={}
    for label in ("identity","incumbent","true_map"):
        values=[]
        for r in rows:
            op=IDENTITY if label=="identity" else r["a"] if label=="true_map" else ridge(r["x"],r["y"],residual)
            values.append(metrics(r,op,prior))
        scores[label]={k:float(np.mean([v[k] for v in values])) for k in ("command","motion","map")}
    save(output,dict(case=rows[0]["support_id"],support_attempts=len(rows),physical_views_with_relabels=count,
                     full_rank_goal_fit_command_max_mm=oracle_max,numeric_convention="passed",scores=scores))


if __name__=="__main__":
    main()
