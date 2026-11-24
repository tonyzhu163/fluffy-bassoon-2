"""Project the production inference outputs into the follow-up's score-free release schema."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from observations import correct, digest, save


def ref(path):
    return dict(path=str(path),sha256=digest(path))


def main():
    run,bank,reader_code,destination=map(Path,sys.argv[1:])
    seal=json.loads((run/"sealed.json").read_text())
    source=ref(run/"source-test.json")
    ck=ref(run/"seed-41.pkl")
    assert source["sha256"]==seal["test"] and ck["sha256"]==seal["checkpoints"]["seed-41.pkl"]
    assert str(bank)==seal["bank"] and not destination.exists()
    law=ref(reader_code/"outputs/frozen-vla-motion-models-20261123-r1/evidence/source-fit.json")
    assert law["sha256"]==json.loads((bank/"runtime-Panda.json").read_text())["source_hash"]
    raw=json.loads((run/"fresh-rows.json").read_text())
    source_test=json.loads((run/"source-test.json").read_text())
    rows={}
    for r in raw:
        method=r["method"]
        if r["family"] not in ("general+","general-") or (method,r["seed"]) not in (("direct",41),("incumbent",0)):
            continue
        suite,ti=r["task"].split("/")
        arm,fam,mode=r["arm"],r["family"],r["mode"]
        dest=rows.setdefault((arm,suite,int(ti),fam,mode),{})
        dest["direct" if method=="direct" else "structured"]=r["estimate"]
    records=[]
    for (arm,suite,task,fam,mode),est in sorted(rows.items()):
        assert set(est)=={"direct","structured"}
        folder=bank/f"{arm}-{suite}-{task}"
        tape=folder/f"{fam}-{mode}.npz"
        case=ref(folder/"case.json")
        meta=json.loads((folder/"case.json").read_text())
        archive=ref(Path(meta["source_record"]["path"]))
        assert archive["sha256"]==meta["source_record"]["sha256"]
        with np.load(tape) as z:
            u,fk=z["issued"].astype("<f8"),z["fk"].astype("<f8")
        assert u.shape==(15,7) and fk.shape==(16,3)
        h=hashlib.sha256(u.tobytes(order="C")+fk.tobytes(order="C")).hexdigest()
        records.append(dict(arm=arm,suite=suite,task=task,family=fam,schedule=mode,seed=41,
             case=case,source_archive=archive,support_tape=ref(tape),source_law_sha256=law["sha256"],
             checkpoint_sha256=ck["sha256"],input_sha256=h,direct_dtype="float32",structured_dtype="float64",**est))
    assert len(records)==80
    fixtures=[]
    for dtype in ("float32","float64"):
        for value in ([[.12,.59],[.81,-.06]],[[.5,.25],[0.,0.]]):
            a=np.array(value,dtype=dtype)
            q=np.array([[.15,-.2],[-.75,.9],[.5,.1]])
            fixtures.append(dict(dtype=dtype,A=a.tolist(),q=q.tolist(),issued=correct(q,a).tolist()))
    save(destination,dict(version=1,reader_pin=(run/"launch.txt").read_text().split()[0],
         source_freeze=ref(run/"sealed.json"),checkpoint41=ck,source_law=law,source_qualification=source,
         qualified=source_test["general-lateral/command/direct-minus-identity"]["mean"]<0,
         inverse_fixture=fixtures,rows=records))
    print(digest(destination),destination)


if __name__=="__main__":
    main()
