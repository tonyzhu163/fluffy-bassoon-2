"""Adapt a parent-released final inference archive; never open an in-progress target directory."""
import argparse
from pathlib import Path
import re

from prepare import correct
from cohort import HERE, atomic, checked, digest, encode, parse, require
import numpy as np


def ref(path):
    return dict(path=str(path),sha256=digest(path.read_bytes()))


def pack(a):
    # Parent supplies final producer pin and digests. Only operator, key and archive identity fields leave this adapter.
    require(a.fit.is_absolute() and a.out.is_absolute() and re.fullmatch("[0-9a-f]{40}",a.reader_pin),"released paths/pin")
    require((a.fit/"exit.txt").read_text().split()[0]=="0", "parent release must name a completed producer")
    seal = parse(checked(a.fit/"sealed.json",a.seal_sha))
    source = parse(checked(a.fit/"source-test.json",seal["test"]))
    selected = parse(checked(a.fit/"selected.json",seal["selected"]))
    learned_sha = seal["checkpoints"]["seed-41.pkl"]
    require(selected["checkpoints"]["seed-41.pkl"]==learned_sha, "source selected checkpoint")
    checkpoint = a.fit/"seed-41.pkl"
    checked(checkpoint,learned_sha)
    raw = parse(checked(a.fit/"fresh-rows.json",a.predictions_sha))
    files = parse(checked(a.fit/"fresh-files.json",a.files_sha))
    from_path = Path(seal["bank"])
    staged = {}
    for row in raw:
        if row["method"] not in ("direct","incumbent") or row["family"] not in ("general+","general-"):
            continue
        if row["seed"] != (41 if row["method"]=="direct" else 0):
            continue
        suite,task = row["task"].split("/")
        key = row["arm"],suite,int(task),row["family"],row["mode"]
        which = "direct" if row["method"]=="direct" else "structured"
        bucket = staged.setdefault(key,{})
        require(which not in bucket,"duplicate inference output")
        bucket[which] = row["estimate"]
    require(len(staged)==80 and all(set(v)=={"direct","structured"} for v in staged.values()),
            "released physical support prediction coverage")
    refs = parse((HERE.parent/"frozen-vla-multiarm-confirmation-20261123-r1/evidence/production-native.json").read_bytes())
    law = HERE.parent/"frozen-vla-motion-models-20261123-r1/evidence/source-fit.json"
    checked(law,refs["source_fit_sha256"])
    rows = []
    for (arm,suite,task,family,mode),est in sorted(staged.items()):
        d = from_path/f"{arm}-{suite}-{task}"
        case = d/"case.json"
        meta = parse(checked(case,files[str(case)]))
        tape = d/f"{family}-{mode}.npz"
        checked(tape,files[str(tape)])
        with np.load(tape,allow_pickle=False) as z:
            h = digest(z["issued"].astype("<f8").tobytes()+z["fk"].astype("<f8").tobytes())
        rows.append(dict(arm=arm,suite=suite,task=task,family=family,schedule=mode,seed=41,
                         case=dict(path=str(case),sha256=files[str(case)]),
                         source_archive={k:meta["source_record"][k] for k in ("path","sha256")},
                         support_tape=dict(path=str(tape),sha256=files[str(tape)]),
                         source_law_sha256=refs["source_fit_sha256"],checkpoint_sha256=learned_sha,input_sha256=h,
                         structured_dtype="float64",direct_dtype="float32",**est))
    fixtures = []
    for dtype in ("float64","float32"):
        for matrix in ([[.25,.5],[.125,.25]],[[.125,.625],[.75,-.25]]):
            A = np.array(matrix,dtype=dtype)
            q = np.array([[.125,-.25],[-.75,.875],[.5,.125]],dtype=np.float64)
            fixtures.append(dict(A=A.tolist(),dtype=dtype,q=q.tolist(),issued=correct(q,A).tolist()))
    p = dict(version=1,reader_pin=a.reader_pin,source_freeze=ref(a.fit/"sealed.json"),
             checkpoint41=ref(checkpoint),source_law=ref(law),source_qualification=ref(a.fit/"source-test.json"),
             qualified=source["general-lateral/command/direct-minus-identity"]["mean"]<0,
             frozen_predictions=dict(path=str(a.fit/"fresh-rows.json"),sha256=a.predictions_sha),
             archived_files=dict(path=str(a.fit/"fresh-files.json"),sha256=a.files_sha),
             inverse_fixture=fixtures,rows=rows)
    data = encode(p)
    atomic(a.out,data)
    print(digest(data),flush=True)


if __name__=="__main__":
    c=argparse.ArgumentParser()
    c.add_argument("fit",type=Path)
    c.add_argument("reader_pin")
    c.add_argument("seal_sha")
    c.add_argument("predictions_sha")
    c.add_argument("files_sha")
    c.add_argument("out",type=Path)
    pack(c.parse_args())
