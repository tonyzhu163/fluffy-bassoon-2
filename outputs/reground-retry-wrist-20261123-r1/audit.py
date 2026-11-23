"""Independent CPU readback of the native-spawn and preloaded real-simulator fixtures."""
from pathlib import Path
from unittest.mock import patch
import os
import numpy as np


def run(out):
    import reference as ref
    import retry, summarize
    from opening import identical
    from reground.eval.jobs import load_traces
    cpu = out.parent
    stock, fast = (cpu/s/"pair" for s in ("native01","native03"))
    a,b = [ref.read(p/"contract.json") for p in (stock,fast)]
    keys = list(map(tuple,a["keys"]))
    assert a["keys"]==b["keys"] and len(keys)==4
    arrays = {}
    for name in ("opening.npz","support.npz"):
        with np.load(stock/name) as x, np.load(fast/name) as y:
            assert set(x.files)<=set(y.files)
            if name=="opening.npz":
                assert set(x.files)==set(y.files)
            assert all(identical(x[f],y[f]) for f in x.files)
            arrays[name]=sorted(x.files)
    science_fields = {}
    ignore = {"ms_per_step","ms_vla","wall_s"}
    for arm in retry.ARMS:
        rec,cmp = [ref.read(p/(arm+".json")) for p in (stock,fast)]
        assert rec["queries"]==cmp["queries"] and rec["budget"]==cmp["budget"]
        for x,y in zip(rec["rows"],cmp["rows"],strict=True):
            assert {k:v for k,v in x.items() if k not in ignore}=={k:v for k,v in y.items() if k not in ignore}
        left,right = [load_traces(p/arm) for p in (stock,fast)]
        assert set(left)==set(right)==set(keys)
        for key in keys:
            assert all(identical(v,right[key][f]) for f,v in left[key].items())
        science_fields[arm]=sorted(rec["rows"][0].keys()-ignore)
    store = out/"fixture-store"
    batch = retry.schedule().index(keys)
    dest=store/"batches"/f"batch{batch:03d}"
    dest.parent.mkdir(parents=True)
    dest.symlink_to(fast,target_is_directory=True)
    # Exercise the production archive join, open/partial budget union and summary writer on one real fixture batch.
    validate=retry.validate
    def check(path,ks):
        validate(path,ks,policy=b["policy"])
    with patch.dict(os.environ,RETRY_PIN=b["code"]), patch.object(retry,"validate",check), \
            patch.object(summarize,"selected",lambda grid,phase:[batch]):
        summarize.report(store,"goals")
    summary=ref.read(store/"readback-goals/summary.json")
    assert summary["new_pairs"]==len(keys) and summary["paired"]["success"]["all"]["n_unique_starts"]==4
    ref.write(out/"passed.json",dict(code=os.environ["RETRY_PIN"],sources=[a["code"],b["code"]], keys=keys,
        byte_exact_arrays=arrays, all_native_science_fields_equal=science_fields,
        complete_trace_fields=["X","A","A_vla","ref","speed"],paired_readback_n=summary["new_pairs"],
        physical_fixture_costs=summary["intended_physical_interactions"],gpu_hours=0))
