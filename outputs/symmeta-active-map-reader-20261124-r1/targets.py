"""Evaluate frozen checkpoints on complete matched physical panels, once, after source selection."""
import json
from pathlib import Path
import sys

import numpy as np

from observations import case_rows, digest, family_group, grouped, interval, mean_report, save
from learn import SEEDS, baselines, evaluate, restore

ARMS = ("UR5e","IIWA")
GROUPS = ("native","family","swap","general")


def differences(rows,metric,method,schedule,other,control):
    def group(reader,strategy,body=None):
        return grouped([r for r in rows if r["method"]==reader and r["strategy"]==strategy and
                        (body is None or r["arm"]==body)],metric,("task",))
    def pair(body):
        a,b=group(method,schedule,body),group(other,control,body)
        if a.keys()!=b.keys():
            raise ValueError("Unpaired endpoints")
        return {k[0]:a[k]-b[k] for k in sorted(a)}
    paired=pair(None)
    vals=list(paired.values())
    if not vals:
        return None
    return dict(mean=float(np.mean(vals)),ci95=interval(vals),tasks=paired,
                arms={arm:float(np.mean(list(pair(arm).values()))) for arm in ARMS})


def report(rows):
    summaries={}
    for name in (*GROUPS,"general+","general-","swap/general"):
        if name=="swap/general":
            # First average positive/reflecting general draws; give swap and general equal weight.
            # Duplicating swaps only balances the number of general draws in the fresh panel.
            sub=[r for r in rows if r["family_group"] in ("swap","general")]
            if any(r["family"]=="general+" for r in sub):
                sub=sub+[r for r in sub if r["family"]=="swap"]
        elif name in ("general+","general-"):
            sub=[r for r in rows if r["family"]==name]
        else:
            sub=[r for r in rows if r["family_group"]==name]
        if not sub:
            continue
        table={}
        for metric in ("command","motion","map","setpoint","support","box"):
            table[metric]={}
            for method in ("direct","incumbent","ridge","identity","true_map","goal_fit"):
                ms=[r for r in sub if r["method"]==method]
                table[metric][method]={s:mean_report([r for r in ms if r["strategy"]==s],metric)
                                       for s in ("passive","along","lateral")}
        contr={}
        for metric in ("command","motion"):
            c={}
            for s in ("passive","along","lateral"):
                c["direct-incumbent/"+s]=differences(sub,metric,"direct",s,"incumbent",s)
                c["direct-ridge/"+s]=differences(sub,metric,"direct",s,"ridge",s)
            for method in ("direct","incumbent","ridge"):
                for control in ("along","passive"):
                    c[method+"/lateral-"+control]=differences(sub,metric,method,"lateral",method,control)
            d=c["direct/lateral-along"]
            a=c["incumbent/lateral-along"]
            if d and a:
                paired={k:d["tasks"][k]-a["tasks"][k] for k in d["tasks"]}
                c["interaction"]=dict(mean=float(np.mean(list(paired.values()))),ci95=interval(list(paired.values())),
                                      tasks=paired,arms={k:d["arms"][k]-a["arms"][k] for k in ARMS})
            contr[metric]=c
        summaries[name]=dict(factorial=table,contrasts=contr)
    return summaries


def main():
    out,fixture,dev=map(Path,sys.argv[1:])
    seal=json.loads((out/"sealed.json").read_text())
    assert seal["selected"]==digest(out/"selected.json") and seal["test"]==digest(out/"source-test.json")
    selected=json.loads((out/"selected.json").read_text())
    for ck,sha in seal["checkpoints"].items():
        assert digest(out/ck)==sha
    source=json.loads((out/"source-test.json").read_text())
    if source["general-lateral/command/direct-minus-identity"]["mean"]>=0:
        save(out/"unresolved.json",dict(reason="Direct learner failed held-source lateral general correction against identity."))
        return
    prior,residual=np.array(selected["prior"]),selected["residual"]
    for label,path,taskids in (("fresh",fixture,(8,6,8,8)),("developmental",dev,(5,3,1,0))):
        raw=[]
        paths={}
        suites=("libero_10","libero_goal","libero_object","libero_spatial")
        for arm in ARMS:
            for suite,task in zip(suites,taskids,strict=True):
                d=path/f"{arm}-{suite}-{task}"
                raw.extend(case_rows(d,prior,residual))
                for f in sorted(d.glob("*")):
                    if f.is_file() and (f.name.endswith((".json",".npz"))) and "-check-" not in f.name:
                        paths[str(f)]=digest(f)
        assert len(raw)==len(ARMS)*4*(5 if label=="fresh" else 4)*5
        scored=baselines(raw,prior,residual,selected["ridge_multiplier"])
        for seed in SEEDS:
            scored+=evaluate(restore(out/f"seed-{seed}.pkl"),raw,seed,prior)
        save(out/f"{label}-rows.json",scored)
        save(out/f"{label}-summary.json",report(scored))
        save(out/f"{label}-files.json",paths)
        print(label,"complete",len(raw),"real support attempts",flush=True)


if __name__=="__main__":
    main()
