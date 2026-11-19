"""Check prior published motion features both on the matched source data and on the original fit."""
import importlib.util
from itertools import product
import json
from pathlib import Path
import sys

from transfer import SOURCE, TARGET, GOALS, VARIANTS, evaluate, np, probe, read_tape


def main(target, evidence):
    other = Path(__file__).resolve().parents[1]/"libero-visual-witness-20261119-r1/transfer.py"
    spec=importlib.util.spec_from_file_location("bowl_prior",other)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    previous, trained=mod.prior_reader()
    src=probe.WITNESS/"cpu02"
    source=[]; labels=[]
    for p,g,v in product(SOURCE,GOALS,VARIANTS):
        with np.load(src/f"p{p}-g{g}-{v}.npz") as z:
            source.append(previous.features(z["issued"],z["states"])["history"])
            labels.append(z["hits"][:,g].any())
    x=np.stack(source); y=np.asarray(labels)
    sd=x.std(axis=0); scale=np.where(sd>1e-12,sd,1.)
    centers=[(x[y==label]/scale).mean(axis=0) for label in (False,True)]
    rows={}; matched={}; prior={}
    for p,g,v in product(TARGET,GOALS,(*VARIANTS,"QUERY")):
        k=p,g,v
        rows[k],_=read_tape(target,k,"motion",None,4)
        if v=="QUERY":continue
        with np.load(target/f"p{p}-g{g}-{v}.npz") as z:
            f=previous.features(z["issued"],z["states"])
        miss,hit=[np.linalg.norm(f["history"]/scale-c) for c in centers]
        matched[k]=float(miss/(miss+hit))
        prior[k]=previous.predict(trained,dict(goal=g,**f))["history"]
    output={}
    for name,scores in (("same_16_bowl_tapes_centroid",matched),("prior_64_bowl_tapes_neighbor",prior)):
        read=evaluate(rows,scores)
        output[name]={k:read[k] for k in ("strict","cyclic_all_assignments","physical_completion")}
    evidence.write_text(json.dumps(output,indent=2)+"\n")


if __name__=="__main__":
    main(*map(Path,sys.argv[1:]))
