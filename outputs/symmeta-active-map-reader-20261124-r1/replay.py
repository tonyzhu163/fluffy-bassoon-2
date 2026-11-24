"""Independent empirical check of a non-axis-aligned relabel on an actual source pulse."""
import json
from pathlib import Path
import sys

import numpy as np

from fixture import ArmEnv, Numeric, schedule
from observations import save


def main():
    bank,dest=map(Path,sys.argv[1:])
    folder=bank/"Panda-libero_10-1"
    meta=json.loads((folder/"case.json").read_text())
    assert meta["split"]=="train"
    with np.load(folder/"archive-controls.npz") as z:
        base=z["support_chunk_means"]
    # Search only for feasibility, before any replay. Preserve nonzero lateral excitation.
    for family in ("general+","general-","swap","family"):
        a=np.array(meta["maps"][family]["A"])
        with np.load(folder/f"{family}-orth+.npz") as z:
            u,fk,ctrl=z["issued"].astype(float),z["fk"],z["physical"]
        if np.max(abs(u-np.repeat(base,5,axis=0)))<1e-6:
            continue
        for deg in (19,-19,11,-11,5,-5,2,-2):
            angle=np.deg2rad(deg)
            t=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            target=a@t.T
            b=base.copy()
            b[:,:2]=b[:,:2]@t.T
            v=u.copy()
            v[:,:2]=v[:,:2]@t.T
            if max(np.linalg.norm(target,np.inf),abs(v).max(),abs(b).max())>1:
                continue
            issued=np.repeat(list(schedule(b,"orth+",meta["source_amplitude"])),5,axis=0)
            if abs(v-issued).max()<8e-8:
                break
        else:
            continue
        break
    else:
        raise ValueError("No continuous feasible active view in the preregistered first source case")
    with ArmEnv("Panda","libero_10",1) as env:
        first=env.reset(meta["support_layout"],meta["support_seed"])
        assert first["environment_reproducibility"]["initial_state_fingerprint"]==meta["support_reset"]["initial_state_fingerprint"]
        sim=Numeric(env,first["observation"])
        _,orig=sim.run(u,a,dest/"original.npz")
        _,remap=sim.run(v,target,dest/"remapped.npz")
    np.testing.assert_array_equal(orig["physical"],ctrl)
    np.testing.assert_allclose(orig["fk"],fk,atol=1e-12,rtol=0)
    save(dest/"replay.json",dict(source=str(folder),family=family,angle_deg=deg,schedule="orth+",
         original_controls=len(orig["issued"]),remapped_controls=len(remap["issued"]),
         analytic_max_abs_controller_difference=float(abs(u[:,:2]@a.T-v[:,:2]@target.T).max()),
         collected_max_abs_controller_difference=float(abs(orig["physical"]-remap["physical"]).max()),
         max_per_step_fk_difference_m=float(np.linalg.norm(orig["fk"]-remap["fk"],axis=-1).max()),
         original_vs_archive_max_abs_fk_m=float(abs(orig["fk"]-fk).max()),
         max_schedule_discrepancy=float(abs(v-issued).max())))


if __name__=="__main__":
    main()
