"""Assemble the compact evidence and reader index from completed, independently rendered strata."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import tarfile

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path, help="tar containing the two release directories' bank, diagnostics, scores and summary JSON")
    ap.add_argument("destination", type=Path)
    args = ap.parse_args()
    rows, diagnostics, scores = [], [], []
    with tarfile.open(args.archive) as archive:
        for name in ("release", "compositional-release"):
            def record(key):
                return json.load(archive.extractfile(f"{name}/{key}.json"))
            bank = record("bank")["rows"]
            rows.extend(bank)
            diagnostics.extend(record("diagnostics"))
            scores.extend(record("scores"))
    rows.sort(key=lambda r: (r["role"], r["arm"], r["index"]))
    assert len({r["id"] for r in rows}) == len(rows) == len(diagnostics) == 72
    assert {r["index"] for r in rows if r["arm"] == "UR5e"} == {12*i+j for i in range(8) for j in range(4,12)}
    assert Counter(r["arm"] for r in rows) == {"Panda":4,"Sawyer":4,"UR5e":64}
    view_names = ("nominal","orbit-","orbit+","roll-","roll+")
    assert len(scores) == sum(len(r["views"]) for r in rows) == 360
    for r in rows:
        assert set(r["views"]) == set(view_names)
        assert all("motion_npz" in v for v in r["views"].values())
    bank_bytes = json.dumps(dict(rows=rows),separators=(",",":"),allow_nan=False).encode()
    (args.destination/"reader-bank.json.gz").write_bytes(gzip.compress(bank_bytes,mtime=0))
    stats = {}
    draws = np.random.default_rng(1124).integers(0,8,(10000,8))
    for view in view_names:
        cond = [r for r in scores if r["arm"]=="UR5e" and r["view"]==view]
        result = dict(common_overlap=float(np.mean([r["overlap"] for r in cond])))
        for metric,start,key in (("tcp_error_px",1,"post_reset_tcp_px"),("tcp_error_px",0,"four_boundary_tcp_px"),
                                 ("rgb_mae",0,"rgb_mae")):
            by_task = {k:np.array([np.mean([r["scores"][k][metric][start:] for r in cond if r["shard"]==i])
                                   for i in range(8)]) for k in ("identity","correct","opposite")}
            for baseline in ("identity","opposite"):
                by_task["correct_minus_"+baseline]=by_task["correct"]-by_task[baseline]
            result[key]={k:dict(mean=float(v.mean()),by_shard=v.tolist(),ci95=np.quantile(v[draws].mean(axis=1),[.025,.975]).tolist())
                         for k,v in by_task.items()}
        stats[view]=result
    marker = [e for d in diagnostics for v in d["landmarks"] for e in v["errors_px"]]
    mm = [e for d in diagnostics if "archive_endpoint_distances_mm" in d for e in d["archive_endpoint_distances_mm"][1:]]
    lookup = {r["id"]:r for r in rows}
    dz = [abs(p[2]-lookup[d["id"]]["plane_z_m"])*1000 for d in diagnostics
          if lookup[d["id"]]["arm"]=="UR5e" for p in d["world_tcp_m"][1:]]
    out = dict(all_64=stats, physical_landmarks=len(marker),landmark_max_px=max(marker),landmark_median_px=float(np.median(marker)),
               target_post_reset_abs_dz_mm=dict(median=float(np.median(dz)),max=max(dz)),
               archive_to_fresh_post_reset_tcp_mm=dict(median=float(np.median(mm)),max=max(mm)),
               arm_counts=dict(Counter(r["arm"] for r in rows)),source_and_heldout_views=len(scores),
               n_original_reconstructed=sum(r["original_orbit"] for r in rows))
    (args.destination/"geometry-summary.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
