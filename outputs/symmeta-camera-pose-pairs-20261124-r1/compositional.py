"""Add held-out compositional action transforms using the frozen physical bank renderer."""
import argparse
import os
from pathlib import Path

from render import ARCHIVE, SOURCE, collect, dump, panel, setup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    opts=ap.parse_args()
    assert os.sched_getaffinity(0) == {46} and not os.environ.get("CUDA_VISIBLE_DEVICES")
    setup(SOURCE)
    opts.out.mkdir(parents=True,exist_ok=False)
    rows, diagnostics = [], []
    for case, entries in panel(SOURCE, ARCHIVE, range(8)):
        for entry in entries[4:8]:
            row, diag=collect(opts.out,"UR5e",case,entry,None)
            rows.append(row)
            diagnostics.append(diag)
            dump(opts.out/"diagnostics.json",diagnostics)
            dump(opts.out/"bank.json",dict(rows=rows))
            print(row["id"],row["native_control_id"],len(rows),flush=True)


if __name__=="__main__":
    main()
