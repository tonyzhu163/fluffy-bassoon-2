"""Screen compact-table geometry in fixed seed order before running a new split."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


def screen(layout, folder, phase, lane, seeds):
    log=folder/f"batch-{phase:02d}-lane-{lane}.log"
    with log.open("x") as f:
        subprocess.run([sys.executable,"-Bu",str(layout),*map(str,seeds)],stdout=f,stderr=subprocess.STDOUT,check=True)
    return [json.loads(s.removeprefix("GEOMETRY ")) for s in log.read_text().splitlines() if s.startswith("GEOMETRY ")]


def admit(start, count, folder):
    folder.mkdir()
    layout=Path(__file__).resolve().parent.parent/"robocasa-compact-bowls-20261109-r1/layout.py"
    results=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for phase in range(12):
            first=start+phase*12
            futures=[pool.submit(screen, layout,folder,phase,i,range(first+4*i,first+4*i+4)) for i in range(3)]
            records=sorted([r for f in futures for r in f.result()],key=lambda r:r["seed"])
            assert [r["seed"] for r in records]==list(range(first,first+12))
            results.extend(records)
            selected=[r["seed"] for r in results if all(r["checks"].values())][:count]
            if len(selected)==count:
                data={"admitted":selected,"examined":[r for r in results if r["seed"]<=selected[-1]]}
                (folder/"admitted.json").write_text(json.dumps(data,indent=2))
                print("ADMITTED",*selected,flush=True)
                return
    raise ValueError("Not enough admissible seeds in the registered range")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("start",type=int)
    p.add_argument("count",type=int)
    p.add_argument("folder",type=Path)
    args=p.parse_args()
    admit(args.start,args.count,args.folder)
