"""Repeat the paired reduction from saved episodes and compare with the live writer."""
import argparse
import json
from pathlib import Path
import socket

from study import require, summarize


def readback(out, cpu):
    parts = (out/"exit.txt").read_text().split()
    require(parts[0] == "0", "attempt did not exit zero")
    origin = json.loads((out/"origin.json").read_bytes())
    result = summarize(out, origin["pin"], cpu)
    expected = json.loads((out/("qualified.json" if cpu else "result.json")).read_bytes())
    require(result == expected and result["sources"] == origin["sources"], "writer/readback disagreement")
    require(all(r["release_row"] == origin["release_rows"][str(r["row"])] for r in result["rows"]),
            "released bank")
    if cpu:
        require(origin["cpu_affinity"] == [38,39], "qualification resources")
    else:
        probe = json.loads((out/"gpu.json").read_bytes())
        launch = dict(x.split("=",1) for x in (out/"launch.txt").read_text().split())
        require(len(parts) == 3 and 0 < int(parts[2])-int(parts[1]) < 3600 and parts[1] == launch["start"] and
                origin["host"] == probe["hostname"] == launch["host"] and launch["pin"] == origin["pin"]
                and probe["status"] == "passed" and len(probe["devices"]) == 1
                and "RTX 4090" in probe["devices"][0]["cuda_name"].upper(), "scheduler receipt")
    print(json.dumps(dict(readback="passed", reader_host=socket.gethostname(), records=result["episodes"],
                          pin=result["pin"], fixture=cpu, native_steps=sum(sum(r["controls"].values())
                          for r in result["rows"]), sources=result["sources"],
                          contrasts=result.get("contrasts"), successes=result.get("successes")), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    readback(args.out, args.cpu)
