"""Label native pre-action observations of the fixed published Panda validation demonstration."""
import argparse
import json
from pathlib import Path

from waypoint_labels import LOOKAHEAD, process, sha, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    source=args.bank.resolve(strict=True)
    idx=json.loads((source/"index.json").read_text())
    if (idx["suite"],idx["case"],idx["task_id"],idx["bodies"]) != ("libero_goal","BASE",5,["Panda"]):
        raise ValueError("Expected native, released Panda source bank")
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    for side in ("Panda","aux"):
        (out/side).mkdir()
    result=dict(suite="libero_goal",case="BASE",task_id=5,bodies=["Panda"],status="running",attempts=[],
                label_schema=1,lookahead=LOOKAHEAD,archive_sha256=idx["archive_sha256"],
                source_index_sha256=sha(source/"index.json"))
    for r in idx["attempts"]:
        if not (r["accepted"] and r["pose"] in range(28,32) and r["route"]=="released_actions" and r["variant"]==0):
            continue
        if not (r["done"] and r["native_success"] and r["terminal_reward"]>0):
            raise ValueError("Invalid published validation episode")
        path=(source/r["file"]).resolve(strict=True)
        if not path.is_relative_to(source):
            raise ValueError("Source tape outside the verified bank")
        digest=sha(path)
        row=process((out,path,r,digest,(),digest))
        result["attempts"].append(row)
    result.update(status="complete",accepted=sum(row["accepted"] for row in result["attempts"]))
    write_json(out/"index.json",result)
    print("Published fixed Panda validation trajectories labelled",result["accepted"],flush=True)
    if not result["accepted"]:
        raise ValueError("No reproduced fixed Panda validation episode")


if __name__=="__main__":
    main()
