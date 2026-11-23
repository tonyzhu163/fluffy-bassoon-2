"""Join cached receipts. For differing trajectories only, locate their first differing saved sample."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import subprocess

from native_read import *


def read_gz(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def write_rows(path, rows):
    tmp=path.with_suffix(".part")
    with gzip.open(tmp,"wt") as f:
        for row in rows:
            f.write(json.dumps(row,separators=(",",":"),allow_nan=False)+"\n")
    tmp.replace(path)


def paired(a,b):
    return {"original":a,"controlled":b}


def key_labels(key):
    return dict(zip(("task","init_id","rep"),key))


def first_diff(a,b):
    length=min(len(a),len(b))
    check(a.dtype==b.dtype==np.float32 and a.shape[1:]==b.shape[1:], (a.shape,b.shape))
    unequal=(a[:length].view(np.uint32)!=b[:length].view(np.uint32)).reshape(length,-1).any(axis=1)
    at=np.flatnonzero(unequal)
    if len(at):
        return dict(index=int(at[0]),reason="value")
    check(len(a)!=len(b),"Digest disagrees with bit comparison")
    return dict(index=length,reason="length")


def differences(old_path,new_path,batch,key,fields):
    if not fields:
        return {}
    old=sorted((old_path/"traces").glob("*.npz"))[batch]
    new=sorted((new_path/"traces").glob("*.npz"))[batch]
    name="__".join(map(str,key))
    with np.load(old,allow_pickle=False) as left,np.load(new,allow_pickle=False) as right:
        return {f:first_diff(left[f"{name}__{f}"],right[f"{name}__{f}"]) for f in fields}


def compare(c,r):
    key=c["key"]
    check(r["key"] == key and r["seed"] == c["seed"] and r["shard"] == c["shard"],key)
    s,h=c["seed"],c["shard"]
    discord=[p for p,v in r["pairs"].items() if v["different_indices"]]
    facts=c["opening"]
    result=dict(seed=s,shard=h,**key_labels(key),batch=c["batch"],lane=c["lane"],
                original_discordant_pairs=discord,opening=facts,original_pair_checks=r["pairs"],
                historical_stream_agentview_sha256=c["historical_agentview"],
                historical_stream_proposal50_sha256=c["historical_stream_proposal50"],
                arms={})
    for arm in ARMS:
        a,b=r["arms"][arm],c["arms"][arm]
        check(a["x0"] == b["x0"] == facts["x0_sha256"],(key,arm,"reset"))
        check(a["csv"]["noise_seed"]==b["csv"]["noise_seed"]==str(facts["noise_seed"]),key)
        check(int(a["csv"]["steps"])>=10 and int(b["csv"]["steps"])>=10,key)
        check(all(a["csv"][k]==b["csv"][k] for k in (*KEYS,"cfg_id","method")),key)
        trace_fields=[f for f in FIELDS if a["trace_sha256"][f]!=b["trace_sha256"][f]]
        csv_diff={f:paired(a["csv"][f],b["csv"][f]) for f in a["csv"] if a["csv"][f]!=b["csv"][f]}
        stage=ROOT/f"common-opening/shards/seed{s}-shard{h:03d}"/arm
        first=differences(original(s,h,arm),stage,c["batch"],key,trace_fields)
        prefix=a["prefix_sha256"]["A_vla"]==facts["proposal10_sha256"]
        result["arms"][arm]=dict(metrics=paired(a["metrics"],b["metrics"]),safety=paired(a["safety"],b["safety"]),
                                 original_proposal10_sha256=a["prefix_sha256"]["A_vla"],same_proposal10=prefix,
                                 trace_sha256=paired(a["trace_sha256"],b["trace_sha256"]),
                                 first_trace_differences=first,non_timing_csv_differences=csv_diff)
    return result



def public_poses(stage, rows):
    """Export small public pose vectors from each captured tape once for human readback."""
    poses=[]
    for batch in range(len(rows)//4):
        fp=stage/"capture"/f"batch{batch}.npz"
        with np.load(fp, allow_pickle=False) as tape:
            for lane in range(4):
                row=rows[4*batch+lane]
                prefix=f"{lane}/obs/robot_state"
                eef=tape[prefix+"/eef/pos"]
                jaw=tape[prefix+"/gripper/qpos"]
                x=tape[f"{lane}/X0"]
                check(digest(x)==row["opening"]["x0_sha256"] and
                      np.array_equal(eef.astype(np.float32),x[:3]) and
                      np.array_equal(jaw.astype(np.float32),x[9:11]), row["key"] if "key" in row else row["init_id"])
                poses.append(dict(eef_pos=eef.tolist(),eef_quat_xyzw=tape[prefix+"/eef/quat"].tolist(),
                                  gripper_qpos=jaw.tolist()))
    return poses

def groups(rows):
    out={}
    for name,selected in (("all",rows),("original_discordant",[r for r in rows if r["original_discordant_pairs"]]),
                          ("original_concordant_companions",[r for r in rows if not r["original_discordant_pairs"]])):
        entry=out[name]=dict(assignments=len(selected),arms={})
        for arm in ARMS:
            aa=[r["arms"][arm] for r in selected]
            accumulated={}
            for field in ("success","unsafe"):
                orig=[a["metrics"]["original"][field] for a in aa]
                ctrl=[a["metrics"]["controlled"][field] for a in aa]
                accumulated[field]=dict(original=sum(orig),controlled=sum(ctrl),
                                        lost=sum(x>y for x,y in zip(orig,ctrl)),gained=sum(x<y for x,y in zip(orig,ctrl)))
            entry["arms"][arm]=dict(**accumulated,same_proposal10=sum(a["same_proposal10"] for a in aa),
                                    identical_all_five_traces=sum(not a["first_trace_differences"] for a in aa),
                                    identical_non_timing_csv=sum(not a["non_timing_csv_differences"] for a in aa))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("work",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    out=args.output
    out.mkdir(exist_ok=True)
    plans=grid()
    observed=read(args.work/"observed.json")
    orig_rows=[]; controlled=[]; receipts={}; unpaired=[]; pending=[]; affected=[]
    cache=args.work/"joined"
    cache.mkdir(exist_ok=True)
    for (s,h),keys in plans.items():
        tag=label(s,h)
        if tag not in observed:
            continue
        r=read_gz(args.work/"original"/f"{tag}.json.gz")
        check([tuple(v["key"]) for v in r]==keys,(s,h))
        for v in r:
            orig_rows.append(dict(seed=s,shard=h,**key_labels(v["key"]),noise_seed=int(v["arms"]["base"]["csv"]["noise_seed"]),
                                  x0_sha256={a:v["arms"][a]["x0"] for a in ARMS},
                                  proposal10_sha256={a:v["arms"][a]["prefix_sha256"]["A_vla"] for a in ARMS},pairs=v["pairs"]))
        if observed[tag]["discordant"]:
            affected.append(tag)
        fp=args.work/"controlled"/f"{tag}.json.gz"
        is_expected=bool(observed[tag]["discordant"]) or (s,h) in SEEN
        if not fp.exists():
            if is_expected:
                pending.append(tag)
            continue
        check(is_expected, f"Unselected controlled shard {tag}")
        data=read_gz(fp)
        cr=data.pop("rows")
        check(len(cr)==len(r),tag)
        data["launcher_receipts"]={}
        for arm,producer in data["attempts"].items():
            d=ROOT/"reference-native/attempts"/("common-opening-"+producer["attempt"])
            reservation=read(d/"reservation.json")
            check(reservation["host"]==producer["host"] and reservation["upstream"]==PIN
                  and reservation["policy"]==POLICY and reservation["seeds"]==[s], d)
            exited=read(d/"exit.json") if (d/"exit.json").exists() else None
            data["launcher_receipts"][producer["attempt"]]=dict(phase=reservation["phase"],
                host=reservation["host"],code=reservation["code"],started_utc=reservation["started_utc"],exit=exited)
        receipts[tag]=data
        # Cache the targeted differing-field readback. Later summaries read only compact data.
        done=cache/f"{tag}.json.gz"
        if not done.exists():
            rows=[compare(c,o) for c,o in zip(cr,r,strict=True)]
            tmp=done.with_suffix(".part")
            with gzip.open(tmp,"wt") as f:
                json.dump(rows,f,separators=(",",":"),allow_nan=False)
            tmp.replace(done)
        joined=read_gz(done)
        pose_path=cache/f"{tag}-poses.json"
        if not pose_path.exists():
            save(pose_path,public_poses(ROOT/f"common-opening/shards/seed{s}-shard{h:03d}",joined))
        for row,pose in zip(joined,read(pose_path),strict=True):
            row["opening"]["public_pose"]=pose
        controlled.extend(joined)
    for file in sorted((args.work/"controlled").glob("*.json.gz")):
        tag=file.name.split(".")[0]
        if tag not in observed:
            unpaired.append(tag)
    write_rows(out/"original-openings.jsonl.gz",orig_rows)
    write_rows(out/"controlled-cases.jsonl.gz",controlled)
    save(out/"shard-receipts.json",receipts)
    agreement={}
    for a,b in PAIRS:
        name=f"{a}/{b}"
        facts=[r["pairs"][name] for r in orig_rows]
        agreement[name]=dict(same_start=sum(p["start"] for p in facts),
                             bit_identical_starts=sum(r["x0_sha256"][a]==r["x0_sha256"][b] for r in orig_rows),
                             bit_identical_ten_proposals=sum(r["pairs"][name]["steps"]==10 and
                                                            r["proposal10_sha256"][a]==r["proposal10_sha256"][b]
                                                            for r in orig_rows),
                             all_ten_available=sum(p["steps"]==10 for p in facts),
                             same_first_ten=sum(not p["different_indices"] and p["steps"]==10 for p in facts),
                             native_grid_exact=sum(p["native_grid"] for p in facts))
    summary=dict(utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),original_shards=len(observed),
                 original_assignments=len(orig_rows),original_grid=agreement,affected=affected,
                 original_discordant=sum(any(p["different_indices"] for p in r["pairs"].values()) for r in orig_rows),
                 missing_originals=[label(*k) for k in plans if label(*k) not in observed],
                 missing_controlled=pending,
                 preassigned_without_originals=[label(*k) for k in sorted(SEEN) if label(*k) not in observed],
                 verified_controls_waiting_original=unpaired,
                 producer_exit_pending=[attempt for d in receipts.values() for attempt,rec in d["launcher_receipts"].items()
                                        if rec["exit"] is None],
                 producer_nonzero_exits=[attempt for d in receipts.values() for attempt,rec in d["launcher_receipts"].items()
                                         if rec["exit"] is not None and rec["exit"]["exit_code"]!=0],
                 verified_shards=list(receipts),batches=sum(len(r["tapes"]) for r in receipts.values()),
                 controlled_vs_historical_stream=dict(same_agentview=sum(r["opening"]["agentview_sha256"]==r["historical_stream_agentview_sha256"]
                                                                         for r in controlled),
                                                     same_proposal50=sum(r["opening"]["proposal50_sha256"]==r["historical_stream_proposal50_sha256"]
                                                                         for r in controlled)),
                 by_subset=groups(controlled),
                 by_shard={tag:groups([r for r in controlled if label(r["seed"],r["shard"])==tag]) for tag in receipts})
    save(out/"summary.json",summary)
    print(json.dumps({k:summary[k] for k in ("original_shards","affected","verified_shards","missing_controlled","batches")}))


if __name__=="__main__":
    main()
