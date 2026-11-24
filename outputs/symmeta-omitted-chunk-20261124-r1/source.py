"""Build the preregistered source-only partition from native passive policy tapes."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from collect_ur5e import read_support
from eval_stage3_scaling import build_collection_plan
import eval_wider_family as pr7
from eval_wider_family import CHUNK_KEYS, LABEL_KEYS
from fluffy_bassoon.xemb.baselines import compute_dis_flow_maps
from fluffy_bassoon.xemb.embodiment import TRAIN_EMBODIMENTS, get_embodiment_spec
from fluffy_bassoon.xemb.wider_family import rot2_batch

HERE = Path(__file__).resolve().parent
CODE_OUTPUTS = Path(pr7.__file__).resolve().parents[1]
ROOT = Path("/mowenjie/fluffy-bassoon-2/outputs")
SCALE = ROOT / "libero-xemb-ur5e-scale-20261120-r1/gpu01/archive"
EARLIER = ROOT / "libero-xemb-ur5e-sysid-20261120-r1/gpu02col/archive"
DIVERSE = ROOT / "xemb-icml-campaign-20261121-r1/gpu12scale/stage3/diverse_shards"
ARCHIVES = {
    SCALE: "37f062958cb63b962ceefb760f0592fdb120da966369ee308b509e2d82449076",
    EARLIER: "65c079f1051e347d781d950643113be9d4de6b92e952e373503d78d0eeb3b82e",
}
# Each suite contributes three reader-fit, one calibration and two evaluation tasks.
SPLIT = {
    "libero_10": ((1, 2, 9), (6,), (4, 7)),
    "libero_goal": ((2, 5, 7), (8,), (4, 9)),
    "libero_object": ((2, 3, 6), (5,), (4, 9)),
    "libero_spatial": ((2, 3, 4), (6,), (5, 9)),
}
ROLES = ("readerfit", "cal", "eval")
KEYS = tuple(k for k in CHUNK_KEYS if k != "dis_flow")
IDS = ("split", "task", "arm", "physical_trace_id", "case_id")


def require(ok, message):
    if not ok:
        raise ValueError(str(message))


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def write_json(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def save(out, name, data):
    with (out / name).open("xb") as f:
        np.savez_compressed(f, **data)


def role(row):
    i, = [i for i, group in enumerate(SPLIT[row["suite"]]) if row["task_id"] in group]
    return ROLES[i]


def tasks_by_role():
    return {r: sorted(f"{s}/{t}" for s in SPLIT for t in SPLIT[s][i]) for i, r in enumerate(ROLES)}


def check_tape(t, label):
    require(np.array_equal(t["valid"], [True] * 3), label)
    require(np.array_equal(t["img_after"][:-1], t["img_before"][1:]), f"Chunk continuity: {label}")
    require(t["img_before"].shape == (3, 64, 64, 3) and t["img_before"].dtype == np.uint8, label)


def recorded(out):
    splits = read_json(CODE_OUTPUTS / "libero-meta-learning-20261117-r1/SPLITS.json")
    expected = {f"{s}/{t}" for s, ts in splits["train"].items() for t in ts}
    split = tasks_by_role()
    require(set.union(*(set(v) for v in split.values())) == expected and len(expected) == 24, split)
    require(sum(map(len, split.values())) == 24, "Repeated task across roles")
    judge_tasks = {f"{s}/{t}" for s, ts in splits["development_scaling"]["task_subsets"]["8"].items() for t in ts}
    require(judge_tasks <= set(split["readerfit"]), "Earlier judge tasks crossed held-out partition")
    with gzip.open(CODE_OUTPUTS / "symmeta-matched-flow-20261123-r1/evidence/source-membership.json.gz", "rt") as f:
        prior = {r["passive_path"]: r for r in json.load(f)}
    require(len(prior) == 1344, "Missing prior scale source hashes")

    by_role = {r: [] for r in (*ROLES, "fit")}
    metadata = []
    for folder, expected_sha in ARCHIVES.items():
        path = folder / "archive.json"
        require(sha(path) == expected_sha, path)
        source = "scale24" if folder == SCALE else "pilot8"
        archive = read_json(path)
        rows = [r for r in archive["trials"] if r["split"] == "train"]
        wanted = {(a, t, i, e.name) for a in ("Panda", "Sawyer")
                  for t in (expected if folder == SCALE else judge_tasks)
                  for i in range(4 if folder == SCALE else 2) for e in TRAIN_EMBODIMENTS}
        for row in rows:
            task = f"{row['suite']}/{row['task_id']}"
            group = role(row) if folder == SCALE else "fit"
            arm = row["arm"]
            wanted.remove((arm, task, row["init_state_id"], row["embodiment"]))
            spec = get_embodiment_spec(row["embodiment"])
            require(all(getattr(spec, k) == row[k] for k in LABEL_KEYS), row["index"])
            require(row["passive_controls"] == 15 and row["model_class"] in ("Mounted"+arm, "OnTheGround"+arm),
                    row["index"])
            rel = row["passive_path"]
            digest = sha(folder / rel)
            if folder == SCALE:
                ref = prior[rel]
                require(digest == ref["sha256"] and all(row[k] == v for k, v in ref.items() if k != "sha256"), rel)
            tape = read_support(folder, row, "passive_attempt1")
            check_tape(tape, rel)
            require(np.array_equal(tape["chunk_steps"], [5, 5, 5]), rel)
            # IDs contain no shift names or true factors. Every row is a physical archive recording.
            ids = dict(task=task, arm=arm, split=group, case_id=f"{source}-{row['index']:04d}",
                       physical_trace_id=digest)
            by_role[group].append((ids, {k:tape[k] for k in KEYS}, [row[k] for k in LABEL_KEYS]))
            metadata.append(dict(**ids, archive=str(path), tape=rel, tape_sha256=digest, reset=row["physics"],
                                 init_state_id=row["init_state_id"], sim_seed=row["seed"], model_class=row["model_class"],
                                 noise_sha256=row["passive_noise_sha256"], opening_sha256=row["passive_first_proposal_sha256"],
                                 first_frame_sha256=row["frame0_img_sha256"], first_state_sha256=row["frame0_state_sha256"]))
        require(not wanted, f"Incomplete source archive: {path}")
    require(len({r["physical_trace_id"] for r in metadata}) == len(metadata) == 1568, "Repeated source tape")
    for group, pieces in by_role.items():
        ids = {k:np.asarray([r[0][k] for r in pieces]) for k in IDS}
        tape = {k:np.stack([r[1][k] for r in pieces]) for k in KEYS}
        labels = {k:np.asarray([r[2][i] for r in pieces], dtype=np.float32) for i,k in enumerate(LABEL_KEYS)}
        if group != "fit":
            tape["dis_flow"] = compute_dis_flow_maps(tape["img_before"], tape["img_after"], tape["valid"])
        if group == "readerfit":
            save(out, "train-geom.npz", {**ids, **tape, **labels})
            continue
        xy = tape["issued_mean"][..., :2]
        public = {**ids, "img_before":tape["img_before"], "img_after":tape["img_after"], "cmd":xy,
                  "dz":tape["delta_eef"][..., 2], "valid":tape["valid"]}
        sealed = dict(case_id=ids["case_id"], true_yaw=labels["yaw_deg"], true_gain=labels["xy_scale"],
                      true_sign=labels["gripper_sign"].astype(np.int32))
        if group == "fit":
            rotation = rot2_batch(labels["yaw_deg"])
            public.update(sealed, world_cmd=labels["xy_scale"][:,None,None] * np.einsum("nij,nkj->nki",rotation,xy),
                          world_delta_xy=np.einsum("nij,nkj->nki",rotation,tape["delta_eef"][...,:2]))
        else:
            save(out, f"labels-{group}.npz", sealed)
            save(out, f"reader-{group}.npz", {**ids, **tape})
        save(out, f"{group}.npz", public)
        print(group, len(pieces), "distinct three-command traces", len(np.unique(xy.reshape(len(xy),-1),axis=0)), flush=True)
    return metadata


def stage3(out):
    lock = read_json(CODE_OUTPUTS / "symmeta-perceiver-retry-20261123-r1/export-inputs.json")["sha256"]
    plan = build_collection_plan()
    require(len(plan) == 1466, "Stage3 plan")
    records, pieces = [], []
    for i in range(8):
        shard = DIVERSE / f"shard-{i}"
        path = shard / "diverse.npz"
        require(sha(path) == lock[path.relative_to(ROOT).as_posix()], path)
        meta = read_json(shard / "meta.json")
        require(meta == plan[i::8], shard)
        sel = np.asarray([j for j, r in enumerate(meta) if role(r) == "readerfit"])
        with np.load(path, allow_pickle=False) as z:
            # The original archive stores 24 slots, three observed. Restrict membership before fitting.
            valid = z["valid"]
            require(np.all(valid[:,:3]) and not valid[:,3:].any(), shard)
            for k in (*LABEL_KEYS, "emb_idx"):
                require(np.array_equal(z[k], np.asarray([r[k] for r in meta], dtype=z[k].dtype)), (path,k))
            part = {k:np.ascontiguousarray(z[k][sel,:3]) for k in CHUNK_KEYS}
            part.update({k:z[k][sel] for k in LABEL_KEYS})
        for j in sel:
            r = meta[j]
            records.append(dict(**r, archive=str(path), member=int(j), task=f"{r['suite']}/{r['task_id']}"))
        pieces.append(part)
    require(len(records) == 732 and len({r["emb_idx"] for r in records}) == 128, "Stage3 fit coverage")
    fit = {k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    fit["task"] = np.asarray([r["task"] for r in records])
    fit["case_id"] = np.asarray([f"stage3-{r['index']:04d}" for r in records])
    save(out, "train-tuned.npz", fit)
    return records


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("pin")
    args = ap.parse_args()
    cv2.setNumThreads(1)
    rows = recorded(args.out)
    diverse = stage3(args.out)
    write_json(args.out / "source-membership.json", rows)
    write_json(args.out / "tuned-membership.json", diverse)
    write_json(args.out / "source-seal.json", dict(revision=args.pin, roles=tasks_by_role(),
               judge_fit_tasks=sorted({r["task"] for r in rows if r["split"]=="fit"}),
               counts=dict(Counter(r["split"] for r in rows)), tuned_fit=len(diverse),
               sha256={p.name:sha(p) for p in sorted(args.out.iterdir()) if p.suffix in (".npz",".json")}))
    print("Sealed two reader training sets, independent judge tapes and both held-out partitions.", flush=True)


if __name__ == "__main__":
    main()
