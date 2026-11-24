"""Build a native input packet from the source worker's sealed export and the original case panel."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = Path("/mowenjie/fluffy-bassoon-2/outputs")
SOURCE = "symmeta-camera-real-source-20261124-r1"
FIT = ROOT / SOURCE / "fit01"
CACHE = ROOT / SOURCE / "cache01"
FIT_PIN = "57fca56ab35a878d8de4672ac33118152fc2d248"
READY = "d9f210ca09e9299e189f86ab7b44407de3cff0187617b7478635b2d803eb72f9"
SEAL = "83e9c4bdc9828a68bcb4dae230b1dbd681d325f69c360c289f01c662143ed5cc"
PLAN = "eefac4d1c6cb3d3761ab2bb3ceeb9643d8ea2c282e5fb415b17e9d1dd640d029"
UR = "symmeta-camera-frame-native-20261124-r1"
KIN = "symmeta-camera-independent-body-20261124-r1"
ORIGINAL = {
    "UR5e": (ROOT/UR/"gpu01", "panel.json", "7f0122115f7d488c2d1bc85fcd793e5fa33457e3ea5f77a93e2db9aee5e567e1"),
    "Kinova3": (ROOT/KIN/"native02", "sites.json", "1e6e54bf0cc9d89e12f7ef773398d4926ab0fcd992a8c8ccc773464b97c932e9"),
}
LAUNCH = {"UR5e":"b9ddfdf765e20977088c41c1ec9c87027ec8d1f72e41a1aa674fa2263a116727",
          "Kinova3":"30f96d30f226125859b79acf9b5d2e3c9fb7cc1db55c5183ca12182d597ba2e5"}
CASES = {"UR5e":"df1c8ba4df813cb1962c8646443dbd2014f1b925283649df86756da621dc0b5f",
         "Kinova3":"7a7f704bab9cc9d33ce6c1fbeb7e9722f22ed72553c67cac8303582a22769795"}
RECEIPT = {"UR5e":"1238d75eb88156825fb6ef0b4e23f54c2de3c0e5ab1a10ed1b46fb6681836117",
           "Kinova3":"4418360848aefcca0905fa75da48dd98e520baea4ec62721c8393e74275b5c52"}
BANK = {
    "UR5e": (ROOT/"symmeta-camera-pose-pairs-20261124-r1/cpu01/ood64/bank.json",
             "8071d5d451fb625fc3cf611cfb94ff3d16bf43ae280b565f7d564cc51f494f08"),
    "Kinova3": (ROOT/KIN/"cpu01/bank64.json", "8639817ce59b79ecfe26564e94c0bb61352a3a5942b13a5c8edd999a6c5d6e60"),
}
WARP = ROOT/"symmeta-camera-frozen-preprocess-20261124-r1/physical01/physical-predictions.json.gz"
WARP_SHA = "abf7ec349e6e1a0b6fe52cf0328b62bc18f372d49a197de8139cdfe0cc3902d4"
SHIPPED = "54157016de0af4f60c63ba8f9fbf782ee3832ef66699dac9dff0028d9ff15c86"
HEADS = ("yaw_deg","xy_scale","gripper_sign")
SENSORS = ("issued_mean","delta_eef","delta_jaw","valid")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked(path, sha):
    raw = Path(path).read_bytes()
    if digest(raw) != sha:
        raise ValueError(f"Digest mismatch: {path}")
    return raw


def read(path, sha):
    raw = checked(path, sha)
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def source_selection():
    ready = read(CACHE/"ready.json", READY)
    seal = read(FIT/"sealed.json", SEAL)
    assert (FIT/"exit.txt").read_text().split()[0] == "0"
    assert (FIT/"launch.txt").read_text().split() == [FIT_PIN,"job-o2g2s29tk504-master-0","fit-score","cache01",READY]
    assert seal["git_pin"] == FIT_PIN and seal["ready_sha256"] == READY
    assert seal["seeds"] == [41,42,43] and set(seal["arms"]) == {"A","B","digital_A","digital_B"}
    assert seal["params_per_arm"] == 2124438 and set(seal["training_exposures"]) == {
        f"{s}-{n}" for s in seal["seeds"] for n in (2500,5000)}
    ids = read(HERE.parent/SOURCE/"source-ids.json", seal["source_ids_sha256"])
    assert seal["source_ids_sha256"] == ready["source_ids_sha256"]
    split = {k:[(r["suite"],r["task_id"]) for r in ids["rows"] if r["split"] == k] for k in ("train","val")}
    assert len(split["train"]) == 128 and len(set(split["train"])) == 16
    assert len(split["val"]) == 64 and len(set(split["val"])) == 8 and not set(split["train"]) & set(split["val"])
    assert all(split["val"].count(t)==8 for t in set(split["val"]))
    assert seal["digital_ids_sha256"] == ready["sha256"]["digital-ids.json"]
    checked(CACHE/"digital-ids.json",seal["digital_ids_sha256"])
    curves = json.loads((FIT/"curves.json").read_bytes())
    means = {}
    for n in (2500,5000):
        rows = [r for r in curves if r["arm"]=="B" and r["step"]==n]
        assert sorted(r["seed"] for r in rows)==[41,42,43] and all(r["validation"]["n"]==64 for r in rows)
        means[n] = float(np.mean([r["validation"]["orbit"] for r in rows]))
        assert math.isfinite(means[n]) and means[n] == seal["source_B_mean"][str(n)]
    step = min((2500,5000), key=means.get)
    assert seal["steps"] == step
    weights = {}
    for model in ("A","B"):
        path = FIT/f"{model}-seed41-step{step}.msgpack"
        sha = seal["checkpoint_sha256"][path.name]
        checked(path,sha)
        weights[model] = (str(path),sha)
    assert weights["A"][1] != weights["B"][1]
    return step, weights


def original(body):
    folder, name, checksum = ORIGINAL[body]
    checked(folder/"launch.txt", LAUNCH[body])
    source = read(folder/name,checksum)
    registry = HERE.parent/folder.parent.name/"cases.csv"
    registered = list(csv.DictReader(checked(registry,CASES[body]).decode().splitlines()))
    rows = source["rows"] if body == "UR5e" else source
    assert len(rows) == len(registered) == 64
    assert [r["index"] for r in rows] == sorted(12*t+s for t in range(8) for s in range(4,12))
    cases = {int(r["index"]):r for r in registered}
    for r in rows:
        assert all(str(r[k]) == value for k,value in cases[r["index"]].items())
    return rows


def select_table(rows, body, step, weights):
    chosen = {}
    for number, row in enumerate(rows):
        model = row["model"]
        # The main table contains the two pure-projection companion fits and unused budgets.
        if model not in weights or row["seed"] != 41 or row["budget"] != step or row["calibration"] != "true" or not row["assigned"]:
            continue
        key = row["index"],model
        assert key not in chosen and row["input_representation"] == model
        assert row["selected"] is True and row["assigned"] is True and row["body"] == body
        assert (row["checkpoint_path"],row["checkpoint_sha256"]) == weights[model]
        chosen[key] = number,row
    assert len(chosen) == 128
    return chosen


def warp_table():
    history = HERE.parent/"symmeta-perceiver-retry-20261123-r1/evidence/train02pro/checkpoints.sha256"
    name, = [line.split()[1] for line in history.read_text().splitlines() if line.startswith(SHIPPED+" ")]
    checkpoint = ROOT/"symmeta-perceiver-retry-20261123-r1/train02pro"/name
    checked(checkpoint, SHIPPED)
    table = {}
    for n,p in enumerate(read(WARP,WARP_SHA)):
        if p["model"]=="frozen_warp" and p["seed"]==41 and p["view"] in ("orbit+","orbit-"):
            assert (p["checkpoint_path"],p["checkpoint_sha256"]) == (str(checkpoint),SHIPPED)
            assert p["calibration"]=="true" and p["supplied_H"]==p["H_ref_to_view"] and p["panel"]=="physical"
            k = p["index"],p["view"]
            assert k not in table
            table[k] = n,p
    assert len(table)==128
    return table


def assigned_site(body, original_row, physical, preds):
    r = dict(original_row)
    i = r["index"]
    angle = r["angle_deg"]
    view = "orbit+" if angle>0 else "orbit-"
    v = physical["views"][view]
    assert physical["arm"]==body and abs(angle)==(30 if body=="UR5e" else 36)
    assert v["angle_deg"] == angle and physical["physics"] == r["calibration_physics"]
    assert physical.get("assigned_view",view)==view and physical["native_control_id"]==r["embodiment"]
    assert physical["suite"]==r["suite"] and physical["task_id"]==r["task_id"] and physical["index"]==i
    assert physical["init_state_id"]==r["support_layout"]==11 and physical["seed"]==r["support_seed"]
    assert physical["archive_sha256"]==r["support_sha256"]
    checked(physical["archive_npz"],r["support_sha256"])
    checked(v["passive_npz"],v["sha256"])
    with np.load(v["passive_npz"],allow_pickle=False) as z:
        obs = np.concatenate((z["frame0_img"][None],z["img_after"]))
        assert obs.shape==(4,64,64,3) and np.array_equal(obs[:-1],z["img_before"]) and z["valid"].all()
        assert z["chunk_steps"].tolist()==[5,5,5]
        obs_sha = digest(obs.tobytes())
        sensors = {k:digest(z[k].tobytes()) for k in SENSORS}
    h_sha = digest(np.asarray(v["H_ref_to_view"],np.float64).tobytes())
    assert (obs_sha,h_sha)==(r["observed_sha256"],r["calibration_sha256"])
    old = r.pop("corrections")
    r.pop("selected",None)
    r.update(corrections={}, sources={}, shard=physical["shard"], noise_seed=r["query_seed"]+817117)
    if body=="UR5e":
        assert r["native_view_sha256"]==v["sha256"] and r["task_shard"]==r["shard"]
        assert all(old["shipped"][k]==r["nominal"][k] for k in HEADS[1:])
    else:
        assert physical["noise_seed"]==r["noise_seed"] and r["query_seed"]==physical["query_seed"]
    assert r["true_map"]==old["true_map"]==dict(zip(HEADS,(physical["control_yaw_deg"],physical["xy_scale"],physical["gripper_sign"])))
    for label,(n,p) in preds.items():
        assert p["physical_id"]==physical["id"] and p["view"]==view and p["camera_deg"]==angle
        assert all(p[k]==r[k] for k in ("suite","task_id","regime","embodiment","index"))
        assert (p["observed_sha256"],p["calibration_sha256"],p["sensor_sha256"])==(obs_sha,h_sha,sensors)
        assert (p["source_sha256"],p["native_view_sha256"])==(r["support_sha256"],v["sha256"])
        heads = {k:p[k] for k in HEADS}
        assert math.isfinite(heads["yaw_deg"]) and math.isfinite(heads["xy_scale"]) and heads["xy_scale"]>0
        assert type(heads["gripper_sign"]) is int and heads["gripper_sign"] in (-1,1)
        if body=="UR5e":
            y = heads["yaw_deg"]
            # Keep the original scalar bit-for-bit whenever already on the principal branch.
            wrapped = y if -180<=y<180 else (y+180)%360-180
            r["corrections"][label] = dict(r["nominal"],yaw_deg=wrapped)
        else:
            r["corrections"][label] = heads
        r["sources"][label] = dict(row_number=n,model=p["model"],seed=41,heads=heads,
                                  checkpoint_sha256=p["checkpoint_sha256"],checkpoint_path=p["checkpoint_path"])
    r.update(view=view,native_view_sha256=v["sha256"],sensor_sha256=sensors,physical_id=physical["id"])
    return r


def build(body):
    checked(HERE/"PLAN.md",PLAN)
    step, weights = source_selection()
    receipt = read(FIT/f"{body}-receipt.json",RECEIPT[body])
    assert receipt["seal_sha256"]==SEAL and receipt["selected"]==step and receipt["assigned_supports"]==64
    bank_path, bank_sha = BANK[body]
    assert receipt["bank_sha256"]==bank_sha and receipt["sha256"]!=receipt["cross_sha256"]
    table = read(FIT/f"{body}-predictions.json.gz",receipt["sha256"])
    assert len(table)==receipt["rows"]==10752
    estimates = select_table(table,body,step,weights)
    old = original(body)
    physical = [r for r in read(bank_path,bank_sha)["rows"] if r["arm"]==body and r["role"]=="heldout"]
    bank = {r["index"]:r for r in physical}
    assert len(physical)==len(bank)==len(old)==64
    simple = warp_table() if body=="UR5e" else None
    sites = []
    for r in old:
        i = r["index"]
        preds = {model+"_real":estimates[i,model] for model in ("A","B")}
        if simple is not None:
            preds["frozen_warp"] = simple[i,r["view"]]
        sites.append(assigned_site(body,r,bank[i],preds))
    return dict(body=body,registration_sha256=PLAN,fit=str(FIT),fit_pin=FIT_PIN,ready_sha256=READY,
                seal_sha256=SEAL,source_receipt_sha256=RECEIPT[body],prediction_sha256=receipt["sha256"],
                selected_step=step,source_checkpoints=weights,seed=41,cases_sha256=CASES[body],
                bank=str(bank_path),bank_sha256=bank_sha,old_panel_sha256=ORIGINAL[body][2],
                old_launch_sha256=LAUNCH[body],original_warp_sha256=WARP_SHA if simple else None,
                rows=sites)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("body",choices=tuple(ORIGINAL))
    parser.add_argument("destination",type=Path)
    args = parser.parse_args()
    packet = build(args.body)
    raw = (json.dumps(packet,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    # A completed release is immutable. An identical invocation is idempotent.
    if args.destination.exists():
        checked(args.destination,digest(raw))
    else:
        with args.destination.open("xb") as f:
            f.write(raw)
    print(args.body,len(packet["rows"]),tuple(packet["rows"][0]["corrections"]),str(args.destination),digest(raw))
