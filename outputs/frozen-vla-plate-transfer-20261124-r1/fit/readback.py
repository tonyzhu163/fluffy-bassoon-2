"""Independent readback of the sealed exports, pair schedule and native source observations."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter

import numpy as np


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def check(ok, detail):
    if not ok:
        raise ValueError(detail)


def audit(out, expected, seal, pin):
    code, start, end = map(int, (out / "exit.txt").read_text().split())
    check(code == 0 and 0 < end-start < 7200, "launcher exit or budget")
    check(sha(out / "fit.json") == expected, "family SHA")
    report = load(out / "fit.json")
    check(report["code_revision"] == pin and report["data_seal_sha256"] == sha(seal)
          and report["resident_stock"] == report["deployed_stock"], "pinned code, data or frozen base")
    check((report["seed"], report["batch"], report["epochs"], report["lr"], report["optimizer"],
           report["draws"]) == (20261124, 32, 8, .001, "Adam", {"train":2, "heldout":2}), "pilot schedule")
    check(sha(out / "optimizer.msgpack") == report["optimizer_sha256"], "Adam checkpoint")
    check(sha(out / "progress.jsonl") == report["progress_sha256"], "update history")
    logs = [json.loads(line) for line in (out / "progress.jsonl").read_text().splitlines()]
    check(len(logs) == report["updates_per_fit"] == 96, "complete update count")
    for i, v in enumerate(logs):
        epoch = i // 12
        order = np.random.default_rng(np.random.SeedSequence([20261124, epoch, 193])).permutation(384)
        check((v["step"], v["epoch"], v["batch"], v["rows"], v["order_sha256"]) ==
              (i+1, epoch+1, i%12+1, 32, hashlib.sha256(order.astype("<i8").tobytes()).hexdigest()), "update schedule")
        check(set(v["train_mse"]) == {"panda", "mixed"} and np.isfinite(list(v["train_mse"].values())).all(),
              "condition losses")
    data = load(seal)
    windows = {}
    exports = {}
    for label in ("panda", "mixed"):
        d = out / label
        head = load(d / "fit.json")
        check(sha(d / "fit.json") == report["exports"][label] and head["condition"] == label
              and head["resident_stock"] == report["resident_stock"] and head["code_revision"] == pin
              and head["data_seal_sha256"] == sha(seal) and head["updates"] == 96, "export receipt")
        check(sha(d / "head.npz") == head["head_sha256"], "adapter archive")
        with np.load(d / "head.npz", allow_pickle=False) as z:
            check(set(z.files) == {"history_in", "history_out"}, "unexpected trained block")
            for k, shape in (("history_in", (52,16)), ("history_out", (16,320))):
                v = z[k]
                check(v.shape == shape and v.dtype == np.float32 and np.isfinite(v).all(), k)
                h = hashlib.sha256(str((str(v.dtype), v.shape)).encode() + b"\0" + v.tobytes()).hexdigest()
                check(h == head["parameters"][k] and h != report["initial_parameters"][k], k)
        inp = load(d / "train-inputs.json")
        windows[label] = load(d / "windows.json")
        source = data["variants"][label]
        source_manifest = seal.parent / source["root"] / "manifest.json"
        check(sha(source_manifest) == source["manifest_sha256"], "sealed source manifest")
        trajectories = load(source_manifest)["splits"]["train"]["trajectories"]
        source_keys = [[r["id"], t] for r in trajectories for t in r["t"]]
        check(sha(d / "train-inputs.json") == head["draws_sha256"] and sha(d / "windows.json") == head["windows_sha256"],
              "training receipts")
        check(inp["identities"] == source_keys == [[r["trajectory_id"],r["t"]] for r in windows[label]]
              and inp["draws"] == [[i, j, (2*i+j)*2] for i in range(192) for j in range(2)], "source draw mapping")
        exports[label] = {"head": head["head_sha256"], "fit": sha(d / "fit.json")}
    one, two = windows["panda"], windows["mixed"]
    check([(r["layout"],r["phase"]) for r in one] == [(r["layout"],r["phase"]) for r in two], "paired slots")
    check(Counter(r["body"] for r in one) == {"Panda":192} and
          Counter(r["body"] for r in two) == {"Panda":96,"Sawyer":96}, "arm totals")
    val = load(out / "heldout-inputs.json")
    val_rows = load(out / "heldout-windows.json")
    check(sha(out / "heldout-inputs.json") == report["heldout_draws_sha256"] and
          sha(out / "heldout-windows.json") == report["heldout_windows_sha256"] and
          val["identities"] == [[r["trajectory_id"],r["t"]] for r in val_rows] and
          val["draws"] == [[i,j,(2*i+j)*2+1] for i in range(64) for j in range(2)], "paired validation draws")
    check({r["layout"] for r in one}.isdisjoint(r["layout"] for r in val_rows), "source validation disjointness")
    validation = {}
    for when in ("before", "after"):
        native = load(out / f"native-{when}.json")
        check(sha(out / f"native-{when}.json") == report[f"native_{when}_sha256"], "native action receipt")
        for body in ("Panda","Sawyer"):
            rec = native[body]
            check(set(rec["zero_actions_sha256"].values()) == {rec["stock_sha256"]} and rec["zero_max_abs"] == 0
                  and rec["nonempty_history"]["stock"]["route"]["graph"] == "stock_native"
                  and rec["nonempty_history"]["mixed"]["route"]["hlo_sha256"] ==
                      rec["nonempty_history"]["panda"]["route"]["hlo_sha256"], f"{body}: native deployment routes")
        metric = out / f"source-{when}.json"
        check(sha(metric) == report["source_metrics"][when], "source scores")
        validation[when] = load(metric)
    contrasts = {}
    for body in ("Panda", "Sawyer"):
        group = validation["after"][body]
        check(validation["before"][body]["stock"] == group["stock"], "stock scoring stability")
        layouts = sorted(group["stock"]["per_layout_native_and_issued"])
        check(len(layouts) == 4, "four held-out layout clusters")
        contrasts[body] = {}
        for ctrl in ("stock", "panda"):
            # Exact enumerated four-layout paired bootstrap, descriptive at this sample size.
            delta = [group["mixed"]["per_layout_native_and_issued"][l][0] -
                     group[ctrl]["per_layout_native_and_issued"][l][0] for l in layouts]
            boot = np.mean(np.asarray(delta)[np.indices((4,)*4).reshape(4,-1)], axis=0)
            contrasts[body][f"mixed_minus_{ctrl}"] = dict(delta=float(np.mean(delta)),
                    layout_deltas=delta, paired_layout_bootstrap95=np.quantile(boot,[.025,.975]).tolist())
    return dict(pin=pin, fit=expected, seal=sha(seal), exports=exports, stock=report["resident_stock"],
                phase_draws_paired=True, updates=96, heldout_windows=64, measured_gpu_seconds=end-start,
                last_update=logs[-1], validation_mse={b:{c:m["native_mse_32"] for c,m in g.items()}
                    for b,g in validation["after"].items()}, source_contrasts=contrasts)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("expected")
    p.add_argument("seal", type=Path)
    p.add_argument("pin")
    args = p.parse_args()
    print(json.dumps(audit(args.out, args.expected, args.seal, args.pin), indent=2, allow_nan=False))
