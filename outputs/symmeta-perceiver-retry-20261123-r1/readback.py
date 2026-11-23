"""Independent CPU readback of the sealed four-panel Pro6K native comparison."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
from itertools import product
import json
from pathlib import Path
import re
from statistics import stdev
from tempfile import TemporaryDirectory

PRED = "c8420c2cb7d0cb38e6f0d73b3a20ce47997dbef7657b652d45eaaa8988eb6c0b"
CONTRACT = "e559a9d57c1ba73e8bb8a06cb5b6b1c7e69d252aaa539718b99b212ca8669441"
DEPLOYMENT = "d568535b758f2b97b382f9b3e280376e769b5942"
ATTEMPT = "native02g4090"
HOST = "job-2s6q9a9xe96e-master-0"
TRAIN_EXPORT = Path("/mowenjie/fluffy-bassoon-2/outputs/symmeta-perceiver-retry-20261123-r1/train02pro/predictions.json")
NVML_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
HISTORY = "68a3c36c91bb13dc2ea3c7d8387e19d7cca6bde7470492f0be356d6d318152c7"
PANELS = {"gpu09test": "ur5e-pi05", "gpu09iiwa": "iiwa-pi05",
          "gpu06kinova3": "kinova3-pi05", "gpu08pi0fix": "ur5e-pi0"}
SEEDS = (41, 42, 43)
READERS = ("tuned", "symmeta")
OOD = {"compositional_ood", "continuous_angle_ood"}
KEY = ("panel", "index", "seed")
CASE = ("suite", "task_id", "embodiment", "regime", "tape_sha256")
STABLE = ("arm", "model_class", "joint_count", "scene_fingerprint", "perturbation_plan",
          "worker_step_index", "action_schema", "noise_seed", "noise_sha256", "horizon", "prompt")
HASH = re.compile("[a-f0-9]{64}")


def check(ok, message):
    if not ok:
        raise ValueError(str(message))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def unique(fields):
    obj = dict(fields)
    check(len(obj) == len(fields), "duplicate JSON key")
    return obj


def load(path, expected=None):
    raw = path.read_bytes()
    if expected is not None:
        check(sha(raw) == expected, f"SHA-256 mismatch: {path}")
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid_number)


def invalid_number(value):
    raise ValueError(f"nonfinite JSON number: {value}")


def key(row):
    return tuple(row[k] for k in KEY)


def pair_path(root, row):
    return root / "pairs" / row["panel"] / f"s{row['seed']}" / f"{row['index']:02d}.json"


def frozen_inputs(root, predictions):
    frozen = load(predictions, PRED)
    check(sha((root / "predictions.json").read_bytes()) == PRED, "native prediction copy")
    contract = load(root / "contract.json", CONTRACT)
    check(contract["predictions_sha256"] == PRED and set(contract["panels"]) == set(PANELS), "input contract")
    expected = set(product(PANELS, range(96), SEEDS))
    by_key = {key(r): r for r in frozen["rows"]}
    check(set(by_key) == expected and len(frozen["rows"]) == len(expected), "frozen case coverage")

    path = Path(__file__).resolve().parents[1] / "symmeta-paired-statistics-20261123-r1" / "outcomes.csv"
    raw = path.read_bytes()
    check(sha(raw) == HISTORY, "historical reference digest")
    archive = {(r["panel"], int(r["index"])): r for r in csv.DictReader(io.StringIO(raw.decode()))}
    cases = {}
    for panel, name in PANELS.items():
        p = load(root / panel / "panel.json", contract["panels"][panel])
        check(p["panel"] == panel and p["predictions_sha256"] == PRED, f"{panel} input origin")
        for r in p["rows"]:
            k = key(r)
            pred = by_key[k]
            check(k not in cases and r["panel"] == panel and r["shard"] == r["index"] // 12 and
                  canonical({f: r[f] for f in pred}) == canonical(pred), f"{k} frozen join")
            h = archive[name, r["index"]]
            q = r["query"]
            check((r["suite"], str(r["task_id"]), r["embodiment"], r["regime"], str(r["shard"]),
                   q["variant"], str(q["init_state_id"]), str(q["seed"])) ==
                  tuple(h[f] for f in ("suite", "task_id", "embodiment", "regime", "task_order",
                                        "variant", "query_init_state_id", "query_reset_seed")), f"{k} historical identity")
            cases[k] = (r, p["refs"][r["shard"]], int(h[f"s{r['seed']}"]))
    check(set(cases) == expected, "sealed case coverage")
    return cases


def pci(value):
    match = re.fullmatch(r"([\da-fA-F]+):([\da-fA-F]{2}):([\da-fA-F]{2})\.([0-7])", value)
    check(match, f"invalid PCI address {value!r}")
    domain, bus, slot, function = match.groups()
    return f"{int(domain, 16):04x}:{bus.lower()}:{slot.lower()}.{function}"


def four_cards(rec, probe):
    check(probe["hostname"] == HOST and probe["status"] == "passed" and probe["purpose"] == "evaluation"
          and probe["allocation_type"] == "baige_container" and probe["admission_policy"] == "exclusive_scheduler_evaluation"
          and probe["idle_utilization_required"] is False and probe["exclusive_training_admission"] is False
          and probe["utilization_role"] == "telemetry_only", "evaluation GPU probe")
    rows = rec["devices"]
    check(len(rows) == len(probe["devices"]) == 4, "expected four receipt/probe devices")
    assigned = probe["gpu_identity_mapping"]["assigned_nvml_uuids"]
    uuids = [d["uuid"] for d in rows]  # Receipt order is CUDA lane order. Scheduler order can differ.
    check(len(assigned) == len(set(uuids)) == len({u.lower() for u in uuids}) == 4
          and set(assigned) == set(uuids) and all(NVML_UUID.fullmatch(u) for u in assigned),
          "scheduler/NVML UUID coverage")
    assigned_set = set(assigned)
    pid = probe["probe_pid"]
    after = probe["compute_processes_after"]
    check(type(pid) is int and pid > 0 and probe["compute_processes_before"] == []
          and all(p["pid"] == pid for p in after) and {p["gpu_uuid"] for p in after} == assigned_set,
          "probe PID allocation")
    lanes = []
    for lane, (device, observed) in enumerate(zip(rows, probe["devices"], strict=True)):
        label = f"lane {lane}"
        check("egl_device" not in observed and
              canonical({k: v for k, v in device.items() if k != "egl_device"}) == canonical(observed),
              f"{label} receipt/probe device mismatch")
        inv = device["inventory"]
        name, token, egl = device["cuda_name"], device["visibility_token"], device["egl_device"]
        address = pci(device["pci_bus_id"])
        check(type(device["allocation_ordinal"]) is int and device["allocation_ordinal"] == lane
              and name.upper() == "NVIDIA GEFORCE RTX 4090" and inv["name"] == name
              and type(device["cuda_total_bytes"]) is int and device["cuda_total_bytes"] > 0
              and device["uuid"] == inv["uuid"] and address == pci(inv["pci_bus_id"])
              and inv["memory_used_mib"] <= probe["maximum_initial_memory_mib"]
              and isinstance(device["cuda_uuid"], str) and device["cuda_uuid"]
              and isinstance(token, str) and (token.isdigit() or token.startswith("GPU-"))
              and type(egl) is int and egl >= 0, f"{label} ordinal, hardware, PCI, UUID or selector")
        proof = device["allocation_check"]
        touched = set(uuids[:lane + 1])
        check(proof["elements"] == 1048576 and proof["result"] == 2097152
              and set(proof["touched_gpu_uuids"]) == touched
              and touched <= set(proof["own_pid_gpu_uuids"]) <= assigned_set,
              f"{label} GPU probe computation/PID UUIDs")
        lanes.append(dict(lane=lane, cuda_name=name, cuda_total_bytes=device["cuda_total_bytes"],
                          visibility_token=token, cuda_uuid=device["cuda_uuid"], nvml_uuid=device["uuid"],
                          pci_bus_id=address, egl_device=egl))
    check(len({d["cuda_name"] for d in lanes}) == 1, "heterogeneous CUDA device names")
    for field in ("cuda_uuid", "pci_bus_id", "visibility_token", "egl_device"):
        check(len({d[field] for d in lanes}) == 4, f"nonunique four-card {field}")
    return lanes


def origin(root):
    config_raw = (root / "config.txt").read_bytes()
    config = unique([line.split("=", 1) for line in config_raw.decode().splitlines()])
    check(root.name == ATTEMPT and config["host"] == HOST and config["commit"] == DEPLOYMENT
          and config["mode"] == "run" and config["cards"] == "4" and config["predictions_sha256"] == PRED
          and config["predictions"] == str(TRAIN_EXPORT), "native launch identity")
    check(config["reuse"] == "", "this launch must have no prior attempts")
    execution = (root / "execution.sha256").read_bytes()
    check(re.fullmatch(rb"[0-9a-f]{64}\n", execution), "invalid execution.sha256")
    digest = execution[:-1].decode()
    receipt_path = root / "receipts" / (digest + ".json")
    check({p for p in (root / "receipts").iterdir() if p.suffix.lower() == ".json"} == {receipt_path}, "single-job receipt coverage")
    rec = load(receipt_path, digest)
    check(type(rec["schema"]) is int and rec["schema"] == 1 and rec["hostname"] == HOST
          and rec["attempt"] == ATTEMPT and rec["deployment"] == DEPLOYMENT
          and rec["contract_sha256"] == CONTRACT, "native receipt identity")
    software = rec["software"]
    check(set(software) == {"jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "numpy", "flax",
                            "orbax-checkpoint", "openpi-client", "simulator", "xla_flags", "jax_matmul_precision", "driver"}
          and set(software["simulator"]) == {"mujoco", "robosuite", "numpy", "PyOpenGL"}, "receipt software fields")
    probe_sha = rec["gpu_probe_sha256"]
    check(isinstance(probe_sha, str) and HASH.fullmatch(probe_sha), "invalid GPU probe digest")
    probe = load(root / "gpu.json", probe_sha)
    lanes = four_cards(rec, probe)
    for panel in PANELS:
        for shard in range(8):
            path = root / panel / f"shard-{shard}-reuse.json"
            check(load(path) == [], f"unexpected recovered pairs at {path}")
    return dict(attempt=ATTEMPT, hostname=HOST, deployment=DEPLOYMENT, execution_sha256=digest,
                probe_sha256=probe_sha, config_sha256=sha(config_raw), prediction_source=config["predictions"],
                assigned_nvml_uuids=probe["gpu_identity_mapping"]["assigned_nvml_uuids"],
                devices=lanes, software=software, reuse_sources=[], adopted_pairs=0)


def native(arm, correction, reset, label):
    rewards, n = arm["native_rewards"], arm["controls"]
    success, terminated, truncated = (arm[k] for k in ("success", "terminated", "truncated"))
    check(canonical(arm["inferred"]) == canonical(correction) and arm["released_prompt"] == reset["prompt"],
          f"{label} applied correction or prompt")
    check(all(type(v) is bool for v in (success, terminated, truncated)) and terminated == success
          and truncated != terminated and type(n) is int and 0 < n <= reset["horizon"]
          and (not truncated or n == reset["horizon"]), f"{label} terminal flags or controls")
    check(isinstance(rewards, list) and len(rewards) == n and all(type(v) in (int, float) for v in rewards)
          and all(v == 0 for v in rewards[:-1]) and rewards[-1] == int(success)
          and type(arm["native_return"]) in (int, float) and arm["native_return"] == sum(rewards),
          f"{label} native rewards, return or success")
    return int(success)


def read_pair(path, row, ref, job, archived_success):
    obj = load(path)
    b = obj["pair"]
    lane = row["shard"] % 4  # live.sh task = 8 * panel position + shard, with four lanes.
    device = job["devices"][lane]
    check(sha(canonical(b)) == obj["sha256"] and canonical(b["row"]) == canonical(row)
          and b["contract_sha256"] == CONTRACT and b["execution_sha256"] == job["execution_sha256"]
          and type(b["lane"]) is int and b["lane"] == lane, f"{path} pair identity/digest")
    arms = b["arms"]
    check(set(arms) == set(READERS), f"{path} reader names")
    a, other = (arms[k] for k in READERS)
    reset, opening = a["reset"], a["opening_proposal_sha256"]
    check(canonical(reset) == canonical(other["reset"]) and set(reset) == set(ref)
          and all(canonical(reset[k]) == canonical(ref[k]) for k in STABLE)
          and HASH.fullmatch(reset["physics"]) and reset["physics"] != row["support_physics"]
          and reset["noise_seed"] == row["query"]["seed"] + 817117
          and reset["renderer_vendor"] == "NVIDIA Corporation" and reset["renderer"], f"{path} full reset")
    for field in ("public", "motor_frame0", "reader_frame0"):
        check(reset[field].keys() == ref[field].keys() and all(HASH.fullmatch(v) for v in reset[field].values()),
              f"{path} {field} observations")
    check(HASH.fullmatch(opening) and opening == other["opening_proposal_sha256"], f"{path} first proposal")
    out = {k: row[k] for k in (*KEY, *CASE)}
    out.update(shard=row["shard"], lane=lane, nvml_uuid=device["nvml_uuid"], cuda_uuid=device["cuda_uuid"],
               pci_bus_id=device["pci_bus_id"], visibility_token=device["visibility_token"], egl_device=device["egl_device"])
    out["trials"] = 1  # One matched trial, hence one episode for each reader.
    for reader, arm in arms.items():
        out[reader] = native(arm, row[reader], reset, f"{path} {reader}")
        out[reader + "_controls"] = arm["controls"]
        out[reader + "_return"] = arm["native_return"]
    out.update(archived_symmeta=archived_success, query_seed=row["query"]["seed"], physics=reset["physics"],
               noise_sha256=reset["noise_sha256"], reset_sha256=sha(canonical(reset)),
               opening_proposal_sha256=opening, pair_sha256=obj["sha256"], execution_sha256=job["execution_sha256"])
    out["condition_sha256"] = sha(canonical([row["tape_sha256"], row["query"], out["reset_sha256"], opening]))
    return out


def interval(task_deltas, n):
    # Two independent draws of four tasks enumerate all 8^8 ordered resamples.
    check(len(task_deltas) == 8, "expected eight tasks")
    half = Counter(map(sum, product(task_deltas, repeat=4)))
    mass = Counter()
    for a, na in half.items():
        for b, nb in half.items():
            mass[a + b] += na * nb
    cumulative = 0
    cuts = []
    for d, freq in sorted(mass.items()):
        cumulative += freq
        while len(cuts) < 2 and cumulative * 40 >= (1, 39)[len(cuts)] * 8**8:
            cuts.append(100 * d / n)
    return cuts


def describe(rows):
    n = len(rows)
    successes = {r: sum(x[r] for x in rows) for r in READERS}
    discord = Counter(x["tuned"] - x["symmeta"] for x in rows)
    tasks = defaultdict(list)
    for x in rows:
        tasks[x["suite"], x["task_id"]].append(x["tuned"] - x["symmeta"])
    check(len(tasks) == 8 and all(len(v) * 8 == n for v in tasks.values()), "unequal task weights")
    return dict(n=n, successes=successes, success_pct={r: 100 * c / n for r, c in successes.items()},
                wins=discord[1], losses=discord[-1], ties=discord[0],
                tuned_minus_symmeta_pp=100 * (discord[1] - discord[-1]) / n,
                task_cluster_ci95_pp=interval([sum(v) for v in tasks.values()], n))


def historical(rows):
    old = sum(x["archived_symmeta"] for x in rows)
    return dict(archived_symmeta_successes=old, fresh_minus_archived_successes=sum(x["symmeta"] for x in rows) - old,
                fresh_only=sum(x["symmeta"] > x["archived_symmeta"] for x in rows),
                archived_only=sum(x["symmeta"] < x["archived_symmeta"] for x in rows))


def panel_stats(rows):
    tasks = defaultdict(list)
    for r in rows:
        tasks[f"{r['suite']}/{r['task_id']}"].append(r)
    fingerprints = {k: {label: len({r[col] for r in group}) for label, col in
                        (("reset", "reset_sha256"), ("opening", "opening_proposal_sha256"),
                         ("execution", "execution_sha256"))} for k, group in tasks.items()}
    official = dict(missing_index_seed=[], task_distinct_fingerprints=fingerprints)
    extras = {}
    for split, group in [("overall", rows), ("ood", [r for r in rows if r["regime"] in OOD])]:
        pooled = describe(group)
        per_seed, seed_intervals, old = {}, {}, {}
        for seed in SEEDS:
            subset = [r for r in group if r["seed"] == seed]
            stats = describe(subset)
            seed_intervals[str(seed)] = stats.pop("task_cluster_ci95_pp")
            per_seed[str(seed)] = stats
            old[str(seed)] = historical(subset)
        rates = list(per_seed.values())
        sd = {r: stdev(s["success_pct"][r] for s in rates) for r in READERS}
        sd["tuned_minus_symmeta"] = stdev(s["tuned_minus_symmeta_pp"] for s in rates)
        official[split] = dict(pooled, per_seed=per_seed, seed_sd_pp=sd, complete=True)
        extras[split] = dict(per_seed_task_cluster_ci95_pp=seed_intervals,
                             historical_symmeta=dict(pooled=historical(group), per_seed=old))
    return official, extras


def compare(actual, expected, where="summary"):
    if isinstance(expected, dict):
        check(isinstance(actual, dict) and actual.keys() == expected.keys(), f"{where}: fields differ")
        for k, v in expected.items():
            compare(actual[k], v, f"{where}.{k}")
    else:
        check(canonical(actual) == canonical(expected), f"{where}: official={actual!r}, readback={expected!r}")


def write_outputs(out, rows, result):
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".readback-", dir=out.parent) as temp:
        stage = Path(temp) / "result"
        stage.mkdir()
        buf = io.StringIO()
        writer = csv.DictWriter(buf, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        packed = gzip.compress(buf.getvalue().encode(), mtime=0)
        (stage / "pairs.csv.gz").write_bytes(packed)
        result = dict(result, pairs_csv_gz_sha256=sha(packed))
        (stage / "readback.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stage.rename(out)  # Linux publishes both files together into a fresh or empty directory.


def run(root, predictions, out):
    check(not out.is_relative_to(root), "choose an output directory outside the native attempt")
    check(not out.exists() or (out.is_dir() and not any(out.iterdir())), "choose a fresh or empty output directory")
    cases = frozen_inputs(root, predictions)
    job = origin(root)
    expected = {pair_path(root, r) for r, _, _ in cases.values()}
    actual = {p for p in (root / "pairs").rglob("*") if p.suffix.lower() == ".json"}
    check(actual == expected, f"raw case coverage: missing={sorted(expected-actual)[:3]} extra={sorted(actual-expected)[:3]}")
    rows, panels, extras = [], {}, {}
    for panel in PANELS:
        batch = []
        for i, s in product(range(96), SEEDS):
            r, ref, old = cases[panel, i, s]
            batch.append(read_pair(pair_path(root, r), r, ref, job, old))
        panels[panel], extras[panel] = panel_stats(batch)
        for split, count in (("overall", 96), ("ood", 64)):
            check(all(v["n"] == count for v in panels[panel][split]["per_seed"].values()), f"{panel}/{split} trials")
        rows.extend(batch)
    counts = dict(contract_sha256=CONTRACT, expected_pairs=1152, completed_pairs=len(rows), complete=True, panels=panels)
    summary_path = root / "summary.json"
    summary_sha = sha(summary_path.read_bytes())
    compare(load(summary_path, summary_sha), counts)
    for p in PANELS:
        for split in extras[p]:
            panels[p][split].update(extras[p][split])
    result = dict(counts, predictions_sha256=PRED, historical_outcomes_sha256=HISTORY,
                  verified_summary_sha256=summary_sha, native_directory=str(root), execution=job, readback_sha256=sha(Path(__file__).read_bytes()),
                  comparison="tuned Perceiver minus same-seed fresh SymMeta; percentage points",
                  interval="exact 8-task paired percentile, inverse empirical CDF, reader seeds held fixed",
                  seed_sd="sample SD (ddof=1) of exact seed percentages and paired differences")
    write_outputs(out, rows, result)
    print(f"Verified {len(rows)} pairs, 768 OOD; all four panels match summary.json. Outputs: {out}")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("native", "predictions", "out"):
        cli.add_argument("--" + name, type=Path, required=True)
    args = cli.parse_args()
    try:
        run(args.native.resolve(), args.predictions.resolve(), args.out.resolve())
    except (ValueError, OSError, TypeError, KeyError, IndexError) as err:
        cli.exit(1, f"Readback failed: {err}\n")
