"""Recount registered native source panels and authenticate the immutable receipt bytes."""

import hashlib
import json
import math
import subprocess
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from large_transfer_stats import (
    ARMS,
    CONDITIONS,
    canonical,
    read,
    require,
    sha,
    sha_string,
)

HERE = Path(__file__).resolve().parent
NAMESPACE = "fluffy-bassoon-large-transfer-v1"
SUPPORT = tuple(range(8, 16))
PULSES = (48, 49)
CHECK = 39


class GateClosed(ValueError):
    pass


# Digest of the canonical implementation() map built from these Git tree bytes.
PRODUCER = "e2f794c41c9141a7aadaf67339da919336362f22"
PRODUCER_IMPLEMENTATION = (
    "0c0f00a185aac05c87f0dd7db48c5f6179eb747573d5e28b1b6a294251c10431"
)
REGISTERED = (
    ("sft01", 1600, "chunk_huber_v1"),
    ("sft02", 6400, "chunk_huber_v1"),
    ("sft03", 6400, "spatial_bins_v1"),
)


@lru_cache(maxsize=1)
def original_protocol():
    """Use the registered tape schema even when the checkout's producer writes a newer one."""
    import large_panel_protocol as current

    filename = f"outputs/{HERE.name}/large_panel_protocol.py"
    checksum = "a43cfe07b8b646fa2ec6f3f483be60f2bb4d77a825771a3b1bb5f2ef371682c2"
    if sha(Path(current.__file__)) == checksum:
        return current
    source = subprocess.check_output(
        ["git", "-C", str(HERE), "show", f"{PRODUCER}:{filename}"]
    )
    require(
        hashlib.sha256(source).hexdigest() == checksum,
        "Wrong registered producer reader",
    )
    module = ModuleType("registered_large_panel_protocol")
    # Only the hash-pinned repository source above is executable. Evidence is data.
    exec(compile(source, f"{PRODUCER}:{filename}", "exec"), module.__dict__)  # noqa: S102 -- verified Git blob
    return module


def protocol_for(data):
    import large_panel_protocol as current

    if current.digest(data["implementation"]) == PRODUCER_IMPLEMENTATION:
        require(
            data["revision"] in (PRODUCER, PRODUCER[:8]),
            "Different registered producer revision",
        )
        return original_protocol()
    require(
        data["revision"] not in (PRODUCER, PRODUCER[:8])
        and data["implementation"] == current.implementation(),
        "Unknown source producer implementation",
    )
    return current


def source_manifest(root, phase, prep_fit, previous=()):
    """Reconstruct the original producer's admission denylist from its preparation fit."""
    import numpy as np

    import generated_split as generated
    import large_panel_protocol as panel

    base = root / phase
    path = base / "manifest.json"
    data = read(path)
    manifest_sha = sha(path)
    start, n, minimum = panel.PHASES[phase]
    producer = protocol_for(data)
    # Always preserve the producer's own preflight and manifest checks.
    require(producer.manifest(root, phase) == data, "Changed producer manifest")
    require(
        read(path.with_suffix(".sha256.json")) == manifest_sha
        and (data["phase"], data["size"], data["required"], data["seed_range"])
        == (phase, n, minimum, [start, start + 9999])
        and all(data[k] == v for k, v in generated.TASK.items())
        and data["preparation_fit_sha256"] == sha(prep_fit / "manifest.json")
        and data["development_selection_sha256"]
        == (sha(root / "accepted/selection.json") if phase == "test" else None),
        "Incorrect registered panel or preparation fit",
    )
    _, fit_seeds, fit_hashes = panel.training(prep_fit)
    for key, expected in (("fitting_seeds", fit_seeds), ("fitting_hashes", fit_hashes)):
        if key in data or data["revision"] not in (PRODUCER, PRODUCER[:8]):
            require(
                data[key] == sorted(expected), "Preparation denylist differs from fit"
            )
    public = data["released"]
    require(
        [r["released_id"] for r in public] == list(range(50))
        and data["released_bank_sha256"] == generated.BANK_SHA,
        "Different released bank",
    )
    generated.public_denylist([r["initial_state_sha256"] for r in public])
    fields = panel.GEOMETRY_FIELDS
    seen = {
        k: fit_hashes | {r[k] for r in (*public, *previous) if k in r} for k in fields
    }
    rows, rejected = data["layouts"], data["rejected"]
    seeds = [r["layout_seed"] for r in rows]
    skips = {r["layout_seed"]: r["matching_hashes"] for r in rejected}
    require(
        len(seeds) == n
        and seeds == sorted(set(seeds))
        and start <= seeds[0] <= seeds[-1] < start + 10000
        and len(skips) == len(rejected)
        and sorted([*seeds, *skips]) == list(range(start, seeds[-1] + 1)),
        "Incomplete registered draw sequence",
    )
    by_seed = dict(zip(seeds, rows, strict=True))
    for seed in range(start, seeds[-1] + 1):
        if seed in skips:
            matched = skips[seed]
            require(
                matched and all(k in seen and v in seen[k] for k, v in matched.items()),
                "Unsubstantiated layout rejection",
            )
            continue
        row = by_seed[seed]
        require(
            row["split"] == phase
            and row["layout_id"] == 1_000_000 + seed
            and row["state_file"] == f"rows/{seed}.npy"
            and sha(base / row["state_file"]) == row["file_sha256"],
            "Changed registered state file",
        )
        # The sampler originally publishes train origins; preparation changes only the split.
        generated.validate_origin({**row, "split": "train"})
        generated.expect(
            row,
            {
                "source_pin": "e5500c690c1498eca1b7050350654964190cdf70",
                "manifest_sha256": "71a6e4e25a2a4a162844d77136e7bff8e6f69c839d2d8194620c4afeabde2bba",
                "bddl_sha256": "c2db20071f58fd43fe029e5bf9176f3c7371ee62cadc311f2b8682c089174c0d",
            },
        )
        state = np.load(base / row["state_file"], allow_pickle=False)
        require(
            state.shape == (79,)
            and state.dtype == np.float64
            and np.isfinite(state).all()
            and hashlib.sha256(
                b'{"dtype":"<f8","shape":[79]}\0'
                + state.astype("<f8", copy=False).tobytes()
            ).hexdigest()
            == row["initial_state_sha256"],
            "Native donor state fingerprint differs",
        )
        free_xy = {
            k: j["qpos"][:2]
            for k, j in row["sampled_scene"]["joints"].items()
            if j["type"] == 0
        }
        require(
            len(free_xy) == 4
            and row["mobile_joint_xy_hash"] == panel.digest(free_xy)
            and set(row["preflight"]) == set(ARMS)
            and all(
                (
                    r["scene_fingerprint"],
                    r["generated_row_fingerprint"],
                    r["seed"],
                    r["init_state_id"],
                )
                == (
                    row["scene_hash"],
                    row["initial_state_sha256"],
                    seed,
                    row["layout_id"],
                )
                and all(
                    sha_string(r[k])
                    for k in (
                        "post_settle_scene_fingerprint",
                        "initial_state_fingerprint",
                    )
                )
                for r in row["preflight"].values()
            )
            and 0 <= row["source_pair_errors"]["max_position_m"] <= 0.001
            and 0 <= row["source_pair_errors"]["max_rotation_rad"] <= 0.01,
            "Invalid paired native admission",
        )
        for key in fields:
            require(row[key] not in seen[key], f"Duplicate {key} at seed {seed}")
            seen[key].add(row[key])
    return data, manifest_sha


def check_phase(root, phase, candidate, data, manifest_sha):
    panel = protocol_for(data)
    stride = 5
    contract = dict(
        **candidate,
        phase=phase,
        stride=stride,
        panel_sha256=manifest_sha,
        evaluator=data["implementation"],
        horizon=300,
        inference_batch_size=8,
    )
    directory = panel.directory(root, phase, candidate, stride)
    result = panel.report_for(
        data, contract, directory
    )  # Reopens every native tape, then recounts rewards.
    require(
        read(directory / "report.json") == result,
        f"{phase} report differs from native tapes",
    )
    n, minimum = (50, 38) if phase == "dev" else (100, 75)
    require(
        result["required_per_arm"] == minimum
        and set(result["arms"]) == set(ARMS)
        and len(data["layouts"]) == n,
        f"Incorrect full {phase} contract",
    )
    layouts = []
    for row in data["layouts"]:
        entry = {
            "seed": row["layout_seed"],
            "row": row["initial_state_sha256"],
            "xy": row["movable_xy_hash"],
            "joint_xy": row["mobile_joint_xy_hash"],
            "scene": row["scene_hash"],
            "state_file": row["state_file"],
            "file_sha256": row["file_sha256"],
        }
        require(
            all(
                sha_string(entry[key]) for key in ("row", "xy", "scene", "file_sha256")
            ),
            "Invalid physical layout hashes",
        )
        layouts.append(entry)
    for field in ("seed", "row", "xy", "scene"):
        require(
            len({r[field] for r in layouts}) == n,
            "Panel lacks distinct physical layouts",
        )
    counts, tapes = {}, {}
    for body in ARMS:
        arm = result["arms"][body]
        rows = arm["episodes"]
        require(
            arm["total"] == n
            and len(rows) == n
            and all(
                (r["arm"], r["layout_seed"], r["scene_hash"], r["movable_xy_hash"])
                == (body, p["seed"], p["scene"], p["xy"])
                for r, p in zip(rows, layouts, strict=True)
            ),
            "Source arms evaluated different ordered physical panels",
        )
        counts[body] = sum(r["success"] for r in rows)
        require(
            counts[body] == arm["successes"], "Source native success recount differs"
        )
        tapes[body] = [r["tape_sha256"] for r in rows]
    require(
        result["passed"] is (min(counts.values()) >= minimum),
        "Incorrect per-arm decision",
    )
    return {
        "planned_per_arm": n,
        "minimum_per_arm": minimum,
        "native_successes": counts,
        "tapes": tapes,
        "manifest_sha256": manifest_sha,
        "producer_revision": data["revision"],
        "producer_implementation_sha256": panel.digest(data["implementation"]),
        "contract_sha256": sha(directory / "contract.json"),
        "report_sha256": sha(directory / "report.json"),
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "layouts": layouts,
    }


def recount_sources(root, fit):
    """Read all 300 development and 200 confirmation tapes before releasing the winner."""
    import large_panel_protocol as panel

    producer = protocol_for(read(root / "dev/manifest.json"))
    seal = producer.selection(root)
    comparison = read(root / "dev/comparison.json")
    rows = comparison["candidates"]
    require(
        len(rows) == len(REGISTERED) and seal["stride"] == 5,
        "Expected the three registered candidates at stride five",
    )
    # e2f794c4 did not serialize fit paths. The three published fits are siblings.
    campaign = Path(fit).resolve().parent.parent
    fits = [campaign / name / "fit" for name, _, _ in REGISTERED]
    data, manifest_sha = source_manifest(root, "dev", fits[0])
    reports, candidates, extended_inputs = [], [], []
    for (name, step, arch), path, row in zip(REGISTERED, fits, rows, strict=True):
        c, _ = panel.candidate(path, step)
        require(
            c["architecture"] == arch
            and (step != 1600 or c["checkpoint_sha256"] == panel.SFT01)
            and all(
                row[k] == c[k]
                for k in ("architecture", "step", "checkpoint_sha256", "fit_id")
            )
            and ("fit" not in row or Path(row["fit"]).parts[-2:] == (name, "fit")),
            "Development comparison substitutes a registered candidate",
        )
        training, seeds, hashes = panel.training(path)
        if step == 6400:
            require(
                training["recipe"]["total_steps"] == step,
                "Wrong extended optimization budget",
            )
            extended_inputs.append(
                (training["collector_index_sha256"], training["seed"])
            )
        require(
            not any(
                r["layout_seed"] in seeds
                or any(r[k] in hashes for k in panel.GEOMETRY_FIELDS)
                for r in data["layouts"]
            ),
            "Development exposure in source fit",
        )
        report = check_phase(root, "dev", c, data, manifest_sha)
        require(
            row["counts"] == report["native_successes"]
            and row["report_sha256"] == report["report_sha256"],
            "Development comparison differs from native tapes",
        )
        reports.append(report)
        candidates.append(c)
    require(
        extended_inputs[0] == extended_inputs[1],
        "Extended fits used different source banks or seeds",
    )
    counts = [r["native_successes"].values() for r in reports]
    winner = max(range(len(counts)), key=lambda i: (min(counts[i]), sum(counts[i]), -i))
    c, dev = candidates[winner], reports[winner]
    require(
        type(comparison["winner"]) is int
        and winner == comparison["winner"]
        and comparison["passed"] is True
        and sha(Path(fit) / "manifest.json") == c["fit_id"]
        and seal
        == dict(
            **c,
            phase="dev",
            stride=5,
            panel_sha256=manifest_sha,
            evaluator=data["implementation"],
            horizon=300,
            inference_batch_size=8,
            report_sha256=dev["report_sha256"],
            comparison_sha256=sha(root / "dev/comparison.json"),
        ),
        "Selection differs from registered development ranking or accepted contract",
    )
    if min(counts[winner]) < 38:
        raise GateClosed(
            f"Development requires 38/50 per arm; observed {dev['native_successes']}"
        )
    test_data, test_sha = source_manifest(root, "test", fits[winner], data["layouts"])
    require(
        test_data["implementation"] == seal["evaluator"], "Sealed evaluator changed"
    )
    test = check_phase(root, "test", c, test_data, test_sha)
    if min(test["native_successes"].values()) < 75:
        raise GateClosed(
            f"Confirmation requires 75/100 per arm; observed {test['native_successes']}"
        )
    return seal, dev, test


def receipt(args):
    import large_panel_protocol as panel
    from large_transfer_fit import source_code

    seal, dev, test = recount_sources(args.root, args.fit)
    stride, step = seal["stride"], seal["step"]
    joint = read(args.joint / "report.json")
    digest = seal["checkpoint_sha256"]
    stage1 = joint["stage1"]
    directory = Path(stage1["directory"])
    require(
        sha(directory / "report.json") == stage1["report_sha256"]
        and sha(directory / "reader-stage1.pt") == stage1["checkpoint_sha256"]
        and all(
            sha(directory / f"practice-{arm}-{pose}.npz")
            == stage1["tape_sha256"][f"practice-{arm}-{pose}.npz"]
            for arm in ARMS
            for pose in (20, 21, 22, 23, 28)
        ),
        "Changed source motion pretraining evidence",
    )
    require(
        digest
        == dev["checkpoint_sha256"]
        == test["checkpoint_sha256"]
        == joint["motor"]["motor_sha256"]
        and joint["status"] == "complete"
        and joint["stage"] == "large-source-joint-v1"
        and joint["motor"]["fit_id"] == seal["fit_id"]
        and joint["motor"]["step"] == step
        and joint["motor"]["prompt"] == seal["prompt"]
        and joint["source_bodies"] == list(ARMS)
        and joint["joint"]["optimizer"] == "Adam"
        and [row["step"] for row in joint["joint"]["curve"]] == list(range(1, 151))
        and joint["joint"]["steps"] == 150
        and joint["joint"]["per_body_per_update"] == 16
        and all(
            math.isfinite(v) and v > 0
            for v in (
                joint["joint"]["parameter_change_l2"][k] for k in ("reader", "adapter")
            )
        )
        and set(joint["stage1"]["tape_sha256"])
        == {
            f"practice-{arm}-{pose}.npz"
            for arm in ARMS
            for pose in (20, 21, 22, 23, 28)
        }
        and joint["code_sha256"] == source_code()
        and joint["checkpoint_sha256"] == sha(args.joint / "reader-adapter.pt"),
        "Joint fit and source panels must use identical frozen checkpoint bytes",
    )
    return {
        "protocol": NAMESPACE,
        "source_development": dev,
        "source_confirmation": test,
        "selection_sha256": sha(args.root / "accepted/selection.json"),
        "step": step,
        "stride": stride,
        "fit_id": seal["fit_id"],
        "motor_sha256": digest,
        "fit_report_sha256": sha(args.joint / "report.json"),
        "reader_sha256": joint["checkpoint_sha256"],
        "prompt": seal["prompt"],
        "source_data": {
            "stage1_report": joint["stage1"]["report_sha256"],
            "stage1_checkpoint": joint["stage1"]["checkpoint_sha256"],
            "stage1_tapes": dict(sorted(joint["stage1"]["tape_sha256"].items())),
            "expert_indices": joint["expert_data"]["indices"],
            "expert_coverage": joint["expert_data"]["coverage"],
            "joint_updates": joint["joint"]["steps"],
            "parameter_change_l2": joint["joint"]["parameter_change_l2"],
        },
        "development_comparison_sha256": sha(args.root / "dev/comparison.json"),
        "evaluator": {
            **panel.implementation(),
            **source_code(),
            "transfer.py": sha(HERE / "transfer.py"),
            **{f.name: sha(f) for f in sorted(HERE.glob("large_transfer*.py"))},
        },
        "target": "UR5e",
        "conditions": list(CONDITIONS),
        "layouts": test["layouts"],
        "supports": [SUPPORT[i % len(SUPPORT)] for i in range(100)],
        "donors": [ARMS[(i + i // len(SUPPORT)) % 2] for i in range(100)],
        "calibration_poses": list(PULSES),
        "calibration_check": CHECK,
        "admission": "all 100 target resets before queries; physical rejection applies to all five conditions",
    }


def authenticate(args):
    payload = Path(args.receipt).read_bytes()
    expected = getattr(args, "receipt_sha256", None)
    if not sha_string(expected) or hashlib.sha256(payload).hexdigest() != expected:
        raise GateClosed("Receipt bytes differ from the required SHA-256 pin")
    plan = json.loads(payload)
    if canonical(plan) != canonical(receipt(args)):
        raise GateClosed(
            "Pinned receipt differs from the current native source recount"
        )
    return plan
