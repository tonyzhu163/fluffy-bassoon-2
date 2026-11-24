"""Admission checks for the offline fit. Reads manifests only, never source examples or optimizer state."""
import hashlib
import json
from pathlib import Path
from panel import MODES, TASKS, require


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect(fit_dir, fit_sha, manifest_path):
    import source_data
    import ur5e_motor
    import cotrain_policy
    import flow_head
    fit_path = fit_dir / "fit.json"
    require(sha(fit_path) == fit_sha, "offline fit receipt lock")
    report = json.loads(fit_path.read_text())
    config = json.loads((fit_dir / "config.json").read_text())
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    require(config_hash == report["contract"] and config["source_code"] == sha(source_data.__file__)
            and config["conversion_code"] == sha(ur5e_motor.__file__), "fit and runtime public input signature")
    for module in (cotrain_policy, flow_head):
        require(config["model_code"][Path(module.__file__).name] == sha(module.__file__), "live injection signature")
    data_hash = sha(manifest_path)
    require(data_hash == report["data_manifest"] == config["data_manifest_sha256"], "source manifest")
    manifest = json.loads(manifest_path.read_text())
    require(manifest["robot"] == "Panda" and manifest["history_shape"] == [52] and manifest["horizon"] == 10
            and set(manifest["splits"]) == {"train", "heldout"}, "registered source training interface")
    test_suites = {suite for suite, _ in TASKS}
    seen = set()
    splits = {}
    for name, part in manifest["splits"].items():
        trajectories = part["trajectories"]
        for v in trajectories:
            require(v["suite"] not in test_suites and v["sequence_sha256"] not in seen
                    and v["terminal_reward"] == 1 and v["is_terminal"], "source disjointness or supervision")
            seen.add(v["sequence_sha256"])
        splits[name] = dict(trajectories=len(trajectories), examples=part["examples"],
                            suites=sorted({v["suite"] for v in trajectories}))
    groups = report["parameters"]
    require(set(groups) == set(MODES[:4]) and report["reloaded_parity"]["max_abs"] == 0, "four fit cells and parity")
    kernels = {c: g["action_out_proj/kernel"] for c, g in groups.items()}
    require(kernels["original"] == kernels["adapter_only"] and
            all(kernels[c] != kernels["original"] for c in ("joint", "policy_only")), "effective trained kernels")
    return dict(fit_sha256=fit_sha, training_revision=report["code_revision"], config_sha256=config_hash,
                source_manifest_sha256=data_hash, splits=splits, parameters=groups)
