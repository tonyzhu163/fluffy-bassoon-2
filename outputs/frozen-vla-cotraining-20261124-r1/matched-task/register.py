"""Bind source examples to original Panda demonstrations and released BASE scene geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from xml.etree import ElementTree

import numpy as np

SUITES = ("libero_object", "libero_spatial")
ARCHIVE = {
    "libero_object": "42189d4415d4c51aaaf0708300653fccc39239cd3f2709079a713cd8d1678a8d",
    "libero_spatial": "ff6f26121653c77280eb40a38773a74141c11a8509f3466058cb56dd2cc60ead",
}
RELEASE = Path("/mowenjie/fluffy-bassoon-2/sources/libero-pro2-e5500c69")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def norm(value):
    return " ".join(value.lower().replace("_", " ").split())


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def original(inputs, out):
    import h5py
    for suite in SUITES:
        path = inputs / f"{suite}.hdf5"
        if sha(path) != ARCHIVE[suite]:
            raise ValueError(f"Incorrect original demonstration file {path}")
        demos, tapes = {}, {}
        with h5py.File(path) as f:
            for name, demo in f["data"].items():
                state = demo["states"][0]
                world = ElementTree.fromstring(demo.attrs["model_file"]).find("worldbody")
                nq = nv = 0
                free = {}
                for j in world.iter():
                    if j.tag not in ("joint", "freejoint"):
                        continue
                    kind = "free" if j.tag == "freejoint" else j.get("type", "hinge")
                    if kind == "free":
                        free[j.attrib["name"]] = state[1+nq:3+nq].tolist()
                    nq += {"free": 7, "ball": 4}.get(kind, 1)
                    nv += {"free": 6, "ball": 3}.get(kind, 1)
                if len(state) != 1 + nq + nv or not free:
                    raise ValueError(f"State/XML mismatch for {name}")
                acts = []
                for a in demo["actions"][:]:
                    # OpenVLA's published no-op filter uses the preceding retained gripper command.
                    if np.linalg.norm(a[:-1]) < 1e-4 and (not acts or a[-1] == acts[-1][-1]):
                        continue
                    acts.append(a)
                tapes[name] = np.asarray(acts, dtype="<f4")
                demos[name] = dict(free_joint_xy_m=free, retained_actions=len(acts))
        dump(out / f"{suite}-original.json", demos)
        np.savez_compressed(out / f"{suite}-actions.npz", **tapes)


def release(selection, out):
    from libero.libero import benchmark
    from libero.libero.envs.env_wrapper import ControlEnv
    from libero_remote.reproducibility import numerical_state_fingerprint

    candidates = read(selection)
    result = {}
    for suite in SUITES:
        source = min(norm(r["prompt"]) for r in candidates if r["suite"] == suite and
                     r["split"] == "train" and r["samples"] > 0)
        tasks = read(RELEASE / "assets/task_manifest.json")["tasks"]
        entry, = [t for t in tasks if t["suite"] == suite and t["case_id"] == "BASE" and
                  norm(Path(t["bddl_path"]).stem.removeprefix("BASE__")) == source]
        suite_obj = benchmark.get_benchmark_dict()[suite]()
        i = entry["base_task_id"]
        if norm(suite_obj.get_task(i).name) != source:
            raise ValueError("Release suite ordering changed")
        bank = suite_obj.get_task_init_states(i)
        bddl = RELEASE / "assets" / entry["bddl_path"]
        if sha(bddl) != entry["bddl_sha256"]:
            raise ValueError("BDDL bytes changed")
        env = ControlEnv(bddl_file_name=str(bddl), robots=["Panda"], use_camera_obs=False,
                         has_offscreen_renderer=False, initialization_noise=None)
        try:
            model = env.sim.model
            free = {n: int(model.jnt_qposadr[i]) for i, n in enumerate(model.joint_names)
                    if int(model.jnt_type[i]) == 0}
            rows = []
            for row in bank:
                row = np.asarray(row)
                if model.na or len(row) != 1 + model.nq + model.nv or not free:
                    raise ValueError("Release bank/model mismatch")
                rows.append(dict(fingerprint=numerical_state_fingerprint(row),
                                 xy={n: row[1+q:3+q].tolist() for n, q in free.items()}))
            prompt = norm(re.search(r"\(:language\s+([^)]*)\)", bddl.read_text(), re.I).group(1))
            result[suite] = dict(entry=entry, source=source, prompt=prompt, rows=rows)
        finally:
            env.close()
    dump(out / "release.json", result)


def contract(data, inputs, out):
    import h5py
    from sim_reset import POSITION_TOL_M
    manifest = read(data / "manifest.json")
    if manifest["robot"] != "Panda":
        raise ValueError("Expected Panda-only source")
    bank = read(inputs / "release.json")
    result = dict(data_manifest=sha(data / "manifest.json"),
                  selection_sha256=manifest["selection_sha256"], tasks={})
    for suite_no, suite in enumerate(SUITES):
        info = bank[suite]
        candidates = sorted({norm(r["prompt"]) for r in manifest["splits"]["train"]["trajectories"]
                             if r["suite"] == suite and r["t"]})
        prompt = candidates[0]
        if prompt != info["source"]:
            raise ValueError("Release lookup disagrees with staged source manifest")
        original = read(inputs / f"{suite}-original.json")
        evidence = []
        with np.load(inputs / f"{suite}-actions.npz", allow_pickle=False) as z:
            tapes = dict(z)
        for split in ("train", "heldout"):
            part = manifest["splits"][split]
            if sha(data / part["file"]) != part["sha256"]:
                raise ValueError("Unexpected source data bytes")
            offset = 0
            with h5py.File(data / part["file"]) as f:
                for r in part["trajectories"]:
                    n = len(r["t"])
                    if r["suite"] == suite and norm(r["prompt"]) == prompt and n:
                        actions = f["actions"][offset:offset+n]
                        matches = [k for k, tape in tapes.items() if len(tape) == r["steps"] and all(
                            np.array_equal(tape[t:t+10], a) for t, a in zip(r["t"], actions, strict=True))]
                        evidence.append(dict(id=r["id"], split=split, samples=n, times=r["t"], steps=r["steps"],
                                             staged_windows_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
                                             original_matches=matches))
                    offset += n
                if offset != len(f["actions"]):
                    raise ValueError("Manifest offset mismatch")
        if not evidence:
            raise ValueError("No source demonstrations")
        complete = all(len(r["original_matches"]) == 1 for r in evidence)
        eligible = sorted({k for r in evidence for k in r["original_matches"]}) if complete else sorted(original)
        # A named free object's planar position differs: the robot pose, time and velocities play no role.
        separations = {}
        rows = []
        for i in reversed(range(len(info["rows"]))):
            row = info["rows"][i]
            compared = {}
            for k in eligible:
                orig_xy = original[k]["free_joint_xy_m"]
                if orig_xy.keys() != row["xy"].keys():
                    raise ValueError("Published Panda and native model free-joint names disagree")
                distances = {j: float(np.linalg.norm(np.array(xy)-row["xy"][j])) for j, xy in orig_xy.items()}
                joint = max(distances, key=distances.get)
                compared[k] = dict(joint=joint, source_xy=orig_xy[joint], released_xy=row["xy"][joint],
                                   separation_m=distances[joint])
            separations[str(i)] = compared
            if all(v["separation_m"] > POSITION_TOL_M for v in compared.values()):
                rows.append(i)
                if len(rows) == 2:
                    break
        disjoint = len(rows) == 2
        if not disjoint:
            rows = [len(info["rows"])-1, len(info["rows"])-2]
        task_id = info["entry"]["base_task_id"]
        result["tasks"][suite] = dict(task_id=task_id, native_prompt=info["prompt"], source_prompt=prompt,
            available_train_prompts=candidates, original_hdf5_sha256=ARCHIVE[suite], trajectories=evidence,
            exact_original_demo_join=complete, compared_demo_ids=eligible, layout_disjoint=disjoint,
            comparison_tolerance_m=POSITION_TOL_M, compared_rows=separations, bddl=info["entry"],
            bank_length=len(info["rows"]), seed_base=42820000 + 10000*suite_no + 100*task_id,
            layouts={str(i):info["rows"][i]["fingerprint"] for i in sorted(rows)})
    dump(out, result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    for name, args in (("original", ("inputs", "out")), ("release", ("selection", "out")),
                       ("contract", ("data", "inputs", "out"))):
        sp = sub.add_parser(name)
        for arg in args:
            sp.add_argument(arg, type=Path)
    args = vars(p.parse_args())
    globals()[args.pop("mode")](**args)
