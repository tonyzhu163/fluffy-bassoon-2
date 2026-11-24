"""Join genuine post-view predictions to the registered native starts and archived nuisance factors."""
import gzip
import math
from pathlib import Path

import numpy as np

from protocol import PR7, IDENTITY, atomic, checked, digest, encode, parse, require
from paired_records import HASH


def asset(entry):
    path = Path(entry["path"])
    require(path.is_absolute() and HASH.fullmatch(entry["sha256"]), "Expected absolute path and SHA-256")
    return checked(path, entry["sha256"])


def rgb(z):
    frames = np.concatenate([z["frame0_img"][None], z["img_after"]])
    require(frames.dtype == np.uint8 and frames.shape == (4,64,64,3) and
            np.array_equal(frames[:-1],z["img_before"]), "Chronological 15-control support")
    return frames


def physical_view(row, case, name):
    require(row["archive_sha256"] == case["support_sha256"] and row["native_control_id"] == case["embodiment"]
            and row["suite"] == case["suite"] and row["task_id"] == case["task_id"]
            and row["init_state_id"] == 11 and row["seed"] == case["support_seed"]
            and row["frame_steps"] == [0,5,10,15] and row["control_yaw_deg"] == case["true_map"]["yaw_deg"]
            and row["xy_scale"] == case["true_map"]["xy_scale"] and row["gripper_sign"] == case["true_map"]["gripper_sign"]
            and HASH.fullmatch(row["physics"]), "Physical support identity")
    channels = ("issued_mean", "delta_eef", "delta_jaw", "delta_joint", "frame0_state", "valid", "chunk_steps")
    reference = {}
    for key in ("nominal", "orbit-", "orbit+"):
        v = row["views"][key]
        path = Path(v["passive_npz"])
        checked(path, v["sha256"])
        with np.load(path, allow_pickle=False) as z:
            frames = rgb(z)
            require(z["chunk_steps"].tolist() == [5,5,5] and z["valid"].all(), "Physical support duration")
            if key == "nominal":
                reference = {k:z[k] for k in channels}
            else:
                require(all(np.array_equal(z[k],reference[k]) for k in channels), "View changed trajectory")
            if key == name:
                observed = digest(frames.tobytes())
    with np.load(case["original_tape"], allow_pickle=False) as original:
        require(np.array_equal(reference["issued_mean"], original["issued_mean"]), "Incorrect replay controls")
    v = row["views"][name]
    h = np.asarray(v["H_ref_to_view"], dtype=np.float64)
    require(v["angle_deg"] == (0 if name=="nominal" else case["orbit_deg"]) and h.shape == (3,3)
            and np.isfinite(h).all() and np.linalg.matrix_rank(h) == 3, "Physical geometry")
    return v["sha256"], observed, digest(h.tobytes())


def digital_view(case, angle):
    import cv2
    path = Path(case["original_tape"])
    with np.load(path, allow_pickle=False) as z:
        raw = rgb(z)
    h = np.vstack([cv2.getRotationMatrix2D((31.5,31.5),angle,1.),[0.,0.,1.]])
    def warp(a, fill):
        return cv2.warpPerspective(a, h, (64,64), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(fill,)*3)
    frames = np.stack([warp(f,127) for f in raw])
    mask = warp(np.ones((64,64),np.float32),0) >= 1.
    frames[:,~mask] = 127
    return case["support_sha256"], digest(frames.tobytes()), digest(h.tobytes())


def load_table(receipt):
    table = parse(gzip.decompress(asset(receipt)))
    require(len(table) == receipt["rows"], "Score receipt row count")
    require(isinstance(table, list) and table, "Actual reader export list")
    index = {}
    for i,p in enumerate(table):
        if p["seed"] != 41 or p["calibration"] != "true" or p["model"] not in ("A","B","shipped"):
            continue
        key = (p["index"],p["view"],p["native_view_sha256"],p["model"])
        require(key not in index, f"Duplicate prediction {key}")
        index[key] = (i,p)
    return index


def select_bank(rows, bank):
    active = {}
    for r in bank["rows"]:
        if r["arm"] == "UR5e" and r["role"] == "heldout":
            require(r["index"] not in active, "Duplicate physical trajectory")
            if {"nominal","orbit+","orbit-"} <= r["views"].keys():
                active[r["index"]] = r
    expected = [{r["index"] for r in rows},
                {r["index"] for r in rows if r["slot"] >= 8},
                {r["index"] for r in rows if r["slot"] >= 8 and r["task_shard"] in (0,6)}]
    for ids in expected:
        if ids <= active.keys():
            return {i:active[i] for i in ids}
    raise ValueError("No complete registered physical rectangle")


def earned_rows(cases, physical, tables, seal, seal_dir, controls):
    rows = []
    checkpoints = {}
    for arm in ("A","B"):
        name = f"{arm}-seed41-step{seal['steps']}.msgpack"
        path = seal_dir/name
        sha = seal["checkpoint_sha256"][name]
        checked(path, sha)
        checkpoints[arm] = str(path),sha
    require(checkpoints["A"][1] != checkpoints["B"][1], "Distinct fitted reader artifacts required")
    shipped_name = "perceiver_cont_warm-passive_attempt1-seed41.msgpack"
    shipped_sha, = [line.split()[0] for line in (PR7/"evidence/train02pro/checkpoints.sha256").read_text().splitlines()
                    if Path(line.split()[1]).name == shipped_name]
    checkpoints["shipped"] = (str(seal_dir.parents[1]/PR7.name/"train02pro/checkpoints"/shipped_name), shipped_sha)
    checked(Path(checkpoints["shipped"][0]), shipped_sha)
    for c in cases:
        i = c["index"]
        real = i in physical
        study = "physical" if real else "digital"
        if study not in tables:
            continue
        angle = c["orbit_deg"] if real else c["digital_roll_deg"]
        view = ("orbit+" if angle>0 else "orbit-") if real else f"roll{angle:+d}"
        native_sha, obs_sha, h_sha = physical_view(physical[i], c, view) if real else digital_view(c, angle)
        table = tables[study]
        selected = {}
        for arm in ("A","B","shipped"):
            key = i,view,native_sha,arm
            if arm == "shipped" and key not in table:
                continue
            row_num, p = table[key]
            require(p["panel"] == study and p["camera_deg"] == angle and
                    all(p[k] == c[k] for k in ("suite","task_id","embodiment","regime")) and
                    (p["source_sha256"],p["observed_sha256"],p["calibration_sha256"]) ==
                    (c["support_sha256"],obs_sha,h_sha), f"Post-view join {key}")
            require(all(type(p[k]) in (float,int) and math.isfinite(p[k]) for k in ("yaw_deg","xy_scale"))
                    and p["xy_scale"] > 0 and type(p["gripper_sign"]) is int and p["gripper_sign"] in (-1,1)
                    and all(math.isclose(p[k], c["true_map"][j], rel_tol=1e-6,abs_tol=1e-6) for k,j in
                            (("true_yaw_deg","yaw_deg"),("true_gain","xy_scale"),("true_sign","gripper_sign"))),
                    f"Output decode or true map {key}")
            require((p["checkpoint_path"],p["checkpoint_sha256"]) == checkpoints[arm], "selected checkpoint")
            selected[arm] = dict(row_number=row_num, prediction=p)
        if "nominal" in controls:
            nominal_sha, nominal_obs, nominal_h = (physical_view(physical[i], c, "nominal") if real
                                                     else digital_view(c, 0))
            rank, p = table[i,"nominal",nominal_sha,"shipped"]
            require((p["checkpoint_path"],p["checkpoint_sha256"]) == checkpoints["shipped"]
                    and p["source_sha256"] == c["support_sha256"] and p["camera_deg"] == 0
                    and (p["observed_sha256"],p["calibration_sha256"]) == (nominal_obs,nominal_h)
                    and all(p[k] == c[k] for k in ("suite","task_id","embodiment","regime"))
                    and math.isfinite(p["yaw_deg"]), "Wrong nominal-view reference")
            selected["nominal"] = dict(row_number=rank, prediction=p)
        corrections = {arm:dict(c["nominal"], yaw_deg=item["prediction"]["yaw_deg"]) for arm,item in selected.items()}
        if "identity" in controls:
            corrections["identity"] = IDENTITY
        if "true_map" in controls:
            corrections["true_map"] = c["true_map"]
        rows.append(dict(c, study=study, angle_deg=angle, view=view, observed_sha256=obs_sha, calibration_sha256=h_sha,
                         physical_id=physical[i]["id"] if real else None,
                         historical_orbit=physical[i]["original_orbit"] if real else None,
                         calibration_physics=physical[i]["physics"] if real else c["support_physics"],
                         calibration_replay=physical[i]["replay"] if real else "original_tape",
                         native_view_sha256=native_sha, selected=selected, corrections=corrections))
    for study in {r["study"] for r in rows}:
        group = [r for r in rows if r["study"] == study]
        if not all("shipped" in r["corrections"] for r in group):
            for r in group:
                r["corrections"].pop("shipped", None)
    return rows


def assemble(cases, path, sha, out):
    raw = checked(path, sha)
    release = parse(raw)
    require(release["decision"] == "release_native" and release["seed"] == 41, "Parent native release")
    controls = release["optional_controls"]
    require(isinstance(controls,list) and len(set(controls))==len(controls) and
            set(controls) <= {"identity","true_map","nominal"}, "Registered controls only")
    for k in ("offline_physics_evidence","geometry_evidence"):
        asset(release[k])
    bank_rows, physical_scores = [], {}
    for entry in release["physical_receipts"]:
        receipt = parse(asset(entry))
        require(receipt["seal"] == release["training_seal"]["sha256"], "Physical receipt checkpoint seal")
        bank_rows.extend(parse(asset({"path":receipt["bank"], "sha256":receipt["bank_sha"]}))["rows"])
        scores = load_table(receipt)
        require(physical_scores.keys().isdisjoint(scores), "Supply disjoint physical banks or one union bank")
        for rank, p in scores.values():
            p["export_file_sha256"] = receipt["sha256"]
            require(p["panel"] == "physical" and p["bank"] == receipt["bank"], "Physical score bank")
        physical_scores.update(scores)
    bank = select_bank(cases, {"rows":bank_rows})
    tables = {"physical":physical_scores}
    if len(bank)<len(cases) and release.get("digital_receipt"):
        receipt = parse(asset(release["digital_receipt"]))
        require(receipt["seal"] == release["training_seal"]["sha256"] and receipt["bank"] is None,
                "Digital receipt checkpoint seal")
        tables["digital"] = load_table(receipt)
        for _,p in tables["digital"].values():
            p["export_file_sha256"] = receipt["sha256"]
    seal = parse(asset(release["training_seal"]))
    require(HASH.fullmatch(seal["training_exposures"]["41"]) and HASH.fullmatch(seal["source"])
            and seal["steps"] in (2500,5000), "Matched source-only fit receipt")
    rows = earned_rows(cases, bank, tables, seal, Path(release["training_seal"]["path"]).parent, controls)
    require(len(rows) == (len(cases) if "digital" in tables else len(bank)), "Coverage")
    atomic(out/"release.json", raw)
    return rows
