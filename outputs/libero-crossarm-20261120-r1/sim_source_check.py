"""Read source and MJCF assets without importing the simulator or deserializing state archives."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickletools
import re
import subprocess
import xml.etree.ElementTree as ET
from zipfile import ZipFile

WHEEL = "aba065e7b36745738cede259457b2cb349427f3608728d867ef3a2034cb62994"
HERE = Path(__file__).resolve().parent


def literals(path):
    return {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse(path.read_bytes()).body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"SOURCE_PIN", "ARM_XML_SHA", "GRIPPER_XML_SHA", "MANIFEST_SHA"}}


def check(wheel, release):
    pin = literals(HERE / "arm_env.py")
    revision = subprocess.check_output(["git", "-C", str(release), "rev-parse", "HEAD"], text=True).strip()
    if revision != pin["SOURCE_PIN"] or hashlib.sha256(wheel.read_bytes()).hexdigest() != WHEEL:
        raise ValueError("Source revision or robosuite wheel differs from the pinned sources")
    report = dict(libero=revision, robosuite_whl=WHEEL, arms={})
    with ZipFile(wheel) as archive:
        stock = ast.parse(archive.read("robosuite/robots/__init__.py"))
        maps = next(n.value for n in stock.body if isinstance(n, ast.Assign)
                    and getattr(n.targets[0], "id", "") == "ROBOT_CLASS_MAPPING")
        registry = {k.value: v.id for k, v in zip(maps.keys, maps.values)}
        config = json.loads(archive.read("robosuite/controllers/config/osc_pose.json"))
        assert config["type"] == "OSC_POSE" and config["control_delta"] and config["impedance_mode"] == "fixed"
        assert config["input_min"] == -1 and config["input_max"] == 1
        assert config["output_max"] == [0.05] * 3 + [0.5] * 3
        assert config["output_min"] == [-v for v in config["output_max"]]
        for arm, digest in pin["ARM_XML_SHA"].items():
            root = Path(f"robosuite/models/assets/robots/{arm.lower()}")
            raw = archive.read(str(root / "robot.xml"))
            assert hashlib.sha256(raw).hexdigest() == digest and registry[arm] == "SingleArm"
            tree = ET.fromstring(raw)
            bodies = tree.findall("worldbody//body")
            joints = [(body.get("name"), j.get("name")) for body in bodies for j in body.findall("joint")]
            assert [joint for _, joint in joints] == [act.get("joint") for act in tree.findall("actuator/*")]
            assert len(joints) == (6 if arm == "UR5e" else 7)
            camera = tree.find(".//body[@name='right_hand']/camera[@name='eye_in_hand']")
            assert camera is not None and camera.get("mode") == "fixed" and camera.get("fovy") == "75"
            assert camera.get("pos") == "0.05 0 0" and camera.get("quat") == "0 0.707108 0.707108 0"
            meshes = tree.findall("asset/mesh")
            mesh_refs = {geom.get("mesh") for geom in tree.findall("worldbody//geom[@mesh]")}
            assert mesh_refs and mesh_refs <= {mesh.get("name") for mesh in meshes}
            for asset in tree.findall("asset/*[@file]"):
                # Read whole referenced files to check ZIP CRC and catch missing payloads.
                assert len(archive.read(str(root / asset.get("file")))) > 0
            report["arms"][arm] = dict(joints=joints, mjcf_meshes=len(meshes), used_meshes=len(mesh_refs),
                                      robot_xml_sha256=digest)
        grip_root = Path("robosuite/models/assets/grippers")
        raw = archive.read(str(grip_root / "panda_gripper.xml"))
        assert hashlib.sha256(raw).hexdigest() == pin["GRIPPER_XML_SHA"]
        grip = ET.fromstring(raw)
        assert [e.get("type") for e in grip.findall("worldbody//joint")] == ["slide", "slide"]
        assert [e.get("ctrlrange") for e in grip.findall("actuator/position")] == ["0.0 0.04", "-0.04 0.0"]
        assert grip.find(".//body[@name='eef']/site[@name='grip_site']") is not None
        for asset in grip.findall("asset/*[@file]"):
            assert len(archive.read(str(grip_root / asset.get("file")))) > 0
    root = release / "assets"
    raw = (root / "task_manifest.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == pin["MANIFEST_SHA"]
    problems = Counter()
    report["pilot_state_archive_shapes"] = {}
    for entry in json.loads(raw)["tasks"]:
        if entry["case_id"] != "BASE":
            continue
        bddl = root / entry["bddl_path"]
        contents = bddl.read_bytes()
        assert hashlib.sha256(contents).hexdigest() == entry["bddl_sha256"]
        problem = re.search(rb"\(problem\s+(\w+)\)", contents, re.I).group(1).decode().lower()
        problems[problem] += 1
        key = entry["suite"], entry["base_task_id"]
        if key in {("libero_goal", 4), ("libero_goal", 5), ("libero_goal", 1), ("libero_spatial", 1)}:
            name = bddl.stem.removeprefix("BASE__") + ".pruned_init"
            state = release / "simulator-source/libero/libero/init_files" / key[0] / name
            with ZipFile(state) as data:
                ops = [(op.name, value) for op, value, _ in pickletools.genops(data.read("archive/data.pkl"))]
            # NumPy ndarray pickle encodes the (rows, flat_state_size) shape ahead of dtype.
            shape_end = next(i for i in range(2, len(ops)) if ops[i][0] == "TUPLE2"
                             and all(op.startswith("BININT") for op, _ in ops[i-2:i]))
            rows, cols = [v for _, v in ops[shape_end-2:shape_end]]
            assert ("BINUNICODE", "f8") in ops and rows > 0 and cols > 1
            assert any(op == "BINUNICODE" and len(value) == rows * cols * 8 for op, value in ops)
            report["pilot_state_archive_shapes"][f"{key[0]}/{key[1]}"] = [rows, cols]
    assert sum(problems.values()) == 40 and len(report["pilot_state_archive_shapes"]) == 4
    report["base_bddl_problem_counts"] = dict(problems)
    report["syntax_checked"] = []
    for name in ("arm_env.py", "sim_reset.py", "sim_source_check.py", "sim_panda_parity.py"):
        compile((HERE / name).read_bytes(), name, "exec")
        report["syntax_checked"].append(name)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("release", type=Path)
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError("Source checks require Python assertions")
    print(json.dumps(check(args.wheel, args.release), indent=2))
