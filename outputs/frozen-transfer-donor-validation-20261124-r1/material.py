"""Read the recorded controls using each original writer's schema and published checksums."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEV = ROOT / "frozen-transfer-donor-20261124-r1"
OLD = "frozen-vla-native-transfer-20261123-r1"
TRACK = "frozen-vla-tracking-feedback-20261123-r1"
PROG = "frozen-transfer-progress-20261124-r1"
ARCHIVE = HERE / "inputs/target-tapes.tar.gz"
digest = runpy.run_path(str(DEV / "readback.py"))["digest"]
IDENTITY = dict(yaw_deg=0., xy_scale=1., gripper_sign=1)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def read():
    old = json.loads((ROOT/OLD/"gpu-evidence.json").read_bytes())
    track = json.loads((ROOT/TRACK/"native-evidence.json").read_bytes())
    origin = json.loads((ROOT/PROG/"evidence/native01/Kinova3/origin.json").read_bytes())
    prog_archive = ROOT/PROG/"evidence/native01/episodes.tar.gz"
    assert sha(prog_archive.read_bytes()) == "e8fbb476e911cb965a4d467607b1ef3cadc1466af9558fad7a5c79d94ed5dc6a"
    records = {}
    manifest = {}
    with tarfile.open(ARCHIVE) as archive, tarfile.open(prog_archive) as prog:
        for row, mode in [(14,"raw"), (15,"raw"), (16,"raw"), (17,"raw"), (31,"raw"), (31,"tuned")]:
            if row == 31:
                relative = f"gpu01-Kinova3/episodes/libero_goal-3-{mode}.json"
                name = f"{PROG}/{relative}"
                expected = sha(prog.extractfile(relative).read())
                host, pin = origin["host"], origin["pin"]
            else:
                study, pub, suffix, hashes = ((OLD, old, "Kinova3", "sha256") if row < 16 else
                                               (TRACK, track, "raw", "episode_sha256"))
                file = f"libero_goal-3-{row}-{suffix}.json"
                name = f"{study}/gpu01/{file}"
                expected = pub[hashes][file]
                host, pin = [v.split("=")[1] for v in pub["launch"].split()]
            blob = archive.extractfile(name).read()
            assert sha(blob) == expected
            x = json.loads(blob)
            if row == 31:
                q, rec, env, key = x["case"]["query"], x["native"], x["environment"], "sent"
                assert x["origin"] == origin and x["mode"] == mode and not x["fixture"]
                assert rec["inferred"] == (IDENTITY if mode == "raw" else x["case"]["support"]["tuned"])
            else:
                q, rec, env, key = x["case"], x["record"], x["full_reset"], "cmd"
                assert rec["inferred"] == IDENTITY and x.get("mode", "raw") == mode
                assert x.get("corrections") is None
            reset = rec["reset"]
            assert q == dict(arm="Kinova3", suite="libero_goal", task_id=3, init_state_id=row, seed=41810300+row, variant="BASE")
            assert all(env[k] == q[k] for k in ("suite","task_id","init_state_id","seed"))
            assert env["initial_state_fingerprint"] == reset["physics"] and env["scene_fingerprint"] == reset["scene_fingerprint"]
            assert reset["horizon"] == 300 and reset["worker_step_index"] == 0 and reset["noise_seed"] == q["seed"]+817117
            assert reset["renderer_vendor"] == "NVIDIA Corporation" and reset["arm"] == x["model"]["arm"] == "Kinova3"
            assert reset["action_schema"] == dict(shape=[7], dtype="float32", controller="OSC_POSE", control_mode="relative",
                                                  control_freq=20, low=[-1.]*7, high=[1.]*7)
            commands = [action for c in x["chunks"] for action in c[key]]
            n = len(commands)
            assert all(len(c[key]) == min(5, n-i*5) for i,c in enumerate(x["chunks"]))
            assert 0 < n == rec["controls"] <= 300 and all(len(a)==7 and all(-1<=v<=1 for v in a) for a in commands)
            assert rec["native_rewards"] == [0.]*(n-1) + [float(rec["success"])]
            assert rec["native_return"] == int(rec["success"]) and rec["terminated"] == rec["success"]
            assert rec["truncated"] == (n == 300 and not rec["success"]) and rec["truncated"] != rec["terminated"]
            manifest[name] = dict(pin=pin, host=host, sha256=expected, actions_field="chunks[]."+key,
                                  action_sha256=digest(commands), controls=n, success=rec["success"], seed=q["seed"],
                                  noise_sha256=reset["noise_sha256"], row_sha256=env["released_row_fingerprint"])
            records[row,mode] = (x, rec, env, commands)
        assert len(archive.getmembers()) == len(records)
    return records, dict(archive_sha256=sha(ARCHIVE.read_bytes()), members=manifest)


if __name__ == "__main__":
    _, manifest = read()
    p = HERE/"inputs/manifest.json"
    if sys.argv[1:] == ["--write"]:
        p.write_text(json.dumps(manifest, indent=2)+"\n")
    else:
        assert json.loads(p.read_bytes()) == manifest
    print(f"Verified {len(manifest['members'])} published target episodes.")
