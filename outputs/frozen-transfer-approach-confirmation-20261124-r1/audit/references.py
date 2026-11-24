"""Read source evidence from immutable native release files and the two original recording archives."""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import numpy as np

ROOT = Path("/mowenjie/fluffy-bassoon-2")
RUN = "frozen-transfer-approach-confirmation-20261124-r1"
BASE = "ef165714d626305fa756e10fac2b5f7c45446f1c"
RELEASE = "e5500c690c1498eca1b7050350654964190cdf70"
NATIVE = "a8a997e7d06bb131d3f9710204c58e106f0f0904"
MANIFEST = "71a6e4e25a2a4a162844d77136e7bff8e6f69c839d2d8194620c4afeabde2bba"
PRED = "c8420c2cb7d0cb38e6f0d73b3a20ce47997dbef7657b652d45eaaa8988eb6c0b"
WEIGHTS = "ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170"
IDENTITY = {"yaw_deg": 0., "xy_scale": 1., "gripper_sign": 1}
ROWS = range(20, 26)
RELEASE_DIR = ROOT/"sources/libero-pro2-e5500c69"
T = {
    "drawer": ("libero_goal", 3, "open the top layer of the drawer and put the bowl inside", 300, 42710300,
               "frozen-transfer-cue-20261124-r1/evidence/native01/native-tapes.tar.gz",
               "637383c46eabaf3f8aaf776a9534e84aef152a277ab3ca9ac191c5a68c94fd5c",
               "gpu01/episodes/IIWA-34-raw.json", "a610024dc57b1f8783d3dfe000454414db48b1915f858f3759abb9d700b754c9",
               "open_the_top_drawer_and_put_the_bowl_inside", 79,
               "1cd5354ade9ab9f323a518515f961df96123b7daeca017df1f62d3e9ff18794f"),
    "moka": ("libero_10", 8, "put both moka pots on the stove", 520, 42700800,
             "frozen-transfer-progress-20261124-r1/evidence/native01/episodes.tar.gz",
             "e8fbb476e911cb965a4d467607b1ef3cadc1466af9558fad7a5c79d94ed5dc6a",
             "gpu01-IIWA/episodes/libero_10-8-raw.json", "b164c6b2c64a9f8a2a0c1edcb529f7ea2094ea28fd2a242d0258005e31054ad6",
             "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove", 47,
             "44f992e911c84a1340589510ecdaf947a4872966f7678c8364d8bb5d13dc4dac"),
}


def check(ok, label):
    if not ok:
        raise ValueError(label)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fp(array):
    a = np.ascontiguousarray(array)
    return sha(repr((a.shape, a.dtype.str)).encode() + a.tobytes())


def read(path):
    return json.loads(path.read_bytes())


def checked_bytes(path, h):
    data = path.read_bytes()
    check(sha(data) == h, path)
    return data


def checkout(pin):
    path = ROOT/"setups/symmeta-perceiver-followups-20261123-r1"/("src-" + pin[:12])
    check(subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip() == pin, "checkout pin")
    check(subprocess.run(["git", "-C", str(path), "symbolic-ref", "-q", "HEAD"], capture_output=True).returncode == 1, "detached")
    return path


def references(repo):
    import torch

    outputs = repo/"outputs"
    released = json.loads(checked_bytes(RELEASE_DIR/"assets/task_manifest.json", MANIFEST))
    pred = json.loads(checked_bytes(outputs/"symmeta-perceiver-retry-20261123-r1/evidence/train02pro/predictions.json", PRED))
    archive = outputs/"symmeta-fk-crossarm-20261123-r1/evidence/gpu06kinova3-cases.json.gz"
    support_cases = json.loads(gzip.decompress(checked_bytes(archive, "74587eaaf87ce69d733083b068f27b46fc62619c3a7ffe2ffe8248816949783e")))
    result, tapes = {}, {}
    for task, (suite, tid, prompt, horizon, seed_base, arc, ar_hash, name, member_hash, bank_name, width, bank_hash) in T.items():
        entry, = (x for x in released["tasks"] if (x["suite"],x["base_task_id"],x["case_id"]) == (suite, tid, "BASE"))
        bddl = checked_bytes(RELEASE_DIR/"assets"/entry["bddl_path"], entry["bddl_sha256"]).decode()
        goal = bddl.split("(:goal")[1].split("(:perturbation_config")[0].strip()
        check(entry["max_steps"] == horizon and entry["case_kind"] == "base", task+" manifest")
        bank = RELEASE_DIR/"simulator-source/libero/libero/init_files"/suite/(bank_name+".pruned_init")
        checked_bytes(bank, bank_hash)
        data = torch.load(bank, map_location="cpu", weights_only=False)
        check(data.shape == (50,width), task+" released bank axes")
        def row_hash(r):
            arr = np.ascontiguousarray(data[r], dtype="<f8")
            header = json.dumps({"dtype": "<f8", "shape": [width]}, separators=(",", ":")).encode()
            return sha(header + b"\0" + arr.tobytes())
        row_hashes = {r: row_hash(r) for r in ROWS}
        check(len(set(row_hashes.values())) == 6, task+" distinct nominal released rows")
        checked_bytes(outputs/arc, ar_hash)
        with tarfile.open(outputs/arc) as src:
            raw = src.extractfile(name).read()
            check(sha(raw) == member_hash, task+" source member")
            record = json.loads(raw)
            if task == "drawer":
                pool = {}
                for r in range(34, 38):
                    original = json.loads(src.extractfile(f"gpu01/episodes/IIWA-{r}-raw.json").read())
                    check(original["native"]["success"] and original["native"]["native_return"] == 1, "source pool")
                    pool[r] = original["native"]["controls"]
                check(min(pool, key=pool.get) == 34, "source-only drawer choice")
                derivative = outputs/"frozen-transfer-donor-20261124-r1/inputs/raw-tapes.tar.gz"
                checked_bytes(derivative, "5b8b787f7715a5c5ba5eb3a1f16a25d4463222589ca99cac1fd17901d37ee750")
                with tarfile.open(derivative) as arc2:
                    check(arc2.extractfile(name).read() == raw, "derivative drawer source")
        rec = record["native"]
        if task == "drawer":
            query, body, env, chunks = (record[x] for x in ("query","body","env","calls"))
            key, pin = "first_five", record["pin"]
            donor, seed = 34, 42710334
        else:
            query, body, env, chunks = record["case"]["query"], record["model"], record["environment"], record["chunks"]
            key, pin, donor, seed = "original", record["origin"]["pin"], 31, 41800831
        check(query == dict(arm="IIWA", suite=suite, task_id=tid, variant="BASE", init_state_id=donor, seed=seed)
              and body["model_class"] == "MountedIIWA" and body["gripper"] == "PandaGripper"
              and env["released_row_fingerprint"] == row_hash(donor) and env["bddl_sha256"] == entry["bddl_sha256"]
              and not record["fixture"] and record["mode"] == "raw" and rec["success"] is True
              and rec["terminated"] is True and rec["truncated"] is False and rec["inferred"] == IDENTITY
              and rec["released_prompt"] == prompt and rec["reset"]["horizon"] == horizon, task+" source identity")
        n = rec["controls"]
        check(rec["native_rewards"] == [0.]*(n-1)+[1.] and rec["native_return"] == 1 and len(chunks) == (n+4)//5,
              task+" native source endpoint")
        for i,c in enumerate(chunks):
            sent = np.asarray(c["sent"], "<f4")
            expected = np.clip(np.asarray(c[key], "<f4")[:min(5,n-5*i)], -1, 1)
            check(expected.tobytes() == sent.tobytes() and c["prompt"] == prompt, task+" actual raw source controls")
        tape = np.array([a for c in chunks for a in c["sent"]], "<f4")
        k = int(np.flatnonzero(tape[:,6] > 0)[0])
        cut = 5*(k//5)
        check(cut == 105 and np.all(tape[:k,6] < 0), task+" source sign crossing")
        tapes[task] = tape[:cut]
        prediction, = (r for r in pred["rows"] if (r["panel"],r["suite"],r["task_id"],r["seed"],r["embodiment"]) ==
                       ("gpu06kinova3",suite,tid,41,"fam_nom"))
        ix = prediction["index"]
        meta, = (r for r in support_cases["rows"] if r["index"] == ix)
        support_file = ROOT/"outputs/xemb-icml-campaign-20261121-r1/gpu06kinova3/live"/f"shard-{ix//12}/support/00-fam_nom-passive.npz"
        h = prediction["tape_sha256"]
        checked_bytes(support_file,h)
        with np.load(support_file,allow_pickle=False) as arr:
            check(np.array_equal(arr["valid"], np.arange(24)<3) and
                  ("chunk_steps" not in arr or np.array_equal(arr["chunk_steps"],arr["valid"]*5)), "actual passive controls")
        check(meta["support_sha256"] == h and meta["support_layout"] == 11
              and meta["support_seed"] == {"drawer":31810311,"moka":31800811}[task], "distinct support")
        support = dict(panel="gpu06kinova3",index=ix,reader_seed=41,suite=suite,task_id=tid,shift="fam_nom",layout=11,
                       seed=meta["support_seed"],controls=15,tape_sha256=h,physics=meta["physics"],predictions_sha256=PRED,tuned=prediction["tuned"])
        result[task] = dict(suite=suite,task_id=tid,prompt=prompt,horizon=horizon,seed_base=seed_base,goal=goal,bank_sha256=bank_hash,
                            bddl_sha256=entry["bddl_sha256"],release_rows=row_hashes,support=support,source=dict(
                                archive=arc,archive_sha256=ar_hash,member=name,member_sha256=member_hash,pin=pin,query=query,controls=n,
                                native_return=rec["native_return"],first_positive_index=k,cut=cut,grip_previous=float(tape[k-1,6]),
                                grip_first_positive=float(tape[k,6]),full_commands_sha256=sha(tape.tobytes()),
                                prefix_commands_sha256=sha(tapes[task].tobytes())))
        if task == "drawer":
            result[task]["original_successful_iiwa_source_lengths"] = pool
    return result, tapes


def cpu():
    check(os.uname().sysname == "Linux" and os.sched_getaffinity(0) == {46,47}
          and os.environ["CUDA_VISIBLE_DEVICES"] == "" and os.environ["JAX_PLATFORMS"] == "cpu","audit CPU limits")


if __name__ == "__main__" and sys.argv[1] == "references":
    cpu()
    started = time.process_time()
    result, _ = references(checkout(BASE))
    print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),tasks=result,cpu_seconds=time.process_time()-started),indent=2))
