"""Finite native zero-steering comparison on one attested allocated GPU."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import socket
import subprocess
import threading
import time
import traceback

import numpy as np
import torch

from fluffy_bassoon.meta_rl.interface_ppo import InterfaceSpec
from fluffy_bassoon.meta_rl.joint_steering_ppo import JointActionJournal, cases_for
from fluffy_bassoon.meta_rl.joint_steering_ppo_worker import (
    _load_native, _RecordingEnvironment, _save, collect_bank, native_video_writer,
)
from fluffy_bassoon.meta_rl.residual_ppo import ResidualActorCritic, ResidualPPOConfig


class LimitedJournal:
    def __init__(self, root):
        self.journal = JointActionJournal(root, rank=0, category="eval", scope="native-evaluation-repair")
        self.lock = threading.Lock()

    def begin(self, requested, **kwargs):
        with self.lock:
            state = self.journal.snapshot()
            if sum(state["charged"].values()) + requested > 10_000:
                raise RuntimeError("Native repair qualification physical-control ceiling")
            return self.journal.begin(requested, **kwargs)

    def finish(self, *args):
        with self.lock:
            return self.journal.finish(*args)


def load_record(ref):
    raw = Path(ref["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ref["sha256"]
    return pickle.loads(gzip.decompress(raw))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    args = parser.parse_args()
    def stop(signum, frame):
        raise TimeoutError("Qualification allocation deadline; retain all completed evidence")
    signal.signal(signal.SIGTERM, stop)
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.time()
    deadline = start + 3000
    journal = LimitedJournal(args.output / "accounting")
    result = dict(completed=False, hostname=socket.gethostname(), started_unix=start,
                  optimizer_updates=0, deadline_unix=deadline, controls_ceiling=10_000)
    try:
        pin = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.code, text=True).strip()
        root = Path("/mowenjie/fluffy-bassoon-2")
        native = root / "outputs/robocasa-joint-steering-scale50-20261103-r1/production-velocity/PROFILE.json"
        profile = json.loads(native.read_text())
        wanted = [("OpenDrawer", 410100), ("TurnOnSinkFaucet", 540107),
                  ("TurnOnSinkFaucet", 540109), ("OpenStandMixerHead", 550109),
                  ("OpenStandMixerHead", 550136)]
        bank = {(case["task"], case["native_seed"]): case for case in cases_for("test_before")}
        cases = [bank[key] for key in wanted]
        base_request = dict(source_commit=pin, runtime_commit=profile["runtime_commit"],
            model_directory=profile["model_directory"], norm_path=profile["norm_path"])
        motor, factory, signature = _load_native(base_request, args.code)
        assert signature["all_parameters_frozen"]
        assert signature["model_sha256"] == profile["model_sha256"]
        result.update(source_commit=pin, native_signature=signature, cases=cases)
        collections = []
        for arm in ("velocity", "kv_all_layers"):
            spec = (InterfaceSpec(kind="velocity", steering_dim=32, fraction=.125) if arm == "velocity" else
                    InterfaceSpec(kind="kv_all_layers", steering_dim=1152, kv_layers=18, fraction=.64))
            actor = ResidualActorCritic(ResidualPPOConfig(2064, spec.steering_dim)).to(motor.model.device).eval()
            output = args.output / arm
            envs = [_RecordingEnvironment(factory.paired, cases, horizon=None,
                output=output / "videos" / str(lane), monitoring_root=output / "monitoring" / str(lane),
                journal=journal, writer_factory=native_video_writer, rank=0,
                episode_id_prefix=f"{arm}-lane{lane}-", canonical_root=args.output / "shared_cases",
                strict_pairing=True, source=dict(arm=arm, source_commit=pin)) for lane in range(2)]
            request = dict(base_request, cases=cases, spec=asdict(spec), sequence=2,
                phase="test_before", iteration=0, policy_iteration=0, lanes=2,
                deterministic=True, zero_steering=True, batch_mode="fixed_waves_v1")
            collection = collect_bank(actor, motor, envs, request=request,
                output=output / "history", rank=0, deadline_unix=deadline,
                qualification_path=output / "qualification.json")
            _save(output / "collection.pt", collection)
            collections.append(collection)
            result[arm] = dict(execution=collection["execution"],
                episodes=[dict(case=e["case"], success_once=e["success_once"],
                               success_at_end=e["success_at_end"], physical_chunks=e["physical_chunks"])
                          for e in collection["episodes"]])
            (args.output / "PROGRESS.json").write_text(json.dumps(dict(completed_arm=arm,
                elapsed_seconds=time.time() - start, escrow=journal.journal.snapshot()), indent=2) + "\n")
            del actor
        assert collections[0]["execution"] == collections[1]["execution"]
        pairs, pixel_disagreements, action_disagreements = 0, 0, 0
        for left, right in zip(collections[0]["episodes"], collections[1]["episodes"], strict=True):
            assert left["case"] == right["case"]
            assert left["success_once"] == right["success_once"]
            assert left["success_at_end"] == right["success_at_end"]
            for x, y in zip(left["chunks"], right["chunks"], strict=True):
                x, y = load_record(x), load_record(y)
                assert np.array_equal(x["native_noise"], y["native_noise"])
                assert np.count_nonzero(x["control"]) == np.count_nonzero(y["control"]) == 0
                pairs += 1
                pixel_disagreements += int(any(not np.array_equal(x["observation"][key], y["observation"][key])
                    for key in ("main_images", "wrist_images", "extra_view_images")))
                action_disagreements += int(not np.array_equal(x["requested_commands"], y["requested_commands"]))
        result.update(compared_physical_chunks=pairs, pixel_disagreements=pixel_disagreements,
                      action_disagreements=action_disagreements)
        assert not pixel_disagreements and not action_disagreements
        assert not journal.journal.snapshot()["pending"]
        result["completed"] = True
    except Exception:
        result["error"] = traceback.format_exc()
        raise
    finally:
        result.update(ended_unix=time.time(), elapsed_seconds=time.time() - start,
                      escrow=journal.journal.snapshot())
        (args.output / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
