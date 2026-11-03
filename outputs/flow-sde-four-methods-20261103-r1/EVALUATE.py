"""Matched actual-attempt Flow readout; complete cases and cold-copy receipts.

The execution owner supplies a finite remaining-budget admission. Completed
prospective cases count once; operational limits never create success rows.
"""
import argparse
from copy import deepcopy
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code", type=Path, required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--campaign-sha256", required=True)
    p.add_argument("--admission", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--method", choices=("ad", "dpt", "dit", "ic_iql", "native_initial", "native_final"), required=True)
    p.add_argument("--mode", choices=("full", "episode"), default="full")
    p.add_argument("--attempt", required=True)
    p.add_argument("--expected-hostname", required=True)
    p.add_argument("--case-limit", type=int, choices=(16, 100), default=16)
    a = p.parse_args()
    code, root = a.code, Path("/mowenjie/fluffy-bassoon-2")
    sys.path[:0] = [str(code / "src"), os.environ["RLINF_PATH"]]
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import checked_checkout, sha, tensor_hash
    from fluffy_bassoon.meta_rl.joint_steering_ppo import JointActionJournal
    from fluffy_bassoon.meta_rl.joint_steering_ppo_worker import _json as write, _load_native, _versions, native_video_writer
    if sys.platform != "linux" or socket.gethostname() != a.expected_hostname:
        raise ValueError("Use the exact exclusive allocated Linux hostname")
    checked_checkout(code, a.source_commit)
    if sha(a.campaign) != a.campaign_sha256:
        raise ValueError("The prospective complete campaign changed")
    campaign = json.loads(a.campaign.read_text())
    admission = json.loads(a.admission.read_text())
    if (admission.get("campaign_sha256") != a.campaign_sha256
            or admission.get("maximum_lifetime_GPU_seconds") != 27000
            or admission.get("method") != a.method or admission.get("mode") != a.mode
            or admission.get("attempt") != a.attempt
            or type(admission.get("charged_or_reserved_GPU_seconds")) is not int
            or not 0 <= admission["charged_or_reserved_GPU_seconds"] <= 27000
            or type(admission.get("allocation_GPU_seconds")) is not int
            or not 120 <= admission["allocation_GPU_seconds"] <= 27000 - admission["charged_or_reserved_GPU_seconds"]
            or a.case_limit == 100 and admission.get("complete_scope_forecast_fits_remaining") is not True
            or admission.get("old_allocation_and_children_released") is not True):
        raise ValueError("Require a checked finite lifetime budget and exact release admission")
    variant = a.method + "-" + a.mode
    run = a.output / variant
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / ".native-master.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for previous in run.glob("allocation-*.json"):
        previous_attempt = previous.stem.removeprefix("allocation-")
        marker = run / ("RELEASED-" + previous_attempt + ".json")
        if not marker.exists() or not json.loads(marker.read_text()).get("all_native_children_released"):
            raise ValueError("Explicit old-allocation release reconciliation is required")
    allocation = run / ("allocation-" + a.attempt)
    allocation.mkdir(exist_ok=False)
    started = time.time()
    deadline = started + admission["allocation_GPU_seconds"] - 60
    def interrupted(signum, frame):
        raise TimeoutError("Operational evaluation boundary; preserve complete and partial attempts")
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, interrupted)
    signal.setitimer(signal.ITIMER_REAL, max(.01, deadline - time.time()))
    state = dict(completed=False, method=a.method, mode=a.mode, attempt=a.attempt,
        source_commit=a.source_commit, started_unix=started, newly_completed_cases=0)
    try:
        probe = run / ("allocation-" + a.attempt + ".json")
        subprocess.run([sys.executable, "-B", str(code / "scripts/[train]gpu_probe.py"),
            "--expected-count", "1", "--output", str(probe), "--baige-container", "--purpose", "training",
            "--baige-exclusive-training"], check=True, timeout=100)
        proof = json.loads(probe.read_text())
        device = proof["devices"][0]
        os.environ.update(LYCO_BAIGE_GPU_MAP=str(probe), LYCO_BAIGE_ISOLATED_UUID=device["uuid"],
            MUJOCO_EGL_DEVICE_ID=str(device["allocation_ordinal"]))
        import numpy as np
        import torch
        from hydra import compose, initialize_config_dir
        from rlinf.models.embodiment.openpi import get_model
        from fluffy_bassoon.meta_rl.flow_history import DirectActionCodec, FlowDirectMotor, FlowReferenceMotor, ReferencePlaceholder
        from fluffy_bassoon.meta_rl.steering_history_native import NativeAttemptBackend, case_key
        from fluffy_bassoon.meta_rl.steering_history_eval import weights_digest
        from fluffy_bassoon.meta_rl.vla_history import load_policy
        from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import TASKS
        torch.set_num_threads(2)
        # Process-local protocol for this separate completed H400 Flow source.
        # No file or any held scientific PPO job/profile is changed.
        TASKS["OpenDrawer"] = (400, *TASKS["OpenDrawer"][1:])
        if TASKS["CloseFridge"][0] != 900:
            raise ValueError("Keep the prospective native OOD horizon900")
        norm = root / "inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json"
        model = root / "inputs/robocasa365-public-pi05-c484-torch-20261101-r2"
        os.environ.update(ROBOCASA_NATIVE_OUTPUT=str(allocation / "native"),
            ROBOCASA_PI05_TORCH_MODEL=str(model), ROBOCASA_PI05_NATIVE_NORM=str(norm))
        request = dict(source_commit=a.source_commit, runtime_commit="8a1233c3fae6f5905adfda7684c455842b0ecfb8",
            model_directory=str(model), norm_path=str(norm))
        native, factory, signature = _load_native(request, code)
        if (signature["model_sha256"] != "661bf172a2ae9dfca75216697eeaa543408bcbdfa53c98132759b93e0e8993a5"
                or signature["norm_sha256"] != "4aed1af411bd0e0f49d2b0e6d6832b11b8682917231a07173fbc30fd493bbdee"):
            raise ValueError("Original public feature initializer changed")
        stats = json.loads(norm.read_text()); stats = stats.get("norm_stats", stats)
        spec = DirectActionCodec.from_stats(stats)
        assembly_path = root / "outputs/flow-four-method-source-gpu-20261103-r2/preparation/ASSEMBLY.json"
        if sha(assembly_path) != "12433cf163c03a7874f6028e8bdc2a662f25fdc964e9fd972b7ac4d4f2b1760c":
            raise ValueError("Audited physical Flow source changed")
        assembly = json.loads(assembly_path.read_text())
        if assembly["codec"] != spec.record():
            raise ValueError("Training/evaluation action maps differ")
        teacher_dir = root / "outputs/robocasa-opendrawer-flow-sde-20261102-r4/opendrawer_flow_sde/checkpoints/global_step_18/actor/model_state_dict"
        if a.method.startswith("native_"):
            if a.mode != "full":
                raise ValueError("Native anchors have no learned memory reset variant")
            reference = native.model
            reference_identity = signature["model_sha256"]
            if a.method == "native_final":
                if sha(teacher_dir / "full_weights.pt") != "7e65e25408ed653d4c635a7f23b788ac621e8ff9909e866cec88173f519eff61":
                    raise ValueError("Final native reference changed")
                with initialize_config_dir(config_dir=str(code / "configs/native_ppo"), version_base=None):
                    cfg = compose(config_name="robocasa365_pi05_eval")
                cfg.rollout.model.model_path = str(teacher_dir)
                cfg.rollout.model.openpi_data.norm_stats_path = str(norm)
                reference = get_model(cfg.rollout.model, torch_dtype=torch.bfloat16).to("cuda").eval().requires_grad_(False)
                reference_identity = sha(teacher_dir / "full_weights.pt")
            policy = ReferencePlaceholder().to("cuda")
            motor = FlowReferenceMotor(native, spec, reference)
            selected = dict(kind=a.method, sha256=reference_identity)
        else:
            directory = root / "outputs/flow-four-method-fits-20261103-r1" / a.method
            status = json.loads((directory / "training-status.json").read_text())
            if status.get("converged") is not True:
                raise ValueError("Do not label an operational student checkpoint converged")
            candidate = directory / "candidate.pt"
            policy, manifest = load_policy(candidate, expected_sha256=sha(candidate), device="cuda:0")
            contract = manifest["contract"]
            if (contract["objective"] != a.method or contract["mode"] != "full"
                    or contract["train_sha256"] != assembly["datasets"][a.method + "_train"]["sha256"]
                    or contract["validation_sha256"] != assembly["datasets"][a.method + "_validation"]["sha256"]
                    or policy.config.observation_dim != 2073 or policy.config.noise_dim != 600
                    or policy.config.context_length != 128):
                raise ValueError("Selected actor differs from its actual qualified Flow corpus")
            motor = FlowDirectMotor(native, spec)
            selected = dict(kind=a.method, path=str(candidate), sha256=sha(candidate))
        binding = dict(format="flow_four_method_matched_readout_v1", campaign_sha256=a.campaign_sha256,
            selected=selected, source_commit=a.source_commit, runtime_commit=request["runtime_commit"],
            model_sha256=signature["model_sha256"], norm_sha256=signature["norm_sha256"],
            source_assembly_sha256=sha(assembly_path), render_contract="diagnostic_only",
            policy_weights_sha256=weights_digest(policy), task_horizons=dict(OpenDrawer=400, CloseFridge=900))
        write(allocation / "BINDING.json", binding)
        previous_binding = run / "BINDING.json"
        if previous_binding.exists() and json.loads(previous_binding.read_text()) != binding:
            raise ValueError("Explicit fresh-source compatibility review is required")
        if not previous_binding.exists(): write(previous_binding, binding)

        class Journal(JointActionJournal):
            def __init__(self):
                self.root, self.rank, self.category, self.scope = a.output / "accounting", 0, "eval", variant
                self.root.mkdir(parents=True, exist_ok=True)
                self.calls = self.root / "rank-00"; self.calls.mkdir(exist_ok=True)
                self.state_path = self.root / "escrow.json"
                limits = dict(train=0, eval=4000000)
                with self._locked():
                    if not self.state_path.exists():
                        write(self.state_path, dict(format="flow_history_eval_action_escrow_v1", limits=limits,
                            charged=dict(train=0, eval=0), pending={}))
                    if self._read()["limits"] != limits:
                        raise ValueError("Preserve this separate fixed Flow evaluation escrow")
        journal = Journal()
        with np.load(root / "outputs/flow-four-method-source-audit-20261103-r1/RAW-PUBLIC-CORPUS.npz") as z:
            source_seeds = set(map(int, z["native_seeds"].reshape(-1)))
        cases = []
        for slot in range(50):
            for task in ("OpenDrawer", "CloseFridge"):
                seed = campaign["evaluation"]["prospective_native_seed_bases"][task] + slot
                if seed in source_seeds: raise ValueError("Held-out case reused a source seed")
                case = dict(bank="flow_four_method_eval", task=task, native_seed=seed, split="target",
                    slot=slot, reset_ordinal=0, counted=True, layout_source="native_target_constructor_seed_first_reset")
                cases.append(dict(case=case, horizon=TASKS[task][0]))
        cold = Path("/mnt/bos/adgfyl/strong-plant-archives/20261103-flow-four-method-readout-r1")
        sentinel = cold.parent / "20261103-joint-steering-owned-flow-checkpoints-r1/checkpoint-00/actor/model_state_dict/full_weights.pt"
        if not sentinel.exists():
            raise ValueError("Mount the existing private BOS archive for lossless closed-case retention")
        cold.mkdir(exist_ok=True)
        def archive(directory, destination):
            destination.mkdir(parents=True, exist_ok=False)
            receipt = {}
            for path in sorted(directory.rglob("*")):
                if not path.is_file(): continue
                relative = path.relative_to(directory)
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                before = sha(path); size = path.stat().st_size
                with path.open("rb") as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst, length=8 << 20); dst.flush(); os.fsync(dst.fileno())
                if target.stat().st_size != size or sha(target) != before or sha(path) != before:
                    raise ValueError("Cold-copy byte readback differs; preserve both payloads")
                link = path.with_name(path.name + ".cold-link")
                link.symlink_to(target); link.replace(path)
                receipt[str(relative)] = dict(size_bytes=size, sha256=before, destination=str(target))
            return receipt

        initial_versions = tensor_hash(_versions(native.model))
        for first in range(0, a.case_limit, 16):
            chunk = cases[first:min(a.case_limit, first+16)]
            marker = run / f"complete-bank-{first:03d}.json"
            if marker.exists():
                saved = json.loads(marker.read_text())
                if saved["binding"] != binding or saved["cases"] != chunk:
                    raise ValueError("Completed prospective bank changed")
                # Exact retained references must remain available through BOS.
                prior_refs = [ref for refs in saved.get("reused_prefix_references", {}).values() for ref in refs]
                for ref in saved["attempt_references"] + prior_refs:
                    if sha(ref["path"]) != ref["sha256"]: raise ValueError("Completed attempt changed")
                continue
            prior_banks = list((run / "attempts").glob(f"*/bank-{first:03d}"))
            prefix_path = run / f"PREFIX-RECOVERY-{first:03d}.json"
            if prior_banks and not prefix_path.exists():
                raise ValueError("Reconcile actual completed siblings/prefixes and partial costs before recovery")
            prefixes = json.loads(prefix_path.read_text()) if prefix_path.exists() else {}
            if shutil.disk_usage(root).free < 848 * 2**30:
                state["paused"] = "protect_remaining_steering_storage"; break
            if time.time() + 120 >= deadline:
                state["paused"] = "finite_allocation_clock"; break
            directory = run / "attempts" / a.attempt / f"bank-{first:03d}"
            backend = NativeAttemptBackend(output=directory, canonical_root=a.output / "canonical", motor=motor,
                spec=spec, native_factory=factory, journal=journal, writer_factory=native_video_writer,
                binding=binding, maximum_rgb_delta=None, memory_mode=a.mode)
            bank_started = time.time()
            results = backend.run_bank(policy, chunk, lanes=16, attempts=5,
                prefix_references=prefixes, deadline_unix=deadline)
            elapsed = time.time()-bank_started
            if tensor_hash(_versions(native.model)) != initial_versions:
                raise ValueError("Frozen native feature model mutated")
            physical_controls = sum(item["horizon"]*(5-len(prefixes.get(case_key(item["case"]), ()))) for item in chunk)
            # Publish actual complete result before cold-copying closed payloads.
            record = dict(binding=binding, cases=chunk, results=results, reused_prefix_references=prefixes,
                attempt_references=backend.references, collection_seconds=elapsed,
                newly_executed_policy_controls=physical_controls, controls_per_second=physical_controls/elapsed,
                actual_batch_sizes=backend.last_batch_sizes, source_escrow=journal.snapshot())
            write(marker, record)
            archive_started = time.time()
            receipt = archive(directory, cold / variant / a.attempt / f"bank-{first:03d}")
            write(run / f"cold-bank-{first:03d}.json", dict(files=receipt,
                elapsed_seconds=time.time()-archive_started,
                actual_collection_and_archive_seconds=time.time()-bank_started,
                preserved_bytes=sum(item["size_bytes"] for item in receipt.values())))
            state["newly_completed_cases"] += len(chunk)
            write(allocation / "PROGRESS.json", dict(state, checked_unix=time.time(), last_bank=first,
                complete_cases=sum(len(json.loads(m.read_text())["cases"]) for m in run.glob("complete-bank-*.json"))))
        complete_cases = sum(len(json.loads(m.read_text())["cases"]) for m in run.glob("complete-bank-*.json"))
        state.update(completed=complete_cases==100, complete_cases=complete_cases,
            protocol_and_actual_selected_policy_qualified=complete_cases>=16,
            unchanged_policy_weights=weights_digest(policy)==binding["policy_weights_sha256"], escrow=journal.snapshot())
        write(allocation / "RESULT.json", state)
    except BaseException as error:
        import traceback
        state.update(error=repr(error), traceback=traceback.format_exc())
        write(allocation / "FAILURE.json", state)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        live_children = [child.pid for child in mp.active_children() if child.is_alive()]
        write(run / ("RELEASED-" + a.attempt + ".json"), dict(hostname=a.expected_hostname,
            all_native_children_released=not live_children, live_children=live_children,
            started_unix=started, ended_unix=time.time(), source_commit=a.source_commit))
        lock.close()


if __name__ == "__main__":
    main()
