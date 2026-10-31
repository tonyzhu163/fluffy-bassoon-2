"""Prospective six-lineage source collection inside ONE owner allocation.

Processes run independent complete episodes concurrently. Only action admission
and canonical ledger writes pass through the parent. This entry retains the
existing one-visible-GPU backend; concurrency2..6 needs an actual capacity and
real-episode receipt, and an eligible frontend, before any model is loaded.
Legacy robotwin_steering_source remains unchanged and available.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys

from . import robotwin_steering_source as serial
from .steering_condition_collection import collect_lineage, SOURCE_METHOD_KEYS
from fluffy_bassoon.rl.robotwin_source_broker import (
    SourceAdmissionStop, SourceBudgetBroker, SourceBudgetClient,
)


FORMAT = "frozen_motus_parallel_source_v1"
ALLOCATION = "frozen_motus_parallel_source_allocation_v1"
SHARDS = "frozen_motus_source_shards_v1"


def method_limits(profile, episode_target=24):
    if episode_target not in (24, 48):
        raise ValueError("Only original initial24 or separately admitted upper48")
    bound = profile.episode_limit + profile.reset_action_bound
    validation_episodes = 2 * (3 if episode_target == 24 else 5)
    return {method: dict(source=episode_target*bound, source_validation=validation_episodes*bound)
            for method in SOURCE_METHOD_KEYS}


def validate_parallel_admission(scope, allocation, profile, *, source, hostname, output, profile_sha):
    """New prospective execution envelope; no migration of historical charges."""
    limits = scope.get("phase_limits", {})
    if (scope.get("schema") != "frozen_pi05_endtoend_scope_v1"
            or limits.get("all_in_simulator_actions") != 300000
            or limits.get("outputs_gib") != 64
            or limits.get("fresh_cpu_qualification_worker_seconds") != 1800):
        raise ValueError("Original action/storage/CPU scientific phase caps differ")
    concurrency = allocation.get("max_concurrency")
    threads, cpus = allocation.get("worker_cpu_threads"), allocation.get("allocated_cpu_count")
    if (allocation.get("schema") != ALLOCATION or type(concurrency) is not int or not 2 <= concurrency <= 6
            or type(threads) is not int or threads < 1 or type(cpus) is not int or cpus < concurrency*threads
            or allocation.get("scope_sha256") != profile.scope_sha256
            or allocation.get("profile_sha256") != profile_sha or allocation.get("source_commit") != source
            or allocation.get("hostname") != hostname or allocation.get("gpu_count") != 1
            or not allocation.get("job_id") or allocation.get("episode_target") != 24
            or allocation.get("worker_devices") != dict.fromkeys(SOURCE_METHOD_KEYS, "cuda:0")
            or allocation.get("worker_method_limits") != method_limits(profile)
            or profile.native_horizon != 32 or profile.episode_limit != 400 or profile.reset_action_bound != 0):
        raise ValueError("Require bound initial24 six-lineage concurrent one-visible-GPU execution profile")
    output, phase = Path(output).resolve(), Path(allocation["phase_output_root"]).resolve()
    corpus = Path(allocation["source_corpus_root"]).resolve()
    if (Path(allocation["output"]).resolve() != output or not output.is_relative_to(phase)
            or not corpus.is_relative_to(phase) or corpus.is_relative_to(output)
            or output.is_relative_to(corpus) or output == phase or corpus == phase):
        raise ValueError("Fresh job output and separate canonical phase shard index must be within shared output")
    # Human approved EACH job under8h. Preserve historical cost fields as
    # accounting; do not reinterpret the old scope's cumulative8h as that approval.
    seconds = allocation.get("reserved_gpu_seconds")
    if (allocation.get("authorization_basis") != "human_each_job_under_8h"
            or type(seconds) is not int or not 0 < seconds < 28800
            or any(type(allocation.get(k)) is not int or allocation[k] < 0
                for k in ("prior_phase_gpu_seconds", "other_reserved_phase_gpu_seconds"))):
        raise ValueError("This one parent job requires explicit under8h authorization and original cost accounting")
    clocks = [allocation[k] for k in ("driver_deadline_unix", "allocation_started_unix",
        "maximum_episode_seconds", "maximum_episode_update_seconds", "cleanup_reserve_seconds")]
    if (any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in clocks)
            or allocation["driver_deadline_unix"] > allocation["allocation_started_unix"] + seconds
            or allocation["maximum_episode_seconds"] <= 0 or allocation["maximum_episode_update_seconds"] < 0
            or allocation["cleanup_reserve_seconds"] < 60):
        raise ValueError("Finite whole-episode/update and parent cleanup clock required")
    ledger = allocation["action_ledger"]
    profile.validate_budget(type("BoundHeader", (), {"header": ledger})())
    if (ledger["campaign_id"] != scope["phase_id"] or ledger["contract"].get("phase_id") != scope["phase_id"]
            or ledger["contract"].get("motor_sha256") != profile.motor_sha256
            or not (Path(ledger["root"])/"events/0000000000.json").is_file()):
        raise ValueError("Reuse the original canonical campaign; no worker-specific allowance")
    full, chunk = allocation["max_full_recovery_bytes"], allocation["max_chunk_output_bytes"]
    reserve, other = allocation["recovery_output_reservation_bytes"], allocation["other_future_output_reservation_bytes"]
    if (any(type(v) is not int or v < 0 for v in (full, chunk, reserve, other)) or not full or not chunk
            or reserve < (60+concurrency)*full or reserve+other+concurrency*chunk > 64 << 30):
        raise ValueError("Admit all original full recoveries plus concurrent replacement/raw-write headroom")


def _artifact(reference):
    path = Path(reference["path"])
    if "root" in reference:
        root = Path(reference["root"]).resolve()
        path = (root/path).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Canonical shard reference escaped its root")
    return path, serial.read_bound(path, reference["sha256"])


def validate_capacity(receipt, allocation, profile, budget, *, source, hostname, gpu_uuid, frozen_fingerprint):
    """Actual qualification is a separately charged owner experiment."""
    concurrency = allocation["max_concurrency"]
    if (receipt.get("format") != "frozen_motus_parallel_source_capacity_v1"
            or receipt.get("status") != "passed" or receipt.get("source_commit") != source
            or receipt.get("hostname") != hostname or receipt.get("max_concurrency") != concurrency
            or receipt.get("gpu_count") != 1 or receipt.get("device") != "cuda:0"
            or receipt.get("worker_cpu_threads") != allocation["worker_cpu_threads"]
            or receipt.get("allocated_cpu_count") != allocation["allocated_cpu_count"]
            or not gpu_uuid or receipt.get("gpu_uuid") != gpu_uuid
            or receipt.get("frozen_tensor_sha256") != frozen_fingerprint
            or receipt.get("mapping_sha256") != profile.mapping_sha256
            or receipt.get("interface_sha256") != profile.interface_sha256
            or receipt.get("motor_sha256") != profile.motor_sha256
            or receipt.get("native_horizon") != 32 or receipt.get("episode_limit") != 400
            or receipt.get("max_full_recovery_bytes") != allocation["max_full_recovery_bytes"]
            or receipt.get("max_chunk_output_bytes") != allocation["max_chunk_output_bytes"]
            or receipt.get("overlapping_complete_episodes") is not True):
        raise ValueError("Actual exact-code/frontend/concurrency capacity and overlapping real episodes required")
    cost = serial.read_reference(receipt["scheduler_cost_receipt"])
    if (cost.get("status") != "closed" or type(cost.get("actual_gpu_seconds")) is not int
            or not 0 < cost["actual_gpu_seconds"] < 28800):
        raise ValueError("Every capacity qualification requires its actual closed scheduler cost")
    rows = receipt.get("episodes", [])
    if (len(rows) != concurrency or len({r["pid"] for r in rows}) != concurrency
            or len({r["method"] for r in rows}) != concurrency):
        raise ValueError("Qualification needs distinct concurrent processes and actual complete episodes")
    decisions = budget.snapshot(include_decisions=True)["decisions"]
    seen, intervals, renderer_pcis = set(), [], set()
    for row in rows:
        if (row.get("complete") is not True or row.get("method") not in SOURCE_METHOD_KEYS
                or row.get("frozen_before") != frozen_fingerprint or row.get("frozen_after") != frozen_fingerprint):
            raise ValueError("Require actual canonical complete episodes and frozen motor identity")
        manifest_path, manifest = _artifact(row["episode_manifest_ref"])
        renderer = serial.read_reference(row["renderer_binding"])
        ids = row["decision_ids"]
        if (not ids or len(ids) != len(set(ids)) or set(ids) & seen
                or manifest.get("pending_decision") is not None
                or manifest.get("completed_chunks") != len(ids) or len(manifest.get("chunks", [])) != len(ids)
                or manifest.get("executed_primitive_steps") != row["actual_actions"]
                or not 0 < row["actual_actions"] <= 400
                or renderer.get("pid") != row["pid"]
                or not renderer.get("expected_cuda_pci")
                or renderer.get("expected_cuda_pci") != renderer.get("actual_render_pci")):
            raise ValueError("Qualification raw episode/renderer/charged action identity differs")
        chunks = [serial.read_bound(manifest_path.parent/ref["path"], ref["sha256"])
                  for ref in manifest["chunks"]]
        if (any(c.get("status") != "complete" or c.get("decision_metadata", {}).get("campaign_decision_id") != identity
                for c, identity in zip(chunks, ids))
                or chunks[-1]["learning_result"].get("budget_truncated") is not False
                or not (chunks[-1]["learning_result"].get("terminated") or chunks[-1]["learning_result"].get("truncated"))):
            raise ValueError("Actual complete raw chunks must bind every charged qualification intent")
        charged = [decisions.get(identity, {}) for identity in ids]
        if (any(d.get("status") != "resolved" or d.get("method") != row["method"] for d in charged)
                or sum(d["actual_actions"] for d in charged) != row["actual_actions"]):
            raise ValueError("Capacity actions must already be resolved in the SAME canonical campaign")
        seen.update(ids)
        intervals.append((chunks[0]["intent_created_unix_ns"]/1e9,
            chunks[-1]["intent_created_unix_ns"]/1e9 + chunks[-1]["native_result"]["duration_seconds"]))
        renderer_pcis.add(renderer["actual_render_pci"])
    if len(renderer_pcis) != 1 or max(start for start, _ in intervals) >= min(end for _, end in intervals):
        raise ValueError("Actual qualified physical episode intervals must overlap on the one bound GPU")


class ParallelGuard(serial.RuntimeGuard):
    def __init__(self, allocation, budget):
        super().__init__(allocation)
        self.budget = budget
        self.shard_root, self.raw_bytes = None, 0

    def storage(self):
        used = super().storage()
        extra = self.allocation["max_concurrency"] * (
            self.allocation["max_full_recovery_bytes"] + self.allocation["max_chunk_output_bytes"])
        if used + extra > 64 << 30:
            raise RuntimeError("Concurrent full-state/raw writes exceed shared output headroom")
        return used

    def admit_episode(self, *, episodes=1):
        self.stopping = self.stopping or self.budget.call("guard")["stopping"]
        super().admit_episode(episodes=episodes)

    def check_raw_write(self):
        if self.shard_root is None:
            return
        used = sum(p.stat().st_size for p in self.shard_root.rglob("*") if p.is_file())
        if used - self.raw_bytes > self.allocation["max_chunk_output_bytes"]:
            raise RuntimeError("Actual isolated raw write exceeds qualified concurrent headroom")
        self.raw_bytes = used


def _publication(teacher, reference):
    return dict(lineage=asdict(teacher.lineage), condition_id=teacher.lineage.condition_id,
        mapping_sha256=teacher.profile.mapping_sha256, interface_sha256=teacher.profile.interface_sha256,
        complete_episode_manifests=[r["episode_manifest_ref"] for r in teacher.completed_episodes],
        actual_resets=[r["reset_info"] for r in teacher.completed_episodes],
        initial_full_state=teacher.initial_recovery, final_full_state=reference,
        actual_optimizer_updates=teacher.learner.updates, actual_primitive_steps=teacher.learner.primitive_steps,
        source_validation_receipts=deepcopy(teacher.validation_receipts),
        behavioral_improvement="requires fixed source validation and retry comparison",
        DPT_teacher_qualification="separate physical gate; no automatic permission")


def run_lineage(*, control, budget, profile, lineage, distribution, output, guard, archive, run_id, device="cuda:0"):
    """One original learner, replay, RNG stream and canonical shard per process."""
    output = Path(output)
    sink = serial.BoundedTeacherRecovery(output/"recoveries", guard=guard, control=control)
    def checkpoint(event, payload, *, immutable):
        ref = sink(event, payload, immutable=immutable)
        if ref["size_bytes"] > guard.allocation["max_full_recovery_bytes"]:
            raise RuntimeError("Actual source recovery exceeds qualified capacity")
        return ref
    def identify(learner):
        guard()
        return archive.identify(learner, "sac")
    teacher = serial.CoefficientTeacher(serial.make_learner(profile, lineage, device=device), profile, lineage,
        checkpoint=checkpoint, archive_actor=identify, guard=guard)
    publications = {}

    def episode(current, spec, ledger, **kwargs):
        pre_admitted = kwargs.pop("_parallel_panel_admitted", False)
        if not pre_admitted:
            try:
                guard.admit_episode()
                budget.call("admit", specs=[spec], phase="source")
            except (serial.BoundaryStop, SourceAdmissionStop):
                # collect_lineage already persisted episode_begin. Only clear
                # this provably unstarted cursor; no reset/intent has happened.
                if teacher.decision_count or teacher.pending is not None:
                    raise RuntimeError("Cannot clear an interrupted physical prefix")
                teacher.active_episode = None
                teacher._persist("unstarted_parallel_admission_stop", immutable=True)
                raise
        def check(*_):
            guard()
            check_raw = getattr(guard, "check_raw_write", None)
            if check_raw is not None:
                check_raw()
        record = serial.run_complete_episode(current, spec, ledger, **kwargs,
            before_step=check, on_chunk=check)
        check()  # Also audit a possible zero-valid terminal raw chunk.
        compact = {k: v for k, v in record.items() if k != "control_recovery"}
        path = output/"episodes"/(spec["id"]+".json")
        serial._atomic_json(path, compact)
        budget.call("complete", episode=spec["id"], receipt={k: compact[k] for k in
            ("complete", "chunks", "primitive_steps", "episode_manifest_ref")})
        guard.maximum_episode_seconds = max(guard.maximum_episode_seconds, 1.5*record["wall_seconds"])
        return record

    def validate(checkpoint_ref, specs):
        guard.admit_episode(episodes=2)
        baseline = None
        if specs[0]["milestone"] != 0:
            baseline = serial.read_reference(teacher.validation_receipts["0"])["results"]
            specs = [serial.paired_validation_spec(s, baseline[i]) for i, s in enumerate(specs)]
        budget.call("admit", specs=specs, phase="source_validation")
        saved = serial.BoundedTeacherRecovery.load(checkpoint_ref)["collector"]
        results = []
        for spec in specs:
            distribution.select(lineage.sign)
            frozen = serial.FrozenSourceBehavior(saved, profile, lineage, spec,
                archive_actor=identify, guard=guard, device=device)
            results.append(episode(control, spec, budget, run_id=run_id, phase="source_validation",
                controller=frozen, _parallel_panel_admitted=True))
        path = output/"validation"/(specs[0]["id"].rsplit(":", 1)[0]+".json")
        serial._atomic_json(path, dict(complete=True, phase="source_validation",
            episode_ids=[s["id"] for s in specs], source_checkpoint=checkpoint_ref,
            behavior="bounded uniform iff optimizer updates0; otherwise frozen sampled actual actor",
            results=[{k: v for k, v in r.items() if k != "control_recovery"} for r in results]))
        return dict(path=str(path), sha256=serial.sha(path), complete=True,
            phase="source_validation", episode_ids=[s["id"] for s in specs])

    status, error = "failed", None
    try:
        while len(teacher.completed_episodes) < 24:
            result = collect_lineage(teacher, control=control, budget=budget, run_id=run_id,
                episode_runner=episode, post_command_transform=distribution,
                validate_source=validate, publish_prefix=lambda _: None, episode_target=24, one_episode=True)
            if result["status"] == "shared_action_cap_stop":
                status = result["status"]
                break
            count = len(teacher.completed_episodes)
            if count in (12, 24):
                reference = teacher.last_recovery
                if reference["retention"] != "immutable":
                    reference = teacher._persist("parallel_balanced_prefix", immutable=True)
                path = output/"publications"/f"{count:03d}.json"
                serial._atomic_json(path, _publication(teacher, reference))
                publications[str(count)] = dict(path=str(path), sha256=serial.sha(path))
        else:
            status = "initial_prefix_complete"
    except (serial.BoundaryStop, SourceAdmissionStop) as exc:
        status, error = "boundary_stop", str(exc)
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        recovery_error = None
        if status == "failed":
            try:
                # In particular, capture the current control/global RNG state
                # if a separately frozen validation episode was interrupted.
                teacher._persist("parallel_interrupted_full_state", immutable=True)
            except BaseException as exc:
                recovery_error = repr(exc)  # Keep the prior durable full state; never hide the original error.
        state = dict(format=FORMAT, status=status, error=error, lineage=asdict(lineage),
            complete_episodes=len(teacher.completed_episodes), publications=publications,
            full_recovery=teacher.last_recovery, initial_full_recovery=teacher.initial_recovery,
            actual_optimizer_updates=teacher.learner.updates, actual_primitive_steps=teacher.learner.primitive_steps,
            final_recovery_error=recovery_error,
            closed_physical_boundary=not control.active and teacher.active_episode is None
                and teacher.pending is None and not teacher.updating and not teacher.pending_update_cycles,
            recovery_meaning="Partial physical episodes/optimizer cycles require owner reconciliation; never replay")
        serial._atomic_json(output/"lineage-state.json", state)
    return state


def _worker(connection, args, allocation, profile_data, interface, qualification, lineage_data, header, source,
            control_factory=None):
    """Spawn target: create all mutable learner/runtime objects inside child."""
    from .steering_condition_collection import LineageSpec, SourceProfile
    lineage, profile = LineageSpec(**lineage_data), SourceProfile(**profile_data)
    worker_output = Path(args.output)/"workers"/lineage.method
    worker_output.mkdir(parents=True, exist_ok=False)
    budget = SourceBudgetClient(connection, worker=lineage.method, header=header)
    guard = ParallelGuard(allocation, budget)
    control = env = None
    result = dict(status="failed", lineage=asdict(lineage))
    try:
        guard.admit_episode()
        # These global streams are private to this spawned process, in
        # addition to NoiseSAC's existing independent sampler generators.
        random.seed(lineage.learner_seed)
        serial.np.random.seed(lineage.learner_seed)
        serial.torch.manual_seed(lineage.learner_seed)
        serial.torch.cuda.manual_seed_all(lineage.learner_seed)
        serial.torch.set_num_threads(allocation["worker_cpu_threads"])
        serial.torch.set_num_interop_threads(1)
        device = allocation["worker_devices"][lineage.method]
        serial.torch.cuda.set_device(device)
        shard = Path(allocation["source_corpus_root"])/"shards"/lineage.method
        writer = serial.RobotwinCorpusWriter(shard, dict(schema=SHARDS, source_commit=source,
            motor_sha256=profile.motor_sha256, interface_sha256=args.interface_sha256,
            profile_sha256=args.profile_sha256, lineage=asdict(lineage),
            role="isolated_source_condition_changing_policy_and_fixed_validation"))
        guard.shard_root = shard
        guard.check_raw_write()
        env = serial.RobotwinNoiseEnv(model_path=args.model_path, norm_stats=args.norm_stats,
            robotwin_root=args.robotwin_root, rlinf_root=args.rlinf_root, device=device,
            output_dir=worker_output/"runtime", action_horizon=32, num_steps=5,
            task_episode_limits=serial.load_benchmark_episode_limits(args.robotwin_root, serial.TASKS))
        factory = serial.CoefficientSteering if control_factory is None else control_factory
        control = factory(env, interface, interface_sha256=args.interface_sha256)
        control.set_coefficients(profile.arm, serial.np.zeros(4, serial.np.float32))
        if control.mapping_sha256() != profile.mapping_sha256:
            raise ValueError("Worker actual frontend mapping differs from qualification")
        distribution = serial.CalibrationDistribution(serial.calibration_pair(
            physical_basis=control.physical_basis(), physical_basis_sha256=control.config()["action"]["basis_sha256"],
            arm_joint_bound=interface["arm_joint_bound"], selected_arm=profile.arm,
            command_lower=interface["joint_lower"], command_upper=interface["joint_upper"],
            range_receipt_sha256=args.joint_conventions_sha256, motor_sha256=profile.motor_sha256,
            mapping_sha256=profile.mapping_sha256))
        control.condition_contract = distribution.contract()
        control.post_command_transform = env.backend.post_command_transform = distribution
        env.backend.contract = dict(env.backend.contract, coefficient_steering=control.config())
        env.contract = env.backend.contract
        control.attach_corpus(writer)
        if env.frozen_fingerprint() != qualification["frozen_after"]:
            raise ValueError("Worker frozen motor differs from actual frontend qualification")
        result = run_lineage(control=control, budget=budget, profile=profile, lineage=lineage,
            distribution=distribution, output=worker_output, guard=guard,
            archive=serial.existing_actor_archive(worker_output/"actors"),
            run_id=f"{args.run_id}:{lineage.method}", device=device)
        if env.frozen_fingerprint() != qualification["frozen_after"]:
            raise RuntimeError("Frozen motor changed during independent source learning")
        result["frozen_motor_unchanged"] = True
    except BaseException as exc:
        saved = worker_output/"lineage-state.json"
        if saved.is_file():
            result = json.loads(saved.read_text())
        result.update(status="failed", error=repr(exc))
    finally:
        closed, close_error = False, None
        try:
            if control is not None:
                control.close()
            elif env is not None:
                env.close()
            closed = True
        except BaseException as exc:
            close_error = repr(exc)
            result.update(status="failed", cleanup_error=close_error)
        result["control_closed"] = closed
        serial._atomic_json(worker_output/"cleanup.json", dict(control_closed=closed, cleanup_error=close_error))
        connection.send(dict(operation="finished", result=result))
        connection.close()


def supervise(args, allocation, profile, interface, qualification, budget, source, *, control_factory=None):
    """All six methods execute, in concurrent waves if qualified capacity<6."""
    context = mp.get_context("spawn")
    processes, connections, results = {}, {}, {}
    queued = list(serial.lineage_specs(profile.learner_seed))
    def live(method, pid):
        process = processes.get(method)
        return bool(process and process.pid == pid and process.is_alive())
    broker = SourceBudgetBroker(budget, Path(args.output)/"budget-broker",
        method_limits=allocation["worker_method_limits"], episode_bound=profile.episode_limit, is_live=live)
    def stopping(*_):
        broker.stop()
    old_signals = {s: signal.signal(s, stopping) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        while queued or connections:
            active = sum(p.is_alive() for p in processes.values())
            while queued and active < allocation["max_concurrency"] and not broker.stopping and not broker._hazards():
                lineage = queued.pop(0)
                parent, child = context.Pipe(duplex=True)
                process = context.Process(target=_worker, name=lineage.method,
                    args=(child, args, allocation, asdict(profile), interface, qualification,
                          asdict(lineage), budget.header, source)
                         + (() if control_factory is None else (control_factory,)))
                process.start()
                child.close()
                processes[lineage.method] = process
                broker.register(lineage.method, process.pid)
                connections[parent] = lineage.method
                active += 1
            if not connections:
                if queued and active and not broker.stopping and not broker._hazards():
                    # A child sent its result after closing its environment,
                    # but its model memory is released only on process exit.
                    wait([p.sentinel for p in processes.values() if p.is_alive()], timeout=0.5)
                    continue
                break
            for connection in wait(list(connections), timeout=0.5):
                method = connections[connection]
                try:
                    message = connection.recv()
                except EOFError:
                    message = dict(operation="finished", result=dict(status="worker_crash", method=method))
                if message["operation"] == "finished":
                    results[method] = message["result"]
                    connection.close()
                    del connections[connection]
                    broker.exited(method)
                    continue
                try:
                    value = broker.dispatch(method, message["operation"], message["arguments"])
                    response = dict(sequence=message["sequence"], ok=True, result=value)
                except Exception as exc:
                    response = dict(sequence=message["sequence"], ok=False, error_type=type(exc).__name__, error=str(exc))
                try:
                    connection.send(response)
                except (BrokenPipeError, EOFError):
                    pass  # EOF is retained as a dead worker on the next iteration.
            for connection, method in list(connections.items()):
                if not processes[method].is_alive() and not connection.poll():
                    results[method] = dict(status="worker_crash", method=method)
                    broker.exited(method)
                    connection.close()
                    del connections[connection]
        for process in processes.values():
            process.join()
        return dict(results=results, not_started=[s.method for s in queued], actions=broker.snapshot(),
            ownership_journal=dict(root=str(broker.root), events=broker.sequence, sha256=broker.tip))
    finally:
        for connection in connections:
            connection.close()
        for signum, handler in old_signals.items():
            signal.signal(signum, handler)


def main(argv=None, *, control_factory=None, selection_validator=None):
    args = serial.parser().parse_args(argv)
    if sys.platform != "linux":
        raise RuntimeError("Source models/tests require an actual allocated Linux runtime")
    if args.resume:
        raise ValueError("Parallel-v1 requires fresh shards; interrupted ownership/full recoveries need owner reconciliation")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if source != args.source_commit or subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Require exact clean owner-deployed Git source")
    scope = serial.read_bound(args.scope, args.scope_sha256)
    allocation = serial.read_bound(args.allocation, args.allocation_sha256)
    profile_data = serial.read_bound(args.profile, args.profile_sha256)
    profile = serial.SourceProfile(**profile_data)
    interface = serial.validate_interface(serial.read_bound(args.interface, args.interface_sha256))
    selection = serial.read_bound(args.selection, args.selection_sha256)
    qualification = serial.read_reference(selection["qualification_receipt"])
    if profile.scope_sha256 != args.scope_sha256 or profile.source_selection_sha256 != args.selection_sha256:
        raise ValueError("Exact original scope and scientifically selected frontend required")
    # The current frontend is ineligible. Parallel execution never substitutes
    # scheduling/capacity for actual selected mapping and condition proofs.
    validator = serial.validate_selection if selection_validator is None else selection_validator
    validator(selection, qualification, serial.read_reference(selection["a1_result"]),
        profile, interface, interface_sha=args.interface_sha256,
        selection_rule=serial.read_reference(selection["selection_rule"]))
    validate_parallel_admission(scope, allocation, profile, source=source, hostname=socket.gethostname(),
        output=args.output, profile_sha=args.profile_sha256)
    cpu = serial.read_reference(allocation["CPU_qualification_receipt"])
    if cpu.get("status") != "passed" or cpu.get("source_commit") != source:
        raise ValueError("Actual allocated CPU qualification of this exact orchestration commit required")
    probe = json.loads(args.gpu_probe.read_text())
    if (probe.get("status") != "passed" or probe.get("hostname") != socket.gethostname()
            or len(probe.get("devices", [])) != 1 or not serial.torch.cuda.is_available()
            or serial.torch.cuda.device_count() != 1):
        raise ValueError("Retain qualified one-visible-GPU inference/renderer contract")
    if (serial.sha(args.model_path/"model.safetensors") != profile.motor_sha256
            or serial.sha(args.norm_stats) != interface["normalization_sha256"]
            or interface["joint_convention_sha256"] != args.joint_conventions_sha256):
        raise ValueError("Actual model/normalization/joint mapping differs")
    serial.verify_source_conventions(serial.read_bound(args.joint_conventions, args.joint_conventions_sha256),
        interface, dict(robotwin=args.robotwin_root, rlinf=args.rlinf_root))
    limits = serial.load_benchmark_episode_limits(args.robotwin_root, serial.TASKS)
    if limits[profile.task] != 400 or serial.sha(args.robotwin_root/"task_config/_eval_step_limit.yml") != profile.task_horizon_sha256:
        raise ValueError("Actual native benchmark horizon differs")
    from .steering_history_prefix import build_prefix
    student_profiles = {int(k): serial.read_reference(ref) for k, ref in allocation["student_profiles"].items()}
    if (set(student_profiles) != {12, 24} or any(p.get("prefix_episodes") != n or p.get("source_task") != profile.task
            or p.get("scope_sha256") != profile.scope_sha256
            or p.get("interface", {}).get("mapping_sha256") != profile.mapping_sha256
            or p.get("interface", {}).get("control_interface_sha256") != profile.interface_sha256
            for n, p in student_profiles.items())):
        raise ValueError("Bind exact original TRAIN/VAL and student prefix profiles")
    serial.read_reference(allocation["control_qualification"])
    args.output.mkdir(parents=True, exist_ok=False)
    stat = os.statvfs(args.output)
    if stat.f_bavail*stat.f_frsize < 16 << 30:
        raise RuntimeError("Parallel source setup requires actual16GiB free plus its bounded shared reservation")
    write_probe = args.output/".write-probe"
    with write_probe.open("xb") as target:
        target.write(b"parallel-source-write-fsync-probe\n")
        target.flush()
        os.fsync(target.fileno())
    write_probe.unlink()
    corpus = Path(allocation["source_corpus_root"])
    ledger = allocation["action_ledger"]
    with serial.CampaignBudget(ledger["root"], campaign_id=ledger["campaign_id"], total_actions=300000,
            source_caps=ledger["source_caps"], phase_caps=ledger["phase_caps"], contract=ledger["contract"],
            run_id=args.run_id, source_commit=source) as budget:
        profile.validate_budget(budget)
        if budget.snapshot()["unresolved_decisions"]:
            raise ValueError("Unknown/uncertain canonical intents prohibit worker launch")
        validate_capacity(serial.read_reference(allocation["capacity_qualification_receipt"]), allocation, profile,
            budget, source=source, hostname=socket.gethostname(), gpu_uuid=probe["devices"][0]["uuid"],
            frozen_fingerprint=qualification["frozen_after"])
        corpus.mkdir(parents=True, exist_ok=False)
        serial._atomic_json(corpus/"shards.json", dict(format=SHARDS, source_commit=source,
            profile_sha256=args.profile_sha256, action_ledger_root=str(budget.root.resolve()),
            shards=[dict(lineage=asdict(lineage), root=str(corpus/"shards"/lineage.method))
                    for lineage in serial.lineage_specs(profile.learner_seed)],
            role="Logical append-only phase index; each canonical root has one writer; whole lineage splits unchanged"))
        serial._atomic_json(args.output/"admission.json", dict(format=FORMAT, source_commit=source,
            allocation_sha256=args.allocation_sha256, profile_sha256=args.profile_sha256,
            selection_sha256=args.selection_sha256, gpu_probe_sha256=serial.sha(args.gpu_probe),
            max_concurrency=allocation["max_concurrency"], corpus_index=str(corpus/"shards.json")))
        state = supervise(args, allocation, profile, interface, qualification, budget, source,
            control_factory=control_factory)
        # Preserve all full-state references and the unfiltered canonical tip
        # BEFORE optional corpus export/qualification can fail. Never label a
        # synthetic or partially assembled index as student-admitted data.
        serial._atomic_json(args.output/"parallel-source-recovery.json", dict(state, format=FORMAT,
            source_commit=source, profile=profile.contract(), source_corpus_index=str(corpus/"shards.json"),
            role="Immutable per-worker full recoveries and canonical ownership; scientific admission pending"))
        prefixes = []
        for count in (12, 24):
            cohort = {}
            for method, result in state["results"].items():
                ref = result.get("publications", {}).get(str(count))
                if ref is not None and result.get("frozen_motor_unchanged") and result.get("control_closed"):
                    publication = serial.read_reference(ref)
                    if any(Path(raw["root"]).resolve() != (corpus/"shards"/method).resolve()
                           for raw in publication["complete_episode_manifests"]):
                        raise ValueError("Publication escaped its exact canonical lineage shard")
                    cohort[method] = publication
            if set(cohort) == set(SOURCE_METHOD_KEYS):
                path = args.output/"prefixes"/f"balanced-{count:03d}.json"
                build_prefix(student_profiles[count], cohort, control_qualification=allocation["control_qualification"],
                    output=path, expected_hostname=socket.gethostname())
                prefixes.append(dict(path=str(path), sha256=serial.sha(path)))
        complete = (set(state["results"]) == set(SOURCE_METHOD_KEYS)
            and all(r.get("status") == "initial_prefix_complete" and r.get("closed_physical_boundary")
                    and r.get("frozen_motor_unchanged") and r.get("control_closed")
                    for r in state["results"].values()) and not state["actions"]["unresolved_decisions"])
        serial._atomic_json(args.output/"parallel-source-state.json", dict(state, format=FORMAT,
            status="initial_prefix_complete" if complete else "incomplete_preserved",
            immutable_prefixes=prefixes, source_commit=source, profile=profile.contract(),
            source_corpus_index=str(corpus/"shards.json"),
            scientific_admission="Collection only. AD improvement, same-mapping TRAIN1 DPT teacher and ICRL gates unchanged"))


if __name__ == "__main__":
    main()
