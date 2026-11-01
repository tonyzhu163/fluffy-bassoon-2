#!/usr/bin/env python
"""Persistent native8 calibration followed by the F-only PPO reward service.

Uses the pinned official engine and unchanged graded terminal reward.
Training accepts no simulator-monitor path. It never constructs a native env.
"""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

from fluffy_bassoon.meta_rl.flora_terminal_worker import TerminalWorker
from fluffy_bassoon.meta_rl.rlinf_flora_ppo import (
    FLORA_CODE, FLORA_MODEL, baseline_validation, digest, episode_video_refs, flora_jobs, parse_outputs,
    sha, video_refs_digest, write_new,
)
from fluffy_bassoon.meta_rl.rlinf_flora_ppo_monitor import ProgressLog
from fluffy_bassoon.meta_rl.rlinf_flora_ppo_runtime import read_ref
from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import protocol, require_calibration_permission
from fluffy_bassoon.meta_rl.flora_terminal_shards import gather_terminal_outputs, shard_episode_bank


def score(worker, episodes, output, model_receipt, proto):
    jobs = flora_jobs(episodes, proto)
    for episode in episodes:
        for ref in episode_video_refs(episode):
            if sha(ref["path"]) != ref["sha256"]:
                raise ValueError("Video bytes changed")
    with output.with_suffix(".jobs.jsonl").open("x") as stream:
        for row in jobs:
            stream.write(json.dumps(row) + "\n")
    rows = worker.score_jobs(jobs, output)
    labels = parse_outputs(episodes, rows, proto)
    receipt = dict(code_commit=FLORA_CODE, model_revision=FLORA_MODEL, model_receipt=model_receipt,
        completed=True, jobs_sha256=digest(jobs), video_refs_sha256=video_refs_digest(episodes),
        output=dict(path=str(output), sha256=sha(output)), training_reward_rule="(expected_score-1)/4",
        engine_process_pid=os.getpid(), model_load_count=worker.model_load_count,
        completed_batches=worker.completed_batches, engine="official persistent terminal_reward helpers",
        batch_size=worker.batch_size, gpu_memory_utilization=worker.gpu_memory_utilization)
    write_new(output.with_suffix(".receipt.json"), receipt)
    return rows, labels, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--episodes")
    parser.add_argument("--baseline-monitor")
    args = parser.parse_args()
    if platform.system() != "Linux":
        raise RuntimeError("Scoring only in an allocated Linux GPU job")
    profile = json.loads(Path(args.profile).read_text())
    if profile["mode"] not in ("baseline", "train", "integrated") or not 0 < profile["scheduler_seconds"] < 8 * 3600:
        raise ValueError("Exact finite calibration or F-only service required")
    read_ref(profile["actual_gpu_probe"])
    selected_uuid = None
    if os.environ.get("LYCO_BAIGE_GPU_MAP"):
        import torch
        from fluffy_bassoon.baige_gpu import verify_selected_cuda
        selected_uuid = verify_selected_cuda(torch)
    model_receipt = read_ref(profile["model_receipt"])
    model, code = Path(profile["model"]), Path(profile["official_code"])
    head = subprocess.check_output(["git", "-C", str(code), "rev-parse", "HEAD"], text=True).strip()
    if (head != FLORA_CODE or model_receipt["model_revision"] != FLORA_MODEL
            or not model_receipt["completed"] or Path(model_receipt["model_directory"]).resolve() != model.resolve()):
        raise ValueError("Same qualified official Flora code, weights and local receipt required")
    baseline_count = profile.get('baseline_count', 8)
    if baseline_count != 8 and profile['mode'] != 'baseline':
        raise ValueError('The fifty-case extension is baseline measurement only, not a training migration')
    proto = protocol(profile["task"], baseline_count=baseline_count)
    output = Path(profile["output"])
    output.mkdir(parents=True, exist_ok=False)
    world, rank = profile.get('annotation_world', 1), profile.get('annotation_rank', 0)
    if (type(world) is not int or world not in (1, 8) or type(rank) is not int or not 0 <= rank < world
            or world == 8 and (profile['mode'] != 'train' or selected_uuid is None
                              or profile.get('batch_size') not in (1, 2, 4, 6, 8))):
        raise ValueError('Eight fixed TP1 scorers require train-only profiles and a measured batch size<=8')
    if world == 8:
        required = profile.get('minimum_free_mib_after_engine_warmup')
        if (isinstance(required, bool) or not isinstance(required, (int, float))
                or not math.isfinite(required) or required <= 0
                or not isinstance(profile.get('engine_ready_path'), str)
                or not Path(profile['engine_ready_path']).is_absolute()):
            raise ValueError('Bind a positive measured headroom and fresh absolute engine-ready path before model loading')
        source = Path(__file__).resolve().parents[2]
        if (subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
                != profile['source_commit'] or selected_uuid != profile['expected_gpu_uuid']):
            raise ValueError('Exact production source and allocated scorer GPU identity required')
        # Validation above still sees the real node's scheduler identity.
        # Only this independent TP1 child's model-rendezvous variables change.
        keys = ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE',
                'MASTER_ADDR', 'MASTER_PORT', 'NPROC_PER_NODE')
        write_new(output / 'scheduler-node-environment.json', dict(verified_gpu_uuid=selected_uuid,
            scheduler_node_environment={key: os.environ[key] for key in keys if key in os.environ},
            preserved_visibility={key: os.environ.get(key) for key in
                ('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES')}, independent_tensor_parallel_size=1))
        for key in keys:
            os.environ.pop(key, None)
    progress = ProgressLog(output / "progress", "F", contract_sha256=digest(profile)).start()
    try:
        offload_ref = None
        if world == 8:
            gate = read_ref(profile['baseline_validation'])
            require_calibration_permission(gate, proto,
                calibration_sha256=profile['baseline_validation']['sha256'],
                authorization=read_ref(profile['calibration_exception']) if profile.get('calibration_exception') else None)
            offload_path = Path(profile['F_output']) / 'controller-motor-offload.json'
            while not offload_path.exists():
                if time.time() >= profile['deadline_unix'] - profile['closure_seconds']:
                    raise TimeoutError('No controller CPUoffload proof before finite scorer startup')
                progress.emit(phase='waiting_for_controller_motor_offload')
                time.sleep(1)
            offload = json.loads(offload_path.read_text())
            if (offload.get('completed') is not True or offload.get('device') != 'cpu'
                    or offload.get('predictions') != 'allocated rollout workers only'):
                raise ValueError('The duplicate controller motor must be off GPU before engine warmup')
            offload_ref = dict(path=str(offload_path), sha256=sha(offload_path))
            write_new(output / 'controller-offload-binding.json', offload_ref)
        progress.emit(phase="official_persistent_engine_loading", model_load_count=0)
        engine_start = time.monotonic()
        worker = TerminalWorker(model, batch_size=profile.get('batch_size', 1),
                                gpu_memory_utilization=profile.get('gpu_memory_utilization', .78))
        if world == 8:
            row = subprocess.check_output(['nvidia-smi', '--id=' + selected_uuid,
                '--query-gpu=uuid,memory.used,memory.free,memory.total', '--format=csv,noheader,nounits'],
                text=True, timeout=5).strip().split(',')
            if len(row) != 4 or row[0].strip() != selected_uuid:
                raise ValueError('Postwarmup memory must identify the actual selected card')
            used, free, total = map(float, row[1:])
            required = profile['minimum_free_mib_after_engine_warmup']
            if (not all(math.isfinite(value) for value in (used, free, total))
                    or free < required):
                raise RuntimeError('Measured scorer warmup leaves insufficient declared policy/workspace headroom')
            write_new(profile['engine_ready_path'], dict(format='robocasa_flora_tp1_engine_ready_v1',
                completed=True, annotation_rank=rank, annotation_world=8,
                source_commit=profile['source_commit'], code_commit=FLORA_CODE, model_revision=FLORA_MODEL,
                model_receipt=profile['model_receipt'], profile_sha256=digest(profile),
                gpu_probe=profile['actual_gpu_probe'], gpu_uuid=selected_uuid, engine_process_pid=os.getpid(),
                tensor_parallel_size=1, batch_size=worker.batch_size,
                gpu_memory_utilization=worker.gpu_memory_utilization,
                controller_offload=offload_ref, engine_warmup_seconds=time.monotonic() - engine_start,
                memory=dict(used_mib=used, free_mib=free, total_mib=total,
                    minimum_free_required_mib=required, scope='postwarmup snapshot, not lifetime peak')))
        progress.emit(phase="official_persistent_engine_ready", model_load_count=1,
                      engine_process_pid=os.getpid())
        serve(profile, args, worker, output, proto, progress)
    finally:
        progress.close()


def serve(profile, args, worker, output, proto, progress):
    if profile["mode"] in ("baseline", "integrated"):
        if not args.episodes or not args.baseline_monitor:
            raise ValueError("Freeze the native episode bank before paired calibration")
        episodes = json.loads(Path(args.episodes).read_text())
        if len(episodes) != proto['baseline_count'] or sha(args.episodes) != profile["frozen_native_episodes_sha256"]:
            raise ValueError("Exactly the prospectively frozen native bank, no task selection by Flora")
        progress.emit(phase="native_Flora_calibration", episodes_complete=len(episodes))
        rows, _, receipt = score(worker, episodes, output / "terminal.jsonl", profile["model_receipt"], proto)
        truth = {r["id"]: r["simulator_final_success"] for path in sorted(Path(args.baseline_monitor).glob("*.json"))
                 for r in [json.loads(path.read_text())]}
        if set(truth) != {episode["id"] for episode in episodes}:
            raise ValueError("Calibration video/truth identity mismatch")
        result = baseline_validation([dict(id=row["id"], expected_score=row["expected_score"],
            simulator_final_success=truth[row["id"]]) for row in rows], proto)
        write_new(output / "baseline-validation.json", dict(result, completed=True, scorer_receipt=receipt))
        progress.emit(phase="calibration_passed" if result["go"] else "calibration_failed_comparison_stopped",
                      labels_complete=len(episodes), AUROC=result["AUROC"])
        if profile["mode"] == "baseline" or not result["go"]:
            return
        # Calibration truth is used only above. Neither the training requests
        # nor the F policy receive it; the loaded engine is the same object.
        del truth, episodes, rows, result
    elif args.baseline_monitor or args.episodes:
        raise ValueError("F training scorer cannot receive simulator truth or another case bank")
    gate = (json.loads((output / "baseline-validation.json").read_text())
            if profile["mode"] == "integrated" else read_ref(profile["baseline_validation"]))
    gate_ref = profile.get('baseline_validation')
    require_calibration_permission(gate, proto,
        calibration_sha256=gate_ref['sha256'] if gate_ref else sha(output / 'baseline-validation.json'),
        authorization=read_ref(profile['calibration_exception']) if profile.get('calibration_exception') else None)
    maximum = profile["maximum_batches"]
    if type(maximum) is not int or not 1 <= maximum <= 10:
        raise ValueError("At most ten admitted F batches")
    seen = set()
    exchange = Path(profile["F_output"])
    while time.time() < profile["deadline_unix"] - profile["closure_seconds"] and len(seen) < maximum:
        for path in sorted(exchange.glob("labels-*/request.json")):
            if str(path) in seen:
                continue
            if (path.parent / "ready.json").exists():
                seen.add(str(path))
                continue
            request = json.loads(path.read_text())
            pending = request["pending_checkpoint"]
            # The complete producing-policy/critic/Adam/RNG+rollout checkpoint
            # was saved before the request. No simulator labels are read here.
            if sha(pending["path"]) != pending["sha256"] or request["protocol"] != proto:
                raise ValueError("Pending batch bytes or native task changed")
            jobs = flora_jobs(request["episodes"], proto)
            if request["jobs"] != jobs or request["jobs_sha256"] != digest(jobs):
                raise ValueError("Fresh batch video identity mismatch")
            if profile.get('annotation_world') == 8:
                serve_shard(profile, worker, output, proto, progress, path, request)
                seen.add(str(path))
                continue
            target = output / (path.parent.name + ".jsonl")
            progress.emit(phase="official_Flora_terminal_inference", iteration=len(seen) + 1,
                          episodes_complete=len(request["episodes"]))
            rows, _, _ = score(worker, request["episodes"], target, profile["model_receipt"], proto)
            array = target.with_suffix(".json")
            write_new(array, rows)
            write_new(path.parent / "ready.json", dict(pending_checkpoint_sha256=pending["sha256"],
                jobs_sha256=digest(jobs), code_commit=FLORA_CODE, model_revision=FLORA_MODEL,
                outputs=dict(path=str(array), sha256=sha(array))))
            seen.add(str(path))
            progress.emit(phase="complete_batch_labels_ready", labels_complete=len(rows),
                          model_load_count=worker.model_load_count, completed_batches=worker.completed_batches)
        if (exchange / "FINAL.json").exists():
            break
        if Path(profile.get('shutdown_file', output / 'shutdown.json')).exists():
            break
        time.sleep(1)


def serve_shard(profile, worker, output, proto, progress, path, request):
    """Eight persistent engines; same original all64 ready.json boundary."""
    rank = profile['annotation_rank']
    episodes = request['episodes']
    subset = shard_episode_bank(episodes, rank, 8, proto)
    request_sha, pending = sha(path), request['pending_checkpoint']
    directory = path.parent / 'shards'
    ready_path = directory / f'rank-{rank:02d}.json'
    if not ready_path.exists():
        target = output / (path.parent.name + '.jsonl')
        progress.emit(phase='official_Flora_shard_inference', episodes_complete=8,
                      annotation_rank=rank, annotation_world=8)
        rows, _, receipt = score(worker, subset, target, profile['model_receipt'], proto)
        array = target.with_suffix('.json')
        write_new(array, rows)
        write_new(ready_path, dict(completed=True, annotation_rank=rank, annotation_world=8,
            request_sha256=request_sha, pending_checkpoint_sha256=pending['sha256'],
            jobs_sha256=request['jobs_sha256'], code_commit=FLORA_CODE, model_revision=FLORA_MODEL,
            episode_ids=[row['id'] for row in subset], scorer_receipt=receipt,
            outputs=dict(path=str(array), sha256=sha(array))))
    if rank == 0:
        shard_paths = [directory / f'rank-{index:02d}.json' for index in range(8)]
        while not all(row.exists() for row in shard_paths):
            if time.time() >= profile['deadline_unix'] - profile['closure_seconds']:
                raise TimeoutError('Whole pending batch remains saved; not all64 Flora labels completed')
            progress.emit(phase='waiting_for_all_eight_Flora_shards')
            time.sleep(1)
        shards = {}
        for index, shard_path in enumerate(shard_paths):
            value = json.loads(shard_path.read_text())
            expected = shard_episode_bank(episodes, index, 8, proto)
            if (value.get('completed') is not True or value['annotation_rank'] != index
                    or value['annotation_world'] != 8 or value['request_sha256'] != request_sha
                    or value['pending_checkpoint_sha256'] != pending['sha256']
                    or value['jobs_sha256'] != request['jobs_sha256']
                    or value['code_commit'] != FLORA_CODE or value['model_revision'] != FLORA_MODEL
                    or value['episode_ids'] != [row['id'] for row in expected]
                    or value['scorer_receipt']['model_receipt'] != profile['model_receipt']):
                raise ValueError('Stale, partial or different-model Flora shard cannot release GAE')
            shards[index] = read_ref(value['outputs'])
        if sha(path) != request_sha or sha(pending['path']) != pending['sha256']:
            raise ValueError('Producing batch changed while annotation was pending')
        rows, _ = gather_terminal_outputs(episodes, shards, proto)
        array = path.parent / 'flora-terminal-outputs.json'
        if array.exists():
            if json.loads(array.read_text()) != rows:
                raise ValueError('An interrupted aggregate cannot be overwritten with different labels')
        else:
            write_new(array, rows)
        write_new(path.parent / 'ready.json', dict(pending_checkpoint_sha256=pending['sha256'],
            jobs_sha256=request['jobs_sha256'], code_commit=FLORA_CODE, model_revision=FLORA_MODEL,
            outputs=dict(path=str(array), sha256=sha(array)),
            shard_receipts=[dict(path=str(row), sha256=sha(row)) for row in shard_paths]))
        progress.emit(phase='complete_batch_labels_ready', labels_complete=64,
                      model_load_count=worker.model_load_count, completed_batches=worker.completed_batches)


if __name__ == "__main__":
    main()
