#!/usr/bin/env python
"""Eight persistent official engines annotate the five fixed native50 banks.

Calibration only: no policy, optimizer, training exchange or simulator exists
here. Each per-task result requires all fifty labels and its frozen native bank.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from fluffy_bassoon.meta_rl.flora_terminal_worker import TerminalWorker
from fluffy_bassoon.meta_rl.flora_terminal_shards import baseline_episode_shard, gather_baseline_outputs
from fluffy_bassoon.meta_rl.rlinf_flora_ppo import FLORA_CODE, FLORA_MODEL, baseline_validation, digest, sha, write_new
from fluffy_bassoon.meta_rl.rlinf_flora_ppo_runtime import read_ref
from fluffy_bassoon.meta_rl.rlinf_flora_ppo_monitor import ProgressLog
from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import PPO_TASKS, protocol
from robocasa_residual_score import score


def require_time(profile):
    if time.time() >= profile['deadline_unix']-profile['closure_seconds']:
        raise TimeoutError('Finite calibration deadline; no partial per-task quality result')


def wait_for(paths, profile, progress):
    last_emit = 0.
    while True:
        require_time(profile)
        if any((Path(profile['exchange'])/'errors'/f'rank-{r:02d}.json').exists() for r in range(8)):
            raise RuntimeError('A peer scorer failed; preserve partial outputs without retry')
        if all(path.exists() for path in paths):
            return
        if time.monotonic()-last_emit >= 25:
            progress.emit(phase='waiting_for_frozen_native_bank_or_complete_labels')
            last_emit = time.monotonic()
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    args = parser.parse_args()
    p = json.loads(Path(args.profile).read_text())
    rank = p['annotation_rank']
    if (platform.system() != 'Linux' or p['mode'] != 'fifty_case_baseline_only'
            or p['annotation_world'] != 8 or type(rank) is not int or not 0 <= rank < 8
            or p['batch_size'] != 8 or p['gpu_memory_utilization'] != .78
            or [c['task'] for c in p['cohorts']] != list(PPO_TASKS)):
        raise ValueError('Exact five prospectively named PPO task banks and eight fixed baseline scorers required')
    source = Path(__file__).resolve().parents[2]
    if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip() != p['source_commit']:
        raise ValueError('Exact immutable annotation source required')
    read_ref(p['actual_gpu_probe'])
    import torch
    from fluffy_bassoon.baige_gpu import verify_selected_cuda
    selected = verify_selected_cuda(torch)
    if selected != p['expected_gpu_uuid'] or torch.cuda.device_count() != 1:
        raise ValueError('One verified allocated physical GPU per independent TP1 engine')
    model_receipt = read_ref(p['model_receipt'])
    if (model_receipt['completed'] is not True or model_receipt['model_revision'] != FLORA_MODEL
            or Path(model_receipt['model_directory']).resolve() != Path(p['model']).resolve()
            or subprocess.check_output(['git','-C',p['official_code'],'rev-parse','HEAD'],text=True).strip() != FLORA_CODE):
        raise ValueError('Unchanged official scorer code and model receipt required')
    output, exchange = Path(p['output']), Path(p['exchange'])
    output.mkdir(parents=True, exist_ok=False)
    keys = ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','NPROC_PER_NODE')
    write_new(output/'scheduler-node-environment.json', dict(gpu_uuid=selected,
        scheduler={key:os.environ[key] for key in keys if key in os.environ},
        visibility={key:os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES')}))
    for key in keys:os.environ.pop(key,None)
    progress = ProgressLog(output/'progress','F',contract_sha256=digest(p)).start()
    try:
        require_time(p)
        progress.emit(phase='baseline_only_official_engine_loading', measurement_only=True)
        cold = time.monotonic()
        worker = TerminalWorker(p['model'],batch_size=8,gpu_memory_utilization=.78)
        memory = subprocess.check_output(['nvidia-smi','--id='+selected,
            '--query-gpu=uuid,memory.used,memory.free,memory.total','--format=csv,noheader,nounits'],
            text=True,timeout=5).strip().split(',')
        if len(memory) != 4 or memory[0].strip() != selected:
            raise ValueError('Warmup memory snapshot must identify the allocated GPU')
        write_new(exchange/'engines'/f'rank-{rank:02d}.json',dict(completed=True,
            mode=p['mode'],annotation_rank=rank,source_commit=p['source_commit'],gpu_uuid=selected,
            model_revision=FLORA_MODEL,code_commit=FLORA_CODE,profile_sha256=digest(p),
            batch_size=8,tensor_parallel_size=1,gpu_memory_utilization=.78,
            cold_engine_seconds=time.monotonic()-cold,
            memory_snapshot=dict(used_mib=float(memory[1]),free_mib=float(memory[2]),total_mib=float(memory[3])),
            model_load_count=1,robot_actions=0))
        results = []
        for cohort in p['cohorts']:
            task, path = cohort['task'], Path(cohort['native_qualification'])
            wait_for([path],p,progress)
            native_sha = sha(path)
            if cohort.get('expected_native_sha256') and cohort['expected_native_sha256'] != native_sha:
                raise ValueError('A prebound native cohort receipt changed')
            native = json.loads(path.read_text())
            proto = protocol(task,baseline_count=50)
            if (native['task'] != task or native['count'] != 50 or native['protocol'] != proto
                    or native['optimizer_updates'] != 0):
                raise ValueError('Same complete native SFT fifty-case task bank required')
            for key in ('episodes','monitor'):
                expected = cohort.get('expected_'+key+'_sha256')
                if expected and native[key]['sha256'] != expected:
                    raise ValueError('A prebound native bank identity changed')
            episodes = read_ref(native['episodes'])
            subset = baseline_episode_shard(episodes,rank,proto)
            directory = exchange/'cohorts'/task
            target = output/(task+'.jsonl')
            progress.emit(phase='baseline_only_official_terminal_inference',task=task,
                          episodes_complete=len(subset))
            require_time(p)
            began = time.monotonic()
            rows, _, receipt = score(worker,subset,target,p['model_receipt'],proto)
            array = target.with_suffix('.json')
            write_new(array,rows)
            write_new(directory/'shards'/f'rank-{rank:02d}.json',dict(completed=True,
                annotation_rank=rank,native_receipt_sha256=native_sha,episodes=native['episodes'],
                outputs=dict(path=str(array),sha256=sha(array)),scorer_receipt=receipt,
                inference_seconds=time.monotonic()-began,model_load_count=worker.model_load_count))
            if rank == 0:
                paths = [directory/'shards'/f'rank-{r:02d}.json' for r in range(8)]
                wait_for(paths,p,progress)
                shards = {}
                for r, shard_path in enumerate(paths):
                    value = json.loads(shard_path.read_text())
                    if (value['completed'] is not True or value['annotation_rank'] != r
                            or value['native_receipt_sha256'] != native_sha
                            or value['episodes'] != native['episodes']
                            or value['scorer_receipt']['model_receipt'] != p['model_receipt']):
                        raise ValueError('Partial, stale or different-model baseline shard')
                    shards[r] = read_ref(value['outputs'])
                ordered, _ = gather_baseline_outputs(episodes,shards,proto)
                truth_rows = read_ref(native['monitor'])
                truth = {r['id']:r['simulator_final_success'] for r in truth_rows}
                if (len(truth_rows) != 50 or set(truth) != {e['id'] for e in episodes}
                        or any(type(v) is not bool for v in truth.values())
                        or sum(truth.values()) != native['successes'] or sha(path) != native_sha):
                    raise ValueError('Native scalar truth belongs to another or partial cohort')
                result = baseline_validation([dict(id=row['id'],expected_score=row['expected_score'],
                    simulator_final_success=truth[row['id']]) for row in ordered],proto)
                write_new(directory/'official-outputs.json',ordered)
                write_new(directory/'baseline-validation.json',dict(result,completed=True,
                    measurement_only=True,training_authorization=False,
                    native_receipt=dict(path=str(path),sha256=native_sha),native_episodes=native['episodes'],
                    native_monitor=native['monitor'],official_outputs=dict(path=str(directory/'official-outputs.json'),
                    sha256=sha(directory/'official-outputs.json')),
                    shards=[dict(path=str(path),sha256=sha(path)) for path in paths]))
                results.append(dict(task=task,path=str(directory/'baseline-validation.json'),
                                    sha256=sha(directory/'baseline-validation.json')))
            progress.emit(phase='baseline_task_shard_complete',task=task,labels_complete=len(rows))
        write_new(output/'FINAL.json',dict(completed=True,model_load_count=worker.model_load_count,
            completed_task_shards=5,measurement_only=True,robot_actions=0))
        if rank == 0:write_new(exchange/'ALL-FIVE.json',dict(completed=True,measurement_only=True,
            task_results=results,training_authorization=False,robot_actions=0))
    except BaseException as error:
        write_new(exchange/'errors'/f'rank-{rank:02d}.json',dict(error=f'{type(error).__name__}: {error}'))
        raise
    finally:progress.close()


if __name__ == '__main__':
    main()
