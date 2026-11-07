"""Independent Linux CPU audit of a completed behavior-matched source run."""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, write
from fluffy_bassoon.meta_rl.tensor_archive import read_tensor
from motus_learner_reference import audit
import motus_teacher_entry  # Installs the pinned checkpoint helper search path.
from optimizer_semantics import semantic_digest
from profile_checkpoint import verify

BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')
SOURCES = dict(ppo='8740fdbe13fa7b0d1ae31e0e2da37aed1309d1fd',
    grpo='9596d7849330f768037a97dc1ab660f229569855')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('method',choices=tuple(SOURCES))
    args = parser.parse_args()
    assert os.uname().sysname == 'Linux'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and not torch.cuda.is_initialized()
    method = args.method
    root = BASE/f'motusbehavior{method}01'
    signals = BASE/f'motusbehavior{method}01-signals'
    state = root/'state'
    checked = {}
    def load(path):
        value = json.loads(path.read_bytes())
        checked[str(path)] = sha(path)
        return value
    assert load(signals/'EXIT.json')['exit_code'] == 0
    complete = load(signals/'COMPLETE.json')
    result = load(state/'RESULT.json')
    assert result['status'] == 'completed_admitted_prefix'
    assert result['algorithm'] == method and result['completed_iterations'] == result['intended_iterations'] == 2
    assert result['native_adam_updates_per_rank'] == 30
    assert load(signals/'resources.json')['sampled_reserve_passed']
    cfg = load(state/'resolved-native-config.json')
    assert cfg['native_pilot']['require_behavior_likelihood_match']
    assert cfg['actor']['micro_batch_size'] == 32 and int(cfg['native_pilot']['world_size']) == 8
    assert cfg['algorithm']['group_size'] == (1 if method == 'ppo' else 8)
    checkpoints = []
    signatures = {}
    started = time.monotonic()
    for step in range(3):
        receipt = load(state/'checkpoint-receipts'/f'step-{step:04d}.json')
        assert receipt['completed'] and receipt['global_step'] == step
        assert receipt['full_cold_readback'] and receipt['full_local_stageback']
        checkpoint = Path(receipt['cold'])
        assert sha(checkpoint/'MANIFEST.json') == receipt['manifest_sha256']
        manifest = load(checkpoint/'MANIFEST.json')
        assert manifest['global_step'] == step and manifest['source_commit'] == SOURCES[method]
        verify(checkpoint,manifest)
        assert all((checkpoint/'pilot'/f'{role}-{rank:02d}.pt').is_file()
            for role in ('actor','rollout','env') for rank in range(8))
        for rank in range(8):
            layout = load(checkpoint/'pilot'/f'motus-optimizer-layout-{rank:02d}.json')
            signature = load(checkpoint/'pilot'/f'motus-actor-signature-{rank:02d}.json')
            assert layout['rank'] == rank and layout['optimizer_steps'] == signature['optimizer_steps'] == 15*step
            assert semantic_digest(layout) == signature['optimizer_semantic_sha256']
            assert layout['optimizer_sha256'] == signature['optimizer_sha256']
            sampler = load(checkpoint/'pilot'/f'motus-case-schedule-env-{rank:02d}.json')
            assert sampler['next_reset_ordinal'] == 4*step
            assert sampler['manifest_sha256'] == cfg['native_pilot']['case_manifest_sha256']
            signatures[step,rank] = signature
        checkpoints.append(dict(step=step,path=str(checkpoint),manifest_sha256=sha(checkpoint/'MANIFEST.json'),
            files=len(manifest['files']),bytes=sum(v['bytes'] for v in manifest['files'].values())))
        print(f'{method}: cold checkpoint {step} and all8 strict Adam semantics passed',flush=True)
    reference = []
    for iteration in (1,2):
        for rank in range(8):
            receipt = load(state/'training-rollouts'/f'iteration-{iteration:04d}-policy-{iteration-1:04d}-rank-{rank:02d}.json')
            assert receipt['completed'] and not receipt['evaluation_data']
            assert receipt['rank'] == rank and receipt['collection_iteration'] == iteration
            assert receipt['producing_policy_iteration'] == iteration-1
            assert receipt['optimizer_steps_before'] == 15*(iteration-1)
            archive = Path(receipt['path'])
            assert archive.stat().st_size == receipt['bytes'] and sha(archive) == receipt['sha256']
            raw = read_tensor(archive)
            assert raw['optimizer_steps_before'] == 15*(iteration-1)
            assert raw['producing_policy_iteration'] == iteration-1 and raw['rank'] == rank
            independent = audit(raw['rollout_batch'],method)
            recorded = load(state/'learner-reference'/f'iteration-{iteration:04d}-rank-{rank:02d}.json')
            assert independent['passed'] and independent == recorded
            reference.append(dict(iteration=iteration,rank=rank,**independent))
            del raw
            measured = load(state/'actor-memory'/f'iteration-{iteration:04d}-rank-{rank:02d}.json')
            assert measured['micro_batch'] == 32 and measured['accumulation'] == 8 and measured['global_batch'] == 2048
            frozen = load(state/'actor-invariants'/f'rank-{rank:02d}-step-{iteration-1}-after.json')
            assert frozen['unchanged'] and frozen['before'] == frozen['after']
        print(f'{method}: independent NumPy whole-batch {iteration} passed all8 actors',flush=True)
    coverage = []
    for ordinal in range(8):
        for rank in range(8):
            row = load(state/'group-resets'/f'rank-{rank:02d}-reset-{ordinal:04d}.json')
            assert row['passed'] and row['raw_public_reference_verified']
            assert row['reset_ordinal'] == ordinal and row['rank'] == rank
            assert row['manifest_sha256'] == cfg['native_pilot']['case_manifest_sha256']
            groups = row['groups']
            assert len(groups) == (32 if method == 'ppo' else 4)
            assert all(g['passed'] and g['replica_count'] == (1 if method == 'ppo' else 8) for g in groups)
            coverage.extend(dict(iteration=ordinal//4,seed=g['scene_seed'],replicas=g['replica_count']) for g in groups)
    for rank in range(8):
        replay = load(state/'first-replay'/f'rank-{rank:02d}.json')
        assert replay['passed'] and replay['logprob_max_abs'] == 0 and replay['optimizer_steps'] == 0 and replay['micro_batch'] == 32
    dev = [load(state/'dev'/f'iteration-{step:04d}.json') for step in range(3)]
    assert all(v['completed'] and v['episodes'] == 16 and v['optimizer_updates_during_eval'] == 0 for v in dev)
    changes = {str(step):[rank for rank in range(8) if signatures[step,rank]['trainable_parameters_sha256']
        != signatures[step-1,rank]['trainable_parameters_sha256']] for step in (1,2)}
    assert all(changes[str(step)] for step in (1,2)), 'No trainable shard changed in an update'
    output = BASE/f'motusbehavior{method}01-readback01'
    output.mkdir()
    write(output/'RESULT.json',dict(status='passed_independent_behavior_matched_training_readback',
        source_commit=os.environ['TASK_SOURCE'],training_source_commit=SOURCES[method],algorithm=method,
        completed_iterations=2,Adam_updates_per_rank=30,full_native_checkpoints=checkpoints,
        strict_named_Adam_semantics_passed_all24=True,all24_worker_sidecars_present_each_boundary=True,
        both_fresh_collections_independent_reference=reference,raw_episode_count=2048,
        source_unique_scenes_by_collection={str(i):len({r['seed'] for r in coverage if r['iteration']==i}) for i in (0,1)},
        changed_trainable_shards_by_iteration=changes,original_behavior_match_required_each_collection=True,
        frozen_VLM_unchanged=True,all64_group_resets_verified=True,
        small_progress_panel_success=[v['metrics']['success_once'] for v in dev],
        larger_paired_evaluation_required=True,learning_gain_claim=False,gpu_seconds=0,
        seconds=time.monotonic()-started,checked_json_hashes=checked,complete_receipt=complete))
    print(f'{method}: PASSED {output}/RESULT.json',flush=True)


if __name__ == '__main__':main()
