"""Declarative target request only; no outcomes, model import or reservation."""
import json
from pathlib import Path

METHODS = ('native', 'ad_full', 'ad_static', 'dpt_full', 'dpt_static')
MEMORY = ('ad_full', 'ad_episode_reset', 'dpt_full', 'dpt_episode_reset')
CASES = []
for task in range(90):
    case = dict(benchmark='libero90', case_id=f'libero90:{task:02d}/state-0/seed-123',
        suite='libero_90', task_id=task, initial_state=0, scene_seed=123)
    groups = [dict(phase='A', method=m, support_budget=0, episodes=1, maximum_actions=410) for m in METHODS]
    if task % 10 == 0:
        groups += [dict(phase='B', method=m, support_budget=1, episodes=2, maximum_actions=820) for m in MEMORY]
    CASES.append(dict(case=case, groups=groups))
WORKERS = []
for worker in range(6):
    cases = CASES[worker::6]
    episodes = sum(g['episodes'] for c in cases for g in c['groups'])
    actions = sum(g['maximum_actions'] for c in cases for g in c['groups'])
    WORKERS.append(dict(worker=worker, task_ids=[c['case']['task_id'] for c in cases],
        physical_episodes=episodes, nominal_actions=actions, action_cap=actions+900,
        maximum_gpu_seconds=2700, initial_slice_seconds=2640, gpu_count=1, cpu_cores=16,
        memory_gib=96, shared_memory_gib=8))
assert sum(w['physical_episodes'] for w in WORKERS) == 522
assert sum(w['nominal_actions'] for w in WORKERS) == 214020
assert sum(w['action_cap'] for w in WORKERS) == 219420
REQUEST = dict(format='source40_interim100_L90_target_request_v1', status='UNFUNDED_UNLAUNCHED',
    execution_binding_status='Requires additive interim100-libero90 profile in existing firstwave wrapper; core reused unchanged',
    producer_source='2bc15da3028ab6a3cbaed4b813f3f761a680f1d9',
    producer_run='libero40-source-topup-interim100-20261031-r1', updates=100, seed=0,
    manifest='/mowenjie/fluffy-bassoon-2/outputs/libero40-source-pipeline-cpu-20261031-r2/assembly/topup-inputs.json',
    manifest_sha256='6563d1fc76ba506cf9f44cb648c30455313a33602ab1154e30b37cc54bc54883',
    assembly_file_sha256='89d28ad329b612ba52a4bcf108150a335de378a0e0b209e73b7bd5f47a347160',
    matrix='/mowenjie/fluffy-bassoon-2/outputs/libero40-source-topup-interim100-20261031-r1/experiment/scales/update-000100/index.json',
    matrix_sha256=None, matrix_hash_status='Bind after all four exact100 exports/recoveries complete; no target selection',
    cases=CASES, workers=WORKERS, total_physical_episodes=522, nominal_actions=214020,
    maximum_total_actions=220000, maximum_gpu_seconds=16200, cumulative_source_actions=457851,
    new_source_training_actions=0, target_optimizer_updates=0, source_campaign_immutable=True,
    full1M_500_phase_complete=False, plus_pro_fullsuite_request='Separate; official complete inventory and horizons required; no development-panel substitution')
if __name__ == '__main__':
    path = Path(__file__).with_name('request.json')
    with path.open('x') as f:
        json.dump(REQUEST, f, indent=2, allow_nan=False)
        f.write('\n')
