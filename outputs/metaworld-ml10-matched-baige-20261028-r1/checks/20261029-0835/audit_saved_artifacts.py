import collections, datetime, hashlib, io, json, math, os, pickle, socket, statistics, subprocess, zipfile
from pathlib import Path
from zoneinfo import ZoneInfo

assert socket.gethostname() == 'aibox-rffe453e3884-5f96c78b88-rtvhb'
root = Path('/mowenjie/fluffy-bassoon-2/outputs/metaworld-ml10-matched-baige-20261028-r1')
worktree = '/mowenjie/fluffy-bassoon-2/worktrees/metaworld-ml10-matched-baige-20261028-r1'
source = 'cd0128f03a060aff2d538a91f35058aa822df845'
assert subprocess.check_output(['git', '-C', worktree, 'rev-parse', 'HEAD'], text=True).strip() == source
assert not subprocess.check_output(['git', '-C', worktree, 'status', '--porcelain'], text=True).strip()
check = root / 'checks/20261029-0835'
check.mkdir(exist_ok=True)
jobs = dict(ctm='job-rczwk9dd1v1e', rl2='job-j400lp4d3m6m', ecet='job-wrlr0utzsayo', amago='job-dm64v27p0a30')
expected_hashes = dict(ctm='a34dff8decfef013bde1f74310c67709b5ae0689a2b1f89ed49848db51a58b18', rl2='a9634a8124ffb129211c00ea43d80028b3abdb2c6a97ddd8d0901d737bb03de3', ecet='824de988a0ac8c3c78aeb3d28d2d9e5ea77e919ce2a99697487a09bce331c80a', amago='e0ea9c8d542adfded6fdc656bd3fea5f3bfd21e1d75d5399ee8627026fd31532')

class Inert:
    def __init__(self, *args, **kwargs): self.args = args
    def __setstate__(self, state): self.state = state
class Metadata(pickle.Unpickler):
    def find_class(self, module, name):
        return collections.OrderedDict if (module, name) == ('collections', 'OrderedDict') else Inert
    def persistent_load(self, pid): return Inert(pid)

def finite(x):
    if isinstance(x, dict): return all(finite(v) for v in x.values())
    if isinstance(x, (list, tuple)): return all(finite(v) for v in x)
    return not isinstance(x, float) or math.isfinite(x)

records = {}
pool_hashes = None
families_by_split = {}
for method, job in jobs.items():
    b = root / method
    c = json.loads((b / 'train/config.json').read_text()); t = c['contract']
    assert t['source_commit'] == source and t['seed'] == 1000 and t['benchmark'] == 'ml10'
    assert (t['num_envs'], t['attempts'], t['horizon']) == (50, 5, 500)
    assert t['goal_visible'] and t['resample_instance']
    assert c['execution']['env_workers'] == 8 and c['execution']['allocated_cpus'] == 16
    assert c['target_updates'] == 80 and c.get('eval_every', t['evaluation'].get('eval_every')) == 5
    assert t['evaluation']['attempts'] == 25 and t['evaluation']['lanes'] == 10
    assert t['optimization']['replay_epochs' if method == 'amago' else 'epochs'] == 10
    if pool_hashes is None: pool_hashes = t['pool_hashes']
    assert pool_hashes == t['pool_hashes']
    p = json.loads((b / 'gpu-probe.json').read_text())
    assert p['status'] == 'passed' and p['hostname'] == job + '-master-0'
    assert len(p['devices']) == 1 and p['devices'][0]['cuda_name'] == 'NVIDIA RPBZZZ6'
    assert int((b / 'exit-code.txt').read_text()) == 0
    assert not list(b.rglob('*failure*.json'))
    launcher_seconds = int((b / 'allocation-active-seconds.txt').read_text())
    s = json.loads((b / 'train/summary.json').read_text())
    assert s['completed'] and s['iteration'] == 80 and s['source_commit'] == source
    assert not s.get('qualification_only', False)
    raw = (b / 'train/metrics.jsonl').read_text()
    assert raw.endswith('\n')
    rows = [json.loads(line) for line in raw.splitlines() if line]
    train = [r for r in rows if 'training' in r]
    assert [r['iteration'] for r in train] == list(range(1, 81))
    final = rows[-1]
    assert final['final'] and final['iteration'] == 80
    last_steps = last_hours = -1
    raw_evaluations = []; step_count = 0
    sample_count = opt_count = 0
    for r in rows:
        assert finite(r)
        assert r['environment_steps'] == r['iteration'] * 125000
        assert r['environment_steps'] >= last_steps and r['gpu_hours'] >= last_hours
        last_steps, last_hours = r['environment_steps'], r['gpu_hours']
        if 'training' in r:
            v = r['training']
            if method == 'amago':
                assert r['optimizer_steps'] == r['iteration'] * 125000 * 10 // 48000
                assert r['sampled_replay_transitions'] == r['optimizer_steps'] * 48000
            else:
                assert v['optimizer_steps'] == dict(ctm=50, rl2=20, ecet=78140)[method]
                assert v['forward_samples'] == (2500000 if method == 'ecet' else 1250000)
                if method == 'ecet': assert v['actor_steps'] == v['critic_steps'] == 39070
                sample_count += v['forward_samples']; opt_count += v['optimizer_steps']
        for split, conditions in r.get('evaluation', {}).items():
            for condition, v in conditions.items():
                fs = v['family_by_lane']; a = v['outcomes']; names = sorted(set(fs))
                assert len(names) == (10 if split == 'train_eval' else 5)
                assert len(fs) == len(a) == 10 and all(len(lane) == 25 for lane in a)
                assert all(type(x) is bool for lane in a for x in lane)
                assert all(fs.count(f) == (1 if split == 'train_eval' else 2) for f in names)
                if split not in families_by_split: families_by_split[split] = names
                assert names == families_by_split[split]
                family = {f: [statistics.mean(a[i][j] for i in range(10) if fs[i] == f) for j in range(25)] for f in names}
                curve = [statistics.mean(family[f][j] for f in names) for j in range(25)]
                assert family == v['per_family'] and abs(statistics.mean(curve) - v['success']) < 1e-12
                assert len(v['success_by_episode']) == 25 and all(abs(x-y) < 1e-12 for x,y in zip(curve, v['success_by_episode']))
                assert v['environment_steps'] == 125000 and v['trials'] == 10
                assert len(v['native_returns']) == 10 and all(len(x) == 25 for x in v['native_returns']) and finite(v['native_returns'])
                step_count += v['environment_steps']
                raw_evaluations.append(dict(iteration=r['iteration'], final=r.get('final', False), split=split, condition=condition, success=statistics.mean(curve), first5=statistics.mean(curve[:5]), family_success={f:statistics.mean(x) for f,x in family.items()}))
    assert len(raw_evaluations) == 40 and step_count == 5000000
    assert [r['iteration'] for r in rows if 'evaluation' in r and not r.get('final')] == list(range(0, 81, 5))
    for split in ('train_eval', 'test'):
        assert set(final['evaluation'][split]) == {'normal','memory_reset','reward_blind'}
        assert final['evaluation'][split]['normal'] == train[-1]['evaluation'][split]['normal']
    assert s['last_evaluation'] == final['evaluation']
    with (b / 'train/latest.pt').open('rb') as stream:
        h = hashlib.sha256()
        while chunk := stream.read(1 << 20): h.update(chunk)
        size = os.fstat(stream.fileno()).st_size
        stream.seek(0)
        with zipfile.ZipFile(stream) as z:
            entries = [n for n in z.namelist() if n.endswith('/data.pkl')]
            assert len(entries) == 1
            m = Metadata(io.BytesIO(z.read(entries[0]))).load()
    assert h.hexdigest() == s['checkpoint_sha256'] == expected_hashes[method]
    assert m['contract'] == t and m['iteration'] == 80 and m['version'] == 1 and m['boundary'] == ('between_complete_trials_and_updates' if method == 'amago' else 'between_trials')
    required = {'policy','optimizer','rng'} | ({'sampler','scheduler','scaler','replay','exploration_steps','exploration_numpy_rng'} if method == 'amago' else {'environment_rng'})
    assert required <= m.keys() and {'python','numpy','torch','cuda'} <= m['rng'].keys()
    assert m['counters'] == s['counters']
    n = m['counters']
    assert n['environment_steps'] == 10000000 and n['evaluation_steps'] == step_count
    if method == 'amago':
        opt_count = n['optimizer_steps']; sample_count = n['sampled_replay_transitions']
        assert opt_count == 2083 and sample_count == 99984000 and n['target_sampled_replay_transitions'] == 100000000
    else:
        assert n['optimizer_steps'] == opt_count == dict(ctm=4000, rl2=1600, ecet=6251200)[method]
        assert sample_count == (200000000 if method == 'ecet' else 100000000)
        timing_remainder = n['training_seconds'] - sum(r['timing']['rollout_seconds'] + r['timing']['update_seconds'] for r in train)
        assert 0 <= timing_remainder < 2.0, (method, timing_remainder)
    assert abs(final['training_gpu_hours'] * 3600 - n['training_seconds']) < 1e-6
    assert abs(final['gpu_hours'] * 3600 - n['cumulative_elapsed_seconds']) < 1.0
    assert n['training_seconds'] + n['evaluation_seconds'] <= n['cumulative_elapsed_seconds']
    assert 0 <= launcher_seconds - n['cumulative_elapsed_seconds'] < 60
    hard_cap = (12 if method == 'ecet' else 6) * 3600
    assert launcher_seconds <= hard_cap and s['cuda_peak_allocated_bytes'] > 0
    record = dict(training_timer_overhead_seconds=(None if method == 'amago' else timing_remainder), job=job, completed=True, iteration=80, environment_steps=n['environment_steps'], evaluation_steps=step_count, optimizer_steps=opt_count, sampled_or_forward_transitions=sample_count, learner_active_gpu_hours=final['gpu_hours'], training_only_gpu_hours=final['training_gpu_hours'], launcher_active_seconds=launcher_seconds, launcher_active_gpu_hours=launcher_seconds/3600, cuda_peak_allocated_bytes=s['cuda_peak_allocated_bytes'], parameters=c['parameters'], trainable_parameters=c.get('trainable_parameters',c['parameters']), checkpoint_sha256=h.hexdigest(), checkpoint_bytes=size, recovery_keys_verified=sorted(required), rng_keys_verified=sorted(m['rng']), checkpoint_counters_match_summary=True, qualification_source=(b/'qualification-source.txt').read_text().strip(), raw_conditions_recomputed=len(raw_evaluations), final_success={sp:{cond:v['success'] for cond,v in cs.items()} for sp,cs in final['evaluation'].items()}, raw_evaluations=raw_evaluations, normalization=t['normalization'], counter_summary=n)
    if method == 'ecet':
        assert set(m['optimizer']) == {'actor','critic'}
        record['optimizer_recovery'] = {k:dict(state_entries=len(v['state']),parameter_groups=len(v['param_groups'])) for k,v in m['optimizer'].items()}
        assert all(x['state_entries'] > 0 and x['parameter_groups'] > 0 for x in record['optimizer_recovery'].values())
    records[method] = record
    (check / (method + '-metrics.jsonl')).write_text(raw)
receipt = dict(check_time_shanghai=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), cluster='Baige', source=source, state='all four original methods complete and independently audited', shared_worktree_clean=True, common_pool_hashes=pool_hashes, families=families_by_split, inspection='Saved artifacts only; restricted inert metadata unpickler; no Torch import, tensor storage reads, model computation or tests.', records=records)
(check / 'completion-audit.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='records'},indent=2))
print(json.dumps({k:{a:b for a,b in v.items() if a not in ('raw_evaluations','counter_summary')} for k,v in records.items()},indent=2))
