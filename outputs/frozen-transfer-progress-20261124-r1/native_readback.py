"""Post-run evidence for the two immutable gpu01 shards. Requires the pinned study on PYTHONPATH."""
import argparse
from collections import Counter
import ctypes
from itertools import combinations_with_replacement
import hashlib
import json
from math import ceil, factorial, prod
import os
from pathlib import Path
import sys

import numpy as np

from contract import IDENTITY, MODES, TASKS, source_law
from intervention import alignment
import readback as pinned


PIN = "d70e206a00c6c5d59f190ec4a9b0bc6df2ec82c4"
ARMS = ("IIWA", "Kinova3")
CONTRASTS = (("progress", "raw"), ("progress", "tuned"), ("progress", "fixed"), ("fixed", "tuned"))


def check(condition, context):
    if not condition:
        raise ValueError(str(context))


def read(path):
    return json.loads(path.read_bytes())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def exact(left, right, dtype, context):
    a = np.asarray(left, dtype=dtype)
    b = np.asarray(right, dtype=dtype)
    check(a.shape == b.shape and a.tobytes() == b.tobytes(), context)
    return a.size


def blas_info():
    library, = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                if '/numpy.libs/libopenblas' in line}
    blas = ctypes.CDLL(library)

    def query(name):
        func = getattr(blas, name + '64_')
        func.restype = ctypes.c_char_p
        return func().decode()

    return dict(kernel=query('openblas_get_corename'), config=query('openblas_get_config'),
                library=library, numpy=np.__version__, python=sys.version, core=sorted(os.sched_getaffinity(0)))


def kernel_comparison(root):
    """Recompute using each run's actual output plan to isolate the command transform."""
    counts = np.zeros(7, np.int64)
    maximum = np.zeros(7)
    controls = 0
    first = None
    for path in sorted(root.glob('gpu01-*/episodes/*.json')):
        episode = read(path)
        for k, block in enumerate(episode['chunks']):
            actual = np.asarray(block['sent'], np.float32)
            proposal = np.asarray(block['output'], np.float32)[:len(actual)]
            ref = pinned.native_commands(proposal, episode['native']['inferred'])
            mismatch = ref.view('u4') != actual.view('u4')
            counts += mismatch.sum(axis=0)
            maximum = np.maximum(maximum, np.abs(ref.astype(float) - actual).max(axis=0))
            if first is None and mismatch.any():
                i, col = np.argwhere(mismatch)[0]
                first = dict(file=str(path.relative_to(root)), block=k, offset=int(i), coordinate=int(col),
                             observed=float(actual[i, col]), replay=float(ref[i, col]))
            controls += len(actual)
    return dict(blas=blas_info(), controls=controls, different_coordinate_counts=counts.tolist(),
                max_abs_by_coordinate=maximum.tolist(), first_bit_difference=first)


def explicit_commands(output, correction):
    # Two float32 products and their rounded sum; keep them separate from matrix dispatch.
    cmd = np.clip(output, -1, 1).copy()
    if correction == IDENTITY:
        return cmd
    x, y = cmd[:, 0].copy(), cmd[:, 1].copy()
    rot = pinned.rotation(-correction['yaw_deg'])
    inv = np.float32(1 / float(np.clip(correction['xy_scale'], .25, 4)))
    for index in range(2):
        cmd[:, index] = (x * rot[index, 0] + y * rot[index, 1]) * inv
    cmd[:, 2] *= inv
    cmd[:, 6] *= np.float32(correction['gripper_sign'])
    return np.clip(cmd, -1, 1)


def episode_audit(obj, law, totals):
    rec, tape = obj['native'], obj['chunks']
    mode, c = obj['mode'], rec['inferred']
    n, reset = rec['controls'], rec['reset']
    # Each observed success was first attained on the last executed control.
    reward = rec['native_rewards']
    check(len(reward) == n and all(v == 0 for v in reward[:-1]) and reward[-1] in (0, 1), 'reward tape')
    success = bool(reward[-1])
    check(rec['native_return'] == int(success) and rec['success'] is success and
          rec['terminated'] is success and rec['truncated'] is (not success) and
          (success or n == reset['horizon']), 'reward / endpoint / horizon')
    check(obj['noise_shape'] == [10, 32] and tape[0]['policy_input'] == reset['motor_frame0'], 'opening/dims')
    h, w = obj['noise_shape']
    noise = np.random.default_rng(reset['noise_seed']).standard_normal((ceil(reset['horizon']/5), h, w)).astype('f4')
    check(pinned.fp(noise) == reset['noise_sha256'], 'bank')
    phases = []
    prefix = []
    for k, block in enumerate(tape):
        original = np.asarray(block['original'], np.float32)
        plan = original.copy()
        expected_phase = None
        if k and mode in ('fixed', 'progress'):
            prev = tape[k-1]
            donor = np.asarray(prev['original'], np.float32)
            expected_phase = 5
            if mode == 'progress':
                carry = np.asarray(tape[k-2]['sent'], np.float32) if k > 1 else np.zeros((5, 7), np.float32)
                expected_phase, info = alignment(donor, prev['before'], prev['tcp'][-1], carry, c, law)
                distances = block['alignment']['distances_m']
                check(expected_phase == min(range(6), key=lambda j: (distances[j], -j)), 'recorded nearest index')
                check(info.keys() == block['alignment'].keys(), 'phase fields')
                for key, value in info.items():
                    dtype = np.float32 if key == 'donor_issued_xyz' else np.float64
                    totals['exact_alignment_scalars'] += exact(value, block['alignment'][key], dtype, (k, key))
            phases.append(expected_phase)
            old = np.clip(donor[expected_phase:expected_phase+5, :3], -1, 1)
            plan[:5, :3] = np.float32(.5) * (np.clip(original[:5, :3], -1, 1) + old)
        check(block['j'] == expected_phase and
              (block['alignment'] is not None) == bool(k and mode == 'progress'), 'actual phase')
        check(original.shape == (h, 7) and pinned.fp(noise[k]) == block['noise_sha256'], 'chunk shape/noise')
        totals['exact_full_plan_scalars'] += exact(plan, block['output'], np.float32, (k, 'canonical'))
        count = min(5, n-5*k)
        totals['exact_sent_scalars'] += exact(explicit_commands(plan[:count], c), block['sent'],
                                              np.float32, (k, 'sent'))
        if k < 6:
            first, last = (np.asarray(v, np.float32) for v in (block['before'], block['tcp'][-1]))
            prefix.append(float(np.linalg.norm((last-first)[:2]) * 1000))
    check(round(sum(prefix[:3]), 3) == rec['prefix_xy_disp_15_mm'] and
          round(sum(prefix), 3) == rec['prefix_xy_disp_30_mm'], 'native motion summaries')
    totals.update(dict(episodes=1, controls=n, proposals=len(tape), successes=int(success),
                       truncated=int(not success)))
    return dict(success=success, native_return=sum(reward), terminated=rec['terminated'], truncated=rec['truncated'],
                controls=n, chunks=len(tape), opened_on_attempt=obj['accepted_create'], seconds=rec['seconds'],
                eligible_controls=(n-5 if phases else 0), phase_histogram=dict(Counter(phases)))


def endpoints(groups):
    # Enumerate the 6,435 unordered samples and count their ordered multiplicities independently.
    size = len(TASKS)
    samples = [(ids, factorial(size)//prod(factorial(v) for v in Counter(ids).values()))
               for ids in combinations_with_replacement(range(size), size)]
    check(sum(w for _, w in samples) == size**size, 'bootstrap mass')
    rows = [v for group in groups for v in group]
    result = dict(cases=len(rows), successes={m: sum(r[m] for r in rows) for m in MODES}, contrasts={})
    for a, b in CONTRASTS:
        ds = [int(r[a])-int(r[b]) for r in rows]
        paired = [sum(int(g[t][a])-int(g[t][b]) for g in groups) for t in range(size)]
        dist = Counter()
        for ids, w in samples:
            dist[sum(paired[i] for i in ids)] += w
        bounds = []
        for cutoff in (1, 39):
            mass = 0
            for delta, count in sorted(dist.items()):
                mass += count
                if 40*mass >= cutoff*size**size:
                    bounds.append(100*delta/len(rows))
                    break
        result['contrasts'][f'{a}-minus-{b}'] = dict(net_pp=100*sum(ds)/len(rows), wins=ds.count(1),
            losses=ds.count(-1), ties=ds.count(0), rescue_denominator=sum(not r[b] for r in rows),
            harm_denominator=sum(r[b] for r in rows), task_bootstrap_ci95_pp=bounds)
    return result


def cross_body(left, right, bank):
    env, reset, model = ([r[key] if key != 'reset' else r['native']['reset'] for r in (left, right)]
                         for key in ('environment', 'reset', 'model'))
    same_env = ('source_pin', 'manifest_sha256', 'suite', 'task_id', 'seed', 'init_state_id',
                'bddl_sha256', 'released_row_fingerprint', 'scene_fingerprint',
                'perturbation_plan', 'perturbation_plan_fingerprint')
    same_reset = ('scene_fingerprint', 'prompt', 'noise_seed', 'noise_sha256', 'horizon', 'action_schema',
                  'renderer_vendor', 'renderer', 'perturbation_plan')
    for fields, records in ((same_env, env), (same_reset, reset)):
        for name in fields:
            check(records[0][name] == records[1][name], ('cross-arm', name))
    check(env[0]['released_row_fingerprint'] == bank['row31_fingerprint'] and
          env[0]['bddl_sha256'] == bank['bddl_sha256'] and reset[0]['horizon'] == bank['native_horizon'],
          'release-bank receipt')
    different = sorted(k for k in model[0] if model[0][k] != model[1][k])
    check(model[0].keys() == model[1].keys() and set(different) ==
          {'arm', 'model_class', 'joint_names', 'joint_links', 'xml_sha256', 'arm_meshes'}, 'compiled model mismatch')
    check(reset[0]['physics'] != reset[1]['physics'], 'same full body state on different arms')
    return dict(scene=env[0]['scene_fingerprint'], released_row=env[0]['released_row_fingerprint'],
                noise=reset[0]['noise_sha256'], noise_seed=reset[0]['noise_seed'], model_fields_different=different,
                post_settle_hash_equal=env[0]['post_settle_scene_fingerprint']==env[1]['post_settle_scene_fingerprint'],
                bodies={a: dict(model_class=m['model_class'], xml=m['xml_sha256'], physics=r['physics'],
                                model_nq=m['model_nq'], model_nv=m['model_nv'], settled=e['post_settle_scene_fingerprint'],
                                pose_errors_to_settled_donor=e['post_settle_scene_errors'])
                        for a,m,r,e in zip(ARMS, model, reset, env)})


def audit(root, paired):
    check(blas_info()['kernel'].lower() == 'sandybridge', 'use the exact paired kernel')
    totals, cases, cross, jobs = Counter(), [], [], {}
    groups = {a: [] for a in ARMS}
    law = source_law()
    banks = read(Path(pinned.__file__).parent/'evidence/release-banks.json')['tasks']
    for a in ARMS:
        folder = root/f'gpu01-{a}'
        check({a: paired[a]} == read(folder/'result.json'), (a, 'worker receipt'))
        origin, probe = read(folder/'origin.json'), read(folder/'gpu.json')
        rc, start, end = map(int,(folder/'exit.txt').read_text().split())
        check(origin['pin'] == PIN and origin['arm'] == a and origin['fixture'] is False and
              probe['hostname'] == origin['host'] and probe['status'] == 'passed' and rc == 0,
              'shard identity or completion')
        jobs[a] = dict(host=origin['host'], device=origin['device'], start_epoch=start, end_epoch=end,
                       seconds=end-start, gpu_uuid=probe['devices'][0]['uuid'])
        check(len(list((folder/'episodes').glob('*.json'))) == 4*len(TASKS), 'episode file count')
    for i, (suite, task) in enumerate(TASKS):
        opening = []
        bank, = [b for b in banks if (b['suite'], b['task_id']) == (suite, task)]
        for a in ARMS:
            episodes, stats = {}, {}
            for mode in MODES:
                path = root/f'gpu01-{a}/episodes/{suite}-{task}-{mode}.json'
                raw = path.read_bytes()
                obj = episodes[mode] = json.loads(raw)
                check(obj['mode'] == mode and obj['case']['query'] == paired[a]['tasks'][i]['query'] and
                      obj['origin'] == read(root/f'gpu01-{a}/origin.json'), path)
                values = stats[mode] = episode_audit(obj, law, totals)
                values.update(raw_json_sha256=sha(raw), raw_json_bytes=len(raw))
            pinned.pairing(episodes)
            first = episodes['raw']['chunks'][0]
            for mode in MODES:
                block = episodes[mode]['chunks'][0]
                check(block['policy_input'] == first['policy_input'], 'all four opening inputs')
                exact(block['original'], first['original'], np.float32, 'all four opening proposals')
            outcomes = {m: values['success'] for m, values in stats.items()}
            check(outcomes == paired[a]['tasks'][i]['success'], 'per-task endpoints')
            groups[a].append(outcomes)
            cases.append(dict(arm=a, suite=suite, task=task, modes=stats))
            opening.append(episodes['raw'])
        cross.append(dict(suite=suite, task=task, **cross_body(*opening, bank)))
    for a in ARMS:
        check(endpoints([groups[a]]) == paired[a]['native'], ('independent counts and intervals', a))
    pooled = endpoints(list(groups.values()))
    check(pooled == paired['pooled_equal_arm'], 'independent pooled counts and intervals')
    return dict(pin=PIN, blas=blas_info(), jobs=jobs, totals=dict(totals), cross_body=cross,
                per_case=cases, pooled_equal_arm=pooled, independent_bootstrap_ordered_draws=len(TASKS)**len(TASKS))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--score', type=Path, help='paired result from the unmodified reader; omit for a kernel diagnostic')
    args = parser.parse_args()
    check(os.sched_getaffinity(0) <= {44,45} and os.environ['CUDA_VISIBLE_DEVICES'] == '' and
          os.environ['JAX_PLATFORMS'] == 'cpu', 'development CPU only')
    result = audit(args.root, read(args.score)) if args.score else kernel_comparison(args.root)
    print(json.dumps(result, separators=(',', ':'), allow_nan=False))
