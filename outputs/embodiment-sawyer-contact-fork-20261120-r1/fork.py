"""Run three paired native Sawyer continuations in the qualified Linux CPU runtime."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'embodiment-sawyer-source-contact-20261120-r1'
sys.path.insert(0, str(SOURCE))
import replay as original

HORIZON = 300
PRIOR_REPORT = '8c04846ccd50d7656b437091b59a0c8049130cbd6094fa2ec5792dcad76bdf28'
BRANCHES = ('raw', 'contact_expert', 'reset_expert')


def same_bytes(label, actual, expected):
    if (actual.dtype != expected.dtype or actual.shape != expected.shape
            or actual.tobytes() != expected.tobytes()):
        raise original.Rejected(f'Byte mismatch in {label}')


def compare_fields(label, actual, expected):
    original.require(set(actual.keys()) == set(expected), f'{label} field names differ')
    for name in expected:
        same_bytes(f'{label}/{name}', actual[name], expected[name])


def fingerprint(fields):
    return hashlib.sha256(json.dumps({k: original.state_sha(v) for k, v in fields.items()},
                                     sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def snapshot(arm, obs):
    """Read native integration, controller memory and clocks without forwarding the simulator."""
    import mujoco
    import numpy as np

    wrapper = arm._env
    scene, sim = wrapper.env, wrapper.sim
    robot, = wrapper.robots
    model, data = sim.model._model, sim.data._data
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, spec), dtype=np.float64)
    mujoco.mj_getState(model, data, state, spec)
    fields = {'integration': state}

    def add(key, value):
        if value is None:
            fields[key + '.none'] = np.array(True)
        elif isinstance(value, dict):
            for part, item in value.items():
                add(f'{key}.{part}', item)
        elif isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                add(f'{key}.{i}', item)
        else:
            array = np.asarray(value)
            original.require(not array.dtype.hasobject, f'Unsupported snapshot field {key}')
            fields[key] = array.copy()

    controller = robot.controller
    for key, value in vars(controller).items():
        # The simulator is captured by mj_getState; interpolators, if configured,
        # have their own goal, interpolation progress and previous reference.
        if key == 'sim':
            continue
        if key.startswith('interpolator') and value is not None:
            add('osc.' + key, vars(value))
        else:
            add('osc.' + key, value)
    add('gripper.current_action', robot.gripper.current_action)
    add('public', obs)
    add('arm.previous_public', arm._previous_public)
    add('arm.tool', arm._tool)
    add('arm.step', arm._step)
    add('arm.done', arm._done)
    add('mj.time', data.time)
    for name in ('qpos', 'qvel', 'ctrl', 'qacc_warmstart'):
        add('mj.' + name, getattr(data, name))
    add('scene.timestep', scene.timestep)
    add('scene.cur_time', scene.cur_time)
    add('scene.done', scene.done)
    add('scene.perturbation_step', scene._perturbation_step)
    return fields


def check_original_step(before, packet, tape, plans, index):
    """The tape validator has already authenticated and reconstructed each issued chunk."""
    same_bytes(f'archived issued action at {index + 1}', packet['issued_action'], tape['actions'][index])
    same_bytes(f'archived sensor motion at {index + 1}', packet['sensor_motion'], tape['motion'][index])
    done = packet['terminated'] or packet['truncated']
    original.require((packet['worker_step_index'], packet['reward'], done, packet['success']) ==
                     (tape['native_steps'][index], tape['rewards'][index], tape['done'][index],
                      tape['success'][index]), f'Archived transition differs at control {index + 1}')
    if index % 5 == 0:
        original.require(original.state_sha(before['state']) == plans[index // 5]['state'],
                         f'Archived replan state differs before control {index + 1}')


def teacher_dependencies(checkout):
    r2 = Path(original.R2)
    r1 = r2.parent / 'libero-paligemma-crossarm-20261120-r1'
    files = [r2 / name for name in ('recover_teacher.py', 'recover.py', 'paddle.py', 'expert_sweep.py')]
    files.extend(r1 / name for name in ('demos.py', 'oracle_fallback.py'))
    hashes = {}
    for rel in files:
        sha = hashlib.sha256(original.git(checkout, 'show', f'{original.PIN}:{rel}')).hexdigest()
        hashes[str(rel)] = original.verified(checkout / rel, sha)
    return hashes


def run_branch(args, layout, state, tape, archived, plans, prior, branch, reference):
    import numpy as np
    from generated_layouts import CASE  # Also loads the modules inspected by the qualified runtime audit.
    from expert import session
    from geometry import Geometry
    from large_panel_sim import physical_reset

    first = prior['geometry']['summary']['hand']['first_touch']
    boundary = first['control']
    start = 0 if branch == 'reset_expert' else boundary
    path = {'actions': [], 'action_source': [], 'motion': [], 'rewards': [], 'terminated': [],
            'truncated': [], 'success': [], 'native_steps': [], 'public_state': []}
    snapshots = {}
    receipt = {'first_teacher_control': None if branch == 'raw' else start + 1,
               'suffix_after_controls': start,
               'total_budget': HORIZON, 'suffix_budget': HORIZON - start}
    with session(original.BODY, False) as env:
        if branch == 'raw':
            runtime = original.runtime_identity(args, env, layout)
            original.require(runtime == prior['runtime'], 'Native runtime differs from qualified replay')
            receipt['runtime'] = runtime
        obs, reset, _ = physical_reset(env, original.BODY, False, state, layout)
        original.require(reset['environment_reproducibility']['perturbation_plan']['case_id'] == CASE == 'BASE',
                         'Unexpected native perturbation')
        opening = {}
        original.check_opening(obs, reset, archived, plans[0], opening)
        opening = opening['opening']
        original.require(opening['exact_physics'] and opening['exact_public_state'] and opening['exact_poses']
                         and opening['live_admission_sha256'] == prior['physics_validation']['opening']['live_admission_sha256'],
                         'Opening differs from the original Sawyer start')
        receipt['opening_admission_sha256'] = opening['live_admission_sha256']
        snapshots['opening'] = snapshot(env.arm, obs)
        if reference is not None:
            compare_fields('paired opening', snapshots['opening'], reference['opening'])
        path['public_state'].append(obs['state'].copy())
        teacher = None
        with Geometry(env, obs) as geometry:
            for t in range(HORIZON):
                if t == start:
                    if branch != 'reset_expert':
                        original.require(geometry.first.get('hand') == first,
                                         f'Live first hand contact differs at control {t}')
                        snapshots['handoff'] = snapshot(env.arm, obs)
                        if reference is not None:
                            compare_fields('paired handoff', snapshots['handoff'], reference['handoff'])
                    if branch != 'raw':
                        import teacher as teacher_module
                        original.require(Path(teacher_module.__file__).resolve() == HERE / 'teacher.py',
                                         'Unexpected teacher module')
                        teacher = teacher_module.make_teacher(env.arm, obs)
                        receipt['teacher_handoff'] = teacher.handoff
                        for module in ('recover_teacher', 'paddle'):
                            original.require(Path(sys.modules[module].__file__).resolve()
                                             == args.checkout / original.R2 / f'{module}.py',
                                             f'Unexpected active teacher dependency {module}')
                        compare_fields('teacher construction', snapshot(env.arm, obs),
                                       snapshots['opening' if start == 0 else 'handoff'])
                requested = teacher(obs) if teacher is not None else tape['actions'][t]
                packet = env.arm.step(requested)
                issued = packet['issued_action']
                same_bytes('requested and issued float32 control', issued, np.asarray(requested, dtype=np.float32))
                if teacher is None:
                    check_original_step(obs, packet, tape, plans, t)
                obs = env.observation(packet)
                original.require(packet['worker_step_index'] == t + 1 and obs['prompt'] == plans[0]['prompt'] and
                                 np.isfinite(obs['state']).all() and np.isfinite(packet['reward']),
                                 f'Invalid native observation or step index at control {t + 1}')
                geometry.sample(obs)
                for key, value in (('actions', issued), ('motion', packet['sensor_motion']),
                                   ('action_source', int(teacher is not None)), ('public_state', obs['state']),
                                   ('rewards', packet['reward']), ('native_steps', packet['worker_step_index'])):
                    path[key].append(value)
                for key in ('terminated', 'truncated', 'success'):
                    path[key].append(bool(packet[key]))
                if packet['terminated'] or packet['truncated']:
                    break
            captured = geometry.result()
    for key in path:
        path[key] = np.asarray(path[key])
    path.update(captured['trace'])
    n = len(path['rewards'])
    done = path['terminated'] | path['truncated']
    original.require(done[-1] and not done[:-1].any() and (n == HORIZON or path['success'][-1]) and
                     np.array_equal(path['rewards'], path['success'].astype(path['rewards'].dtype)) and
                     not path['success'][:-1].any(), 'Incomplete branch or inconsistent native reward')
    original.require(n > start, 'No native control executed at or after decision point')
    distance = path['plate_goal_xy_m']
    receipt.update(steps=n, suffix_steps=n-start, native_return=float(path['rewards'].sum()),
                   terminal_reward=float(path['rewards'][-1]), success=bool(path['success'][-1]),
                   terminated=bool(path['terminated'][-1]), truncated=bool(path['truncated'][-1]),
                   expert_actions=int(path['action_source'].sum()),
                   original_actions=int((path['action_source'] == 0).sum()),
                   start_distance_xy_m=float(distance[start]), final_distance_xy_m=float(distance[-1]),
                   snapshots_sha256={k: fingerprint(v) for k, v in snapshots.items()})
    return receipt, path, snapshots


# Boundary arrays contain the reset frame plus one frame after each executed control.
BOUNDARY_FIELDS = {'public_state', 'time_s', 'eef_xyz_m', 'plate_xyz_m', 'goal_xyz_m', 'plate_goal_xy_m'}


def prefix(path, boundary):
    selected = {}
    for key, value in path.items():
        if key == 'plate_pairs':
            selected[key] = value[value[:, 0] <= boundary]
        else:
            selected[key] = value[:boundary + int(key in BOUNDARY_FIELDS)]
    return selected


def episode(args, contract, layout, receipt, prior, row):
    import numpy as np

    seed = layout['layout_seed']
    row['stage'] = 'archived inputs'
    validation = row['archive_validation'] = {}
    state, tape, archived, plans = original.load_episode(args, contract, layout, receipt, validation)
    prior_validation = prior['physics_validation']
    original.require(all(prior_validation[k] == v for k, v in validation.items()) and
                     all(prior['archived'][k] == receipt[k] for k in prior['archived']) and
                     prior['layout_id'] == layout['layout_id'] and prior['status'] == 'validated' and
                     prior_validation['steps'] == receipt['steps'] == prior_validation['exact_motion_steps'] and
                     prior_validation['replans_checked'] == prior_validation['replan_state_hash_matches'] and
                     not prior_validation['violations'], 'Prior validation differs from original tape')
    first = prior['geometry']['summary']['hand']['first_touch']
    boundary = first['control']
    original.require(0 < boundary < receipt['steps'] <= HORIZON, 'No actionable first-contact boundary')
    row.update(archived=prior['archived'], first_hand_touch=first, handoff_completed_controls=boundary,
               remaining_controls=HORIZON-boundary, original_tape=receipt['tape'], branches={})
    prior_path = args.prior / 'validated' / f'{seed}.npz'
    row['prior_trace_sha256'] = original.verified(prior_path, prior['trace_sha256'])
    captures = {}
    raw_prefix = raw_snapshots = None
    for branch in BRANCHES:
        row['stage'] = branch
        result, trajectory, snapshots = run_branch(args, layout, state, tape, archived, plans, prior, branch,
                                                   raw_snapshots)
        row['branches'][branch] = result
        if branch == 'raw':
            raw_snapshots, raw_prefix = snapshots, prefix(trajectory, boundary)
            with np.load(prior_path, allow_pickle=False) as prior_trace:
                for key in prior_trace.files:
                    same_bytes(f'validating full prior trace/{key}', trajectory[key], prior_trace[key])
            original.require(all(result[k] == prior_validation[k] for k in
                                 ('steps', 'native_return', 'success', 'terminal_reward', 'terminated', 'truncated')),
                             'Raw terminal receipt differs from original validated replay')
            result['full_prior_trace_byte_match'] = True
            result['archived_motion_controls_verified'] = result['steps']
            result['archived_replan_states_verified'] = len(plans)
            row['prefix_sha256'] = fingerprint(raw_prefix)
        elif branch == 'contact_expert':
            paired_prefix = prefix(trajectory, boundary)
            compare_fields('full original-action prefix', paired_prefix, raw_prefix)
            result['prefix_sha256'] = fingerprint(paired_prefix)
            result['handoff_byte_match'] = True
        captures.update({f'{branch}__{k}': v for k, v in trajectory.items()})
        for point, fields in snapshots.items():
            captures.update({f'{branch}__{point}__{k}': v for k, v in fields.items()})
    row['stage'] = 'capture'
    target = args.out / 'captures' / f'{seed}.npz'
    tmp = target.with_suffix('.partial')
    with tmp.open('xb') as stream:
        np.savez_compressed(stream, **captures)
    if seed == original.SEEDS.start:
        with np.load(tmp, allow_pickle=False) as saved:
            compare_fields('admission capture readback', saved, captures)
    tmp.replace(target)
    row.update(capture=str(target.relative_to(args.out)), capture_sha256=original.sha(target),
               stage='complete', status='validated')


def case_record(args, layout):
    return args.out / 'cases' / f"{layout['layout_seed']}.json"


def run_case(args, contract, layout, receipt, prior):
    row = dict(layout_seed=layout['layout_seed'], layout_id=layout['layout_id'], arm=original.BODY)
    try:
        episode(args, contract, layout, receipt, prior, row)
    except Exception as exc:
        row.update(original.failure(exc))
    original.write(case_record(args, layout), row)


def dispatch(args, contract, layout, receipt, prior):
    # Fresh interpreter per layout; no other episode shares its native module state.
    path = case_record(args, layout)
    try:
        worker = multiprocessing.get_context('spawn').Process(target=run_case,
                        args=(args, contract, layout, receipt, prior))
        worker.start()
        worker.join()
        if worker.exitcode or not path.is_file():
            raise RuntimeError(f'Native worker exit {worker.exitcode}, case present {path.is_file()}')
        return original.read(path)
    except Exception as exc:
        row = dict(layout_seed=layout['layout_seed'], layout_id=layout['layout_id'], stage='worker',
                   **original.failure(exc))
        original.write(path, row)
        return row


def summarize(rows):
    result = {k: sum(r['status'] == k for r in rows) for k in ('validated', 'rejected', 'exception', 'pending')}
    valid = [row['branches'] for row in rows if row['status'] == 'validated']
    result['native_successes'] = {b: sum(r[b]['success'] for r in valid) for b in BRANCHES}
    result['rescues'] = sum(not r['raw']['success'] and r['contact_expert']['success'] for r in valid)
    result['losses'] = sum(r['raw']['success'] and not r['contact_expert']['success'] for r in valid)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', required=True, type=Path, help='original fc61bf2a evaluator checkout')
    parser.add_argument('--panel', required=True, type=Path, help='original large-panel-v3 root')
    parser.add_argument('--prior', required=True, type=Path, help='validated native01/replay directory')
    parser.add_argument('--out', required=True, type=Path, help='fresh output directory')
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=1, metavar='1..8')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('use the qualified Linux CPU simulator interpreter')
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'LP_NUM_THREADS'):
        os.environ[key] = '1'
    args.checkout, args.panel, args.prior, args.out = (p.resolve() for p in
                                                      (args.checkout, args.panel, args.prior, args.out))
    if any(args.out.is_relative_to(p) for p in (args.checkout, args.panel, args.prior)):
        parser.error('--out must be outside all source directories')
    args.out.mkdir(parents=True, exist_ok=False)
    for subdir in ('captures', 'cases'):
        (args.out / subdir).mkdir()
    rows = {s: dict(layout_seed=s, layout_id=s + 1000000, status='pending') for s in original.SEEDS}
    source = {}
    report = dict(status='running', source=source, arm=original.BODY, horizon=HORIZON,
                  action_source={'0': 'original Huber stride-5 issued tape', '1': 'fixed teacher.make_teacher'},
                  capture_schema='branch__field or branch__opening/handoff__snapshot_field')

    def publish():
        report['cases'] = list(rows.values())
        report['counts'] = summarize(report['cases'])
        original.write(args.out / 'cases.json', report)

    def record(row):
        rows[row['layout_seed']] = row
        publish()
        print(row['layout_seed'], row['status'], row.get('reason', ''), flush=True)

    publish()
    try:
        source['prior_report_sha256'] = original.verified(args.prior / 'report.json', PRIOR_REPORT)
        baseline = original.read(args.prior / 'report.json')
        original.require(baseline['counts'] == dict(validated=50, rejected=0, exception=0, pending=0) and
                         [r['layout_seed'] for r in baseline['episodes']] == list(original.SEEDS),
                         'Incomplete or different prior cohort')
        contract, layouts, receipts = original.prepare(args, source)
        original.require(all(source.get(k) == v for k, v in baseline['source'].items()
                             if k not in ('checkout', 'release', 'panel_root')), 'Prior replay provenance differs')
        source.update(prior_root=str(args.prior), fork_sha256=original.sha(Path(__file__)),
                      teacher_sha256=original.sha(HERE / 'teacher.py'),
                      teacher_dependencies=teacher_dependencies(args.checkout),
                      fork_revision=original.git(HERE, 'rev-parse', 'HEAD').decode().strip())
        first_layout = layouts[0]
        first_seed = first_layout['layout_seed']
        first = dispatch(args, contract, first_layout, receipts[first_seed], baseline['episodes'][0])
        report['native_admission'] = dict(layout_seed=first_seed, status=first['status'])
        record(first)
        original.require(first['status'] == 'validated', 'First-layout native validation failed',
                         first.get('reason'))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = [pool.submit(dispatch, args, contract, layout, receipts[layout['layout_seed']], prior)
                    for layout, prior in zip(layouts[1:], baseline['episodes'][1:], strict=True)]
            for job in as_completed(jobs):
                record(job.result())
        counts = report['counts']
        original.require(counts['validated'] == 50 and counts['native_successes']['raw'] == 20,
                         'The complete 50 cases and 20 original successes are required')
    except Exception as exc:
        report.update(status='failed', failure=original.failure(exc))
        publish()
        print(report['failure']['reason'], file=sys.stderr, flush=True)
        return 1
    report['status'] = 'complete'
    publish()
    print(json.dumps(report['counts'], sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
