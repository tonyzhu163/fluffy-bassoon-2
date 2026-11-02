"""Four independent allocated task workers; finite Stage 0, no PPO updates."""
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time

import imageio.v2 as imageio
import numpy as np
import torch
from hydra import compose, initialize_config_dir

from fluffy_bassoon.baige_gpu import verify_selected_cuda
from fluffy_bassoon.meta_rl.residual_ppo import ResidualActorCritic, ResidualPPOConfig
from fluffy_bassoon.meta_rl.rlinf_flora_ppo import sha, write_new
from fluffy_bassoon.meta_rl.rlinf_native_ppo import ActionJournal, snapshot_rng
from fluffy_bassoon.meta_rl.robocasa_pairing import canonical_case, pairing_native_factory, public_snapshot
from fluffy_bassoon.meta_rl.robocasa_residual import FrozenRoboCasaMotor, RoboCasaResidualEnvironment, collect_pending_lanes
from fluffy_bassoon.meta_rl.warmstart import atomic_save
from fluffy_bassoon.rl.action_residual import ResidualBound

ROOT = Path('/mowenjie/fluffy-bassoon-2')
CODE = Path(os.environ['TASK_META'])
RUN = ROOT / 'outputs/robocasa-interface-screen-20261102-r2'
SOURCE = ROOT / 'outputs/robocasa-interface-screen-20261102-r1'
RANK = int(os.environ['RANK'])
TASKS = ('OpenDrawer', 'PickPlaceDrawerToCounter', 'CoffeeSetupMug', 'TurnOffStove')
TASK = TASKS[RANK]
OUT = RUN / TASK
START = int(os.environ['TASK_STARTED'])
DEADLINE = START + 9000  # reserve closure within the160-minute scheduler job
spec = importlib.util.spec_from_file_location('screen_interfaces', Path(__file__).with_name('INTERFACES.py'))
interfaces = importlib.util.module_from_spec(spec); spec.loader.exec_module(interfaces)
carry_spec = importlib.util.spec_from_file_location('screen_carry', Path(__file__).with_name('CARRY.py'))
carry = importlib.util.module_from_spec(carry_spec); carry_spec.loader.exec_module(carry)


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def case(index):
    # Fresh prospective TRAIN cases; retain native target layout distribution.
    offsets = dict(OpenDrawer=0, PickPlaceDrawerToCounter=20000, CoffeeSetupMug=40000, TurnOffStove=30000)
    return dict(bank='stage0_train', task=TASK, split='target', native_seed=211100 + offsets[TASK] + index,
                reset_ordinal=0, slot=index, counted=True, layout_source='native_target_constructor_seed_first_reset')


def plan(cases, draws):
    seeds, codes, action_noise = [], {}, []
    for value, draw in zip(cases, draws, strict=True):
        sequence = np.random.default_rng(900000000 + value['native_seed'] * 100 + draw + 1)
        for seed in sequence.integers(0, 2**63 - 100000000, 12).tolist():
            seeds.append(seed)
            rng = np.random.default_rng(seed + 17)
            codes[seed] = torch.from_numpy(rng.normal(0, .5, 32).astype(np.float32))
            action_noise.append(rng.normal(0, 1, 6))
    return seeds, codes, np.asarray(action_noise, dtype=np.float32)


class PairedEnvironment(RoboCasaResidualEnvironment):
    def reset(self):
        if time.time() > DEADLINE:
            raise TimeoutError('Finite screen boundary; preserve all partial episodes')
        observation = super().reset()
        physical = self.native.env.get_env_attr('_native_pairing')[0]
        snapshot = public_snapshot(observation, self.case, physical)
        path = self.output / (self.key + '-initial.pt')
        atomic_save(path, snapshot)
        marker, report = canonical_case(RUN / 'canonical' / TASK / str(self.case['native_seed']), snapshot,
            save=atomic_save, load=lambda p: torch.load(p, map_location='cpu', weights_only=False),
            source=dict(source_commit=os.environ['TASK_CODE_PIN'], episode=str(path)))
        write_new(self.output / (self.key + '-pairing.json'), dict(canonical=str(marker), report=report,
            snapshot=dict(path=str(path), sha256=sha(path))))
        return observation


def run_bank(name, cases, draws, arm='native', fraction=0., direct=None, critic=None, deterministic_direct=False):
    destination = OUT / name
    destination.mkdir(parents=True, exist_ok=False)
    carried, original_indices = [], list(range(len(cases)))
    if name == 'native8':
        carried = CARRY_TASK['native_rows']
        original_indices = CARRY_TASK['native_missing_indices']
        cases, draws = [cases[i] for i in original_indices], [draws[i] for i in original_indices]
    seeds, codes, noise = plan(cases, draws)
    adapter = interfaces.InterfaceMotor(motor, arm=arm, controls=codes, fraction=fraction, direct=direct, critic=critic)
    stochastic = arm == 'action' or arm == 'direct' and not deterministic_direct
    bound = ResidualBound.from_std(stats['actions']['std'], indices=range(6), fraction=fraction if arm == 'action' else .25, action_dim=12)
    lanes = min(8, len(cases))
    journals = [ActionJournal(str(RUN / 'actions'), rank=RANK, world=4, attempt=name + '-lane' + str(lane),
                              maximum_actions=None) for lane in range(lanes)]
    environments = [PairedEnvironment(pairing_native_factory(cfg.env.eval), cases, horizon=600,
        output=destination / 'videos', monitoring_root=destination / 'monitoring', journal=journals[lane],
        writer_factory=imageio.get_writer, episode_id_prefix='r2-' + name + '-', rank=RANK) for lane in range(lanes)]
    atomic_save(destination / 'PLAN.pt', dict(source_commit=os.environ['TASK_CODE_PIN'], cases=cases, draws=draws,
        original_indices=original_indices, planned_motor_seeds=seeds, actor_standard_normal=noise if stochastic else None,
        motor_controls={key: value.numpy() for key, value in codes.items()}, carried_rows=carried, arm=arm, fraction=fraction))
    completed_slots = set()
    def checkpoint(slot, records, episode):
        assert slot not in completed_slots
        planned = seeds[slot * 12:(slot + 1) * 12]
        atomic_save(destination / 'episodes' / f'slot-{slot:06d}.pt', dict(case=cases[slot], draw=draws[slot],
            original_index=original_indices[slot], source_commit=os.environ['TASK_CODE_PIN'],
            records=records, episode=episode, planned_motor_seeds=planned,
            selector_draws={seed: adapter.selector_draws[seed] for seed in planned if seed in adapter.selector_draws}))
        completed_slots.add(slot)
        save(OUT / 'progress.json', dict(stage=name, new_completed_episodes=len(completed_slots),
            carried_episodes=len(carried), completed_episodes=len(carried) + len(completed_slots),
            bank_total=len(carried) + len(cases), completed_slots=sorted(completed_slots),
            updated_unix=time.time(), every_complete_episode_checkpointed=True))
    tick = time.monotonic()
    try:
        records, episodes = collect_pending_lanes(policy, bound, adapter, environments,
            planned_motor_seeds=seeds, planned_action_noise=noise if stochastic else None,
            deterministic=not stochastic, verify_zero=not stochastic, episode_callback=checkpoint)
        rows = []
        for index, (episode, native_case, draw) in enumerate(zip(episodes, cases, draws, strict=True)):
            truth = json.loads((destination / 'monitoring' / (episode['id'] + '.json')).read_text())
            assert truth['case'] == native_case
            assert truth['native_counts']['controls'] - truth['native_counts']['reset_controls'] == 600
            rows.append(dict(case=native_case, draw=draw, success=truth['simulator_final_success'],
                original_slot=original_indices[index], source_run='robocasa-interface-screen-20261102-r2',
                truth=dict(path=str(destination / 'monitoring' / (episode['id'] + '.json')),
                           sha256=sha(destination / 'monitoring' / (episode['id'] + '.json'))),
                episode=episode))
        atomic_save(destination / 'records.pt', dict(records=records, rows=rows, carried_rows=carried, arm=arm, fraction=fraction,
            planned_motor_seeds=seeds, controls={k: v.numpy() for k, v in codes.items()},
            actor_standard_normal=noise if stochastic else None, selector_choices=adapter.selector_choices,
            selector_draws=adapter.selector_draws,
            inference_draws=adapter.inference_draws))
        combined_rows = sorted(carried + rows, key=lambda row: row['original_slot'])
        result = dict(completed=True, task=TASK, arm=arm, fraction=fraction, rows=combined_rows,
            carried_episodes=len(carried), new_episodes=len(rows), new_policy_controls=len(rows) * 600,
            policy_controls=(len(carried) + len(rows)) * 600,
            inference_draws=adapter.inference_draws + len(carried) * 12, new_inference_draws=adapter.inference_draws,
            seconds=time.monotonic() - tick, zero_residual_verified=sum(r['zero_residual_verified'] for r in records),
            source_commit=os.environ['TASK_CODE_PIN'])
        write_new(destination / 'RESULT.json', result)
        save(OUT / 'progress.json', dict(stage=name, completed_episodes=len(rows), updated_unix=time.time(),
                                        seconds=result['seconds'], successes=sum(r['success'] for r in rows)))
        print(json.dumps(dict(stage=name, episodes=len(rows), successes=sum(r['success'] for r in rows),
                              seconds=result['seconds'])), flush=True)
        return result, records
    finally:
        for env in environments:
            env.close()


def calibrate(anchor):
    # Use only public initial observations. Never read successes here.
    snapshots = sorted((SOURCE / TASK / anchor / 'videos').glob('*-initial.pt'))[:2]
    observations = []
    for path in snapshots:
        snapshot = torch.load(path, map_location='cpu', weights_only=False)
        observations.append(dict(snapshot['images'], states=snapshot['states'], task_descriptions=snapshot['task_descriptions']))
    zero = torch.zeros(1, 32, device='cuda')
    code = torch.tensor(np.random.default_rng(921001).normal(0, .5, (1, 32)), dtype=torch.float32, device='cuda')
    rows = []
    action_std = np.asarray(stats['actions']['std'])[:12]
    parent = json.loads(Path(CARRY_TASK['interface_qualification']['path']).read_text())
    assert sha(CARRY_TASK['interface_qualification']['path']) == CARRY_TASK['interface_qualification']['sha256']
    for entry in parent['rows']:
        arm, scales = entry['arm'], (entry['selected'],)
        rms = {scale: [] for scale in scales}
        for obs in observations:
            base, _, _ = motor.infer_batch(obs, noise_seeds=[921002])
            adapter = interfaces.InterfaceMotor(motor, arm=arm, controls={921002: zero[0].cpu()}, fraction=scales[0])
            replay, _, _ = adapter.infer_batch(obs, noise_seeds=[921002])
            assert np.array_equal(base, replay), arm + ' zero-control native replay'
            for scale in scales:
                adapter.fraction = scale; adapter.controls = {921002: code[0].cpu()}
                altered, _, _ = adapter.infer_batch(obs, noise_seeds=[921002])
                rms[scale].append(float(np.sqrt(np.mean(((altered - base) / (action_std + 1e-6)) ** 2))))
        target = float(np.sqrt(np.mean((.25 * np.tanh(code[0, :6].cpu().numpy())) ** 2) * .5))
        selected = min(scales, key=lambda scale: abs(np.mean(rms[scale]) - target))
        assert np.mean(rms[selected]) > 0 and np.isfinite(list(rms.values())).all()
        rows.append(dict(arm=arm, scales={str(k): v for k, v in rms.items()}, selected=selected,
                         target_normalized_action_rms=target, native_zero_bitwise_equal=True,
                         scale_selection='Frozen original fraction; two saved public observations verify the new GPU; no reselection'))
    write_new(OUT / 'INTERFACE-QUALIFICATION.json', dict(completed=True, rows=rows,
        noise='32 bounded tanh(u) coordinates repeated H50, replacing native independent Gaussian; zero is not a native draw'))
    return {r['arm']: r['selected'] for r in rows}


def fit_direct(anchor_records):
    # Anchor episodes are TRAIN data. The split is by complete episode.
    records = list(anchor_records)
    source = ROOT / 'outputs/robocasa-eight-gpu-speed-20261102-r1/S8-r2'
    sources = []
    for update in range(1, 7):
        path = source / f'pending-{update:03d}.pt'
        payload = torch.load(path, map_location='cpu', weights_only=False)
        assert payload['stage'] == 'pending_labels'
        records.extend(payload['pending_records'])
        sources.append(dict(path=str(path), sha256=sha(path)))
    x = torch.as_tensor(np.stack([r['features'] for r in records]), device='cuda')
    y = torch.as_tensor(np.stack([r['base_chunk'] for r in records]), device='cuda')
    assert x.shape[1] == 2088 and y.shape[1:] == (50, 12)
    y = (y - torch.tensor(stats['actions']['mean'][:12], device='cuda')) / (torch.tensor(stats['actions']['std'][:12], device='cuda') + 1e-6)
    starts = np.array([r['episode_start'] for r in records], dtype=bool)
    groups = np.cumsum(starts) - 1
    validation = torch.tensor(groups % 5 == 0, device='cuda')
    train_indices = (~validation).nonzero().flatten()
    val_indices = validation.nonzero().flatten()
    torch.manual_seed(921010 + RANK)
    head = interfaces.DirectHead().cuda()
    values = head.inputs(x[train_indices]); head.mean.copy_(values.mean(0)); head.std.copy_(values.std(0, unbiased=False).clamp_min(1e-4))
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4)
    sampler = torch.Generator(device='cuda').manual_seed(921020 + RANK)
    best, best_step, best_state, stale = float('inf'), 0, None, 0
    log = []
    for step in range(1, 4001):
        indices = train_indices[torch.randint(len(train_indices), (128,), generator=sampler, device='cuda')]
        loss = (head(x[indices]) - y[indices]).square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 50 == 0:
            with torch.no_grad():
                val = float((head(x[val_indices]) - y[val_indices]).square().mean())
                target_mask = validation[:len(anchor_records)]
                target_val = float((head(x[:len(anchor_records)][target_mask]) - y[:len(anchor_records)][target_mask]).square().mean())
            log.append(dict(step=step, validation_mse=val, task_validation_mse=target_val))
            if val < best - 1e-5:
                best, best_step, stale = val, step, 0
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            else:
                stale += 1
            atomic_save(OUT / 'direct-latest.pt', dict(model=head.state_dict(), optimizer=optimizer.state_dict(),
                sampler_rng=sampler.get_state(), global_rng=snapshot_rng(), step=step, best_step=best_step,
                best_state=best_state, contract=dict(sources=sources, base_action_features_removed=True,
                    split='Every fifth complete episode validation; no simulator truth used in fitting')))
            if stale >= 12 or time.time() > DEADLINE - 900:
                break
    head.load_state_dict(best_state); head.eval().requires_grad_(False)
    with torch.no_grad():
        target_mask = validation[:len(anchor_records)]
        target_val = float((head(x[:len(anchor_records)][target_mask]) - y[:len(anchor_records)][target_mask]).square().mean())
    atomic_save(OUT / 'direct-selected.pt', dict(model=best_state, best_step=best_step, validation_mse=best))
    result = dict(completed=True, updates=step, selected_update=best_step, validation_mse=best,
                  task_validation_mse=target_val, preliminary_mse_gate=bool(best <= .25 and target_val <= .25),
                  source_refs=sources, anchor_records=len(anchor_records), log=log,
                  PPO_updates=0, purpose='Supervised motor distillation; competence still requires paired episodes')
    write_new(OUT / 'DIRECT-FIT.json', result)
    return head, result


def main():
    global motor, policy, cfg, stats, CARRY_TASK
    assert platform.system() == 'Linux' and socket.gethostname().startswith('job-')
    assert int(os.environ['WORLD_SIZE']) == 4 and 0 <= RANK < 4 and torch.cuda.device_count() == 1
    assert subprocess.check_output(['git', '-C', str(CODE), 'rev-parse', 'HEAD'], text=True).strip() == os.environ['TASK_CODE_PIN']
    assert not subprocess.check_output(['git', '-C', str(CODE), 'status', '--porcelain'], text=True).strip()
    OUT.mkdir(parents=True, exist_ok=False)
    assert sha(RUN / 'CARRY-AUDIT.json') == os.environ['TASK_CARRY_SHA']
    CARRY_TASK = next(row for row in json.loads((RUN / 'CARRY-AUDIT.json').read_text())['tasks'] if row['task'] == TASK)
    probe = OUT / 'gpu-probe.json'
    subprocess.run([sys.executable, '-B', str(CODE / 'scripts/[train]gpu_probe.py'), '--expected-count', '1',
        '--purpose', 'training', '--baige-container', '--baige-exclusive-training', '--output', str(probe)], check=True)
    device = json.loads(probe.read_text())['devices'][0]
    os.environ.update(CUDA_VISIBLE_DEVICES=device['visibility_token'], LYCO_BAIGE_GPU_MAP=str(probe),
                      LYCO_BAIGE_ISOLATED_UUID=device['uuid'], MUJOCO_EGL_DEVICE_ID=str(device['allocation_ordinal']))
    verify_selected_cuda(torch)
    cpu = Path('/sys/fs/cgroup/cpu.max').read_text().split(); assert int(cpu[0]) / int(cpu[1]) >= 32
    assert int(Path('/sys/fs/cgroup/memory.max').read_text()) >= 128 << 30
    filesystem = os.statvfs(OUT); assert filesystem.f_bavail * filesystem.f_frsize >= 96 << 30
    (OUT / 'write-probe').write_text('project shared-storage write probe\n')
    source = json.loads((CODE / 'outputs/robocasa-joint-six-update-proposal-20261102-r1/BINDING-TEMPLATE.json').read_text())['policy_profile']
    initializer = source['initializer']
    assert sha(Path(initializer['model_directory']) / 'model.safetensors') == initializer['model_sha256']
    assert sha(initializer['norm_path']) == initializer['normalization_sha256']
    assert subprocess.check_output(['git', '-C', os.environ['RLINF_PATH'], 'rev-parse', 'HEAD'], text=True).strip() == source['rlinf_execution_commit']
    manifest = dict(task=TASK, rank=RANK, hostname=socket.gethostname(), started_unix=START,
        source_commit=os.environ['TASK_CODE_PIN'], device=device, initializer=initializer,
        CPU=32, RAM_GiB=128, SHM_GiB=16, lanes=8, quota='Shared2TiB volume; no separate quota interface identified',
        free_bytes=filesystem.f_bavail * filesystem.f_frsize, deadline=DEADLINE,
        aggregate_authorized_GPU_seconds=43200, maximum_fresh_GPU_seconds=38400,
        prior_conservative_GPU_seconds=2924, carry_audit_sha256=os.environ['TASK_CARRY_SHA'], PPO_updates=0)
    write_new(OUT / 'manifest.json', manifest)
    while not all((RUN / task / 'manifest.json').exists() for task in TASKS):
        if time.time() > START + 600:
            raise TimeoutError('Actual four-task GPU admission incomplete')
        time.sleep(2)
    manifests = [json.loads((RUN / task / 'manifest.json').read_text()) for task in TASKS]
    assert len({value['device']['uuid'] for value in manifests}) == 4
    assert all(value['source_commit'] == os.environ['TASK_CODE_PIN'] for value in manifests)
    with initialize_config_dir(config_dir=str(CODE / 'configs/native_ppo'), version_base=None):
        cfg = compose(config_name='robocasa365_pi05_eval')
    cfg.env.eval.episode_horizon_source = 'max_episode_steps'
    cfg.rollout.model.model_path = initializer['model_directory']
    cfg.rollout.model.openpi_data.norm_stats_path = initializer['norm_path']
    from rlinf.models.embodiment.openpi import get_model
    model = get_model(cfg.rollout.model, torch_dtype=torch.bfloat16).to('cuda').eval().requires_grad_(False)
    stats = json.loads(Path(initializer['norm_path']).read_text()); stats = stats.get('norm_stats', stats)
    motor = FrozenRoboCasaMotor(model, stats)
    policy = ResidualActorCritic(ResidualPPOConfig(feature_dim=2088, residual_dim=6)).cuda().eval().requires_grad_(False)
    parent_bank, anchor_records, anchor_refs = carry.load_anchors(TASK)
    assert anchor_refs == CARRY_TASK['anchor_record_refs']
    failures, all_anchors = parent_bank['selected'], parent_bank['all_anchors']
    assert failures == CARRY_TASK['selected'] and len(failures) == 20
    for value in failures:
        old_directory = SOURCE / 'canonical' / TASK / str(value['native_seed'])
        marker = json.loads((old_directory / 'complete.json').read_text())
        original = old_directory / 'initial.pt'; assert sha(original) == marker['snapshot_sha256']
        directory = RUN / 'canonical' / TASK / str(value['native_seed']); directory.mkdir(parents=True, exist_ok=False)
        atomic_save(directory / 'initial.pt', torch.load(original, map_location='cpu', weights_only=False))
        write_new(directory / 'complete.json', dict(case=value,
            source=dict(parent_canonical=dict(path=str(original), sha256=sha(original)), source_commit=os.environ['TASK_CODE_PIN']),
            snapshot_sha256=sha(directory / 'initial.pt')))
    write_new(OUT / 'FAILURE-BANK.json', dict(selected=failures, all_anchors=all_anchors,
        selection='Original selected20 and all original anchors reused by reference; no new failure selection',
        source_bank=CARRY_TASK['bank'], anchor_record_refs=anchor_refs))
    scales = calibrate('anchor-000')
    bank = [value for value in failures for _ in range(8)]
    draws = list(range(8)) * len(failures)
    summaries = {}
    supplementary_controls = 0
    for name, arm, fraction in (('native8', 'native', 0.), ('action025', 'action', .25),
            ('action050', 'action', .5), ('noise32', 'noise', 0.),
            ('conditioning', 'conditioning', scales['conditioning']), ('velocity', 'velocity', scales['velocity'])):
        summaries[name], _ = run_bank(name, bank, draws, arm, fraction)
    direct, fit = fit_direct(anchor_records)
    coverage = dict(direct_fit=fit)
    if fit['preliminary_mse_gate']:
        qualified_cases = [case(i) for i in range(200, 208)]
        original, _ = run_bank('direct-qualification-native', qualified_cases, [0] * 8)
        # Deterministic direct head: use arm directly but zero residual actor.
        candidate, _ = run_bank('direct-qualification-head', qualified_cases, [0] * 8, 'direct', .25,
                                direct=direct, deterministic_direct=True)
        a, b = sum(r['success'] for r in original['rows']), sum(r['success'] for r in candidate['rows'])
        supplementary_controls += original['policy_controls'] + candidate['policy_controls']
        admitted = b >= max(1, a - 1)
        coverage['direct_competence'] = dict(native_successes=a, direct_successes=b, cases=8, admitted=admitted,
            gate='Preliminary eight-case TRAIN competence check; no broad equivalence claim')
        if admitted:
            summaries['direct'], _ = run_bank('direct', bank, draws, 'direct', .25, direct=direct)
    teacher = ROOT / 'outputs/robocasa-eight-gpu-speed-20261102-r1/S8-r2/final.pt'
    assert sha(teacher) == '84b4a25f1f0e55a22f4fdf265766fa5f6295504db2cf9a0a3ba8a0a28cc44f4d'
    saved = torch.load(teacher, map_location='cpu', weights_only=False)
    critic = ResidualActorCritic(ResidualPPOConfig(**saved['contract']['ppo'])).cuda().eval().requires_grad_(False)
    critic.load_state_dict(saved['policy'], strict=True)
    summaries['select8'], _ = run_bank('select8', failures, [0] * len(failures), 'select', 0., critic=critic)
    coverage['selector'] = dict(type='Frozen simulator-source PPO value heuristic on candidate-specific public features',
        source=dict(path=str(teacher), sha256=sha(teacher)), candidates_per_chunk=8, trials_per_case=1,
        caveat='Value heuristic is not a qualified Q-function or a learned history selector; no simulator oracle at deployment')
    compact = {}
    for name, summary in summaries.items():
        per_case = {}
        for row in summary['rows']:
            per_case.setdefault(str(row['case']['native_seed']), []).append(bool(row['success']))
        compact[name] = dict(episodes=len(summary['rows']), successes=sum(r['success'] for r in summary['rows']),
            recovered_cases=sum(any(values) for values in per_case.values()), cases=len(per_case),
            per_case=per_case, inference_draws=summary['inference_draws'], seconds=summary['seconds'])
    write_new(OUT / 'RESULT.json', dict(completed=True, task=TASK, source_commit=os.environ['TASK_CODE_PIN'],
        headroom_only=True, PPO_updates=0, panels=compact, qualification=coverage,
        incomplete_families=[] if 'direct' in summaries else ['direct: warm-start/competence gate not passed'],
        physical_policy_controls=sum(s['new_policy_controls'] for s in summaries.values()) + supplementary_controls,
        reused_original_policy_controls=600 * (len(all_anchors) + len(CARRY_TASK['native_rows'])),
        carried_native_rows_missing_chunk_records=len(CARRY_TASK['native_rows'])))


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    try:
        main()
    finally:
        # Master remains alive until every independent task node records exit.
        # Each shell publishes its terminal clock; the shell handles the barrier.
        pass
