"""Selected non-ceiling target DSRL; qualified task57 loop adapted only for task/bank/controls.

PixelSAC math, episode collection, full-slot recovery and physical journals are
reused unchanged. This module is imported only after native selection freezes.
"""
import json
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace
import numpy as np
from . import vla_target_online_dsrl as onset
from . import vla_target_online_dsrl_task57 as qualified
from . import vla_target_online_dsrl_headroom as phase
from .vla_source40_corpus import OnlineSourceSimulator
from .vla_source40_schedule import content_hash


def control_query(sim, motor, learner, case, kind):
    if kind in ('initial_native', 'final_native'):
        return qualified.native_query(sim, motor, learner, case)
    if kind == 'learned':
        return onset.frozen_query(sim, motor, learner, case, record_features=False)
    if kind == 'initial_sac':
        view = SimpleNamespace(agent=learner.agent, replay=learner.replay, updates=1,
            action=learner.action, frozen_signature=lambda: onset._signature(learner))
        rows, summary = onset.frozen_query(sim, motor, view, case, record_features=False)
        summary.update(target_optimizer_updates=learner.updates, underlying_actor_optimizer_updates=learner.updates,
            query_behavior='initialized_bounded_sac', sampling_view_only=True)
        return rows, summary
    if kind != 'initial_tied':
        raise ValueError('Unknown frozen query control')
    signature, length, updates = onset._signature(learner), len(learner.replay), learner.updates
    old_agent, old_motor = learner.agent._rng, motor.policy._rng
    old_numpy, old_python = np.random.get_state(), random.getstate()
    seed = onset._seed(case, 'query', 0)
    sampler = np.random.default_rng(seed)
    motor.policy._rng = onset._key(seed)
    try:
        def choose(observation):
            noise = sampler.standard_normal(32).astype(np.float32)
            sim.context = dict(observation=observation, executed_noise=noise,
                native_initial_noise=onset.expand_noise(noise, 10), behavior='tied_unbounded_gaussian', updates=updates)
            return noise
        rows, summary = onset.run_episode(sim, motor, choose, initial_state=case['initial_state'],
            scene_seed=case['scene_seed'], chunk=10, bounded=False, record_features=False)
        for row in rows:
            row['target_task_key'] = row.pop('source_task_key')
        if onset._signature(learner) != signature or len(learner.replay) != length or learner.updates != updates:
            raise RuntimeError('Untrained tied control modified learner state')
        summary.update(completed=True, settling_steps=10, episode_random_seed=seed,
            query_behavior='tied_unbounded_gaussian', target_optimizer_updates=updates,
            underlying_actor_optimizer_updates=updates, optimizer_updates_during_query=0,
            replay_rows_unchanged=True, weights_unchanged=True)
        return rows, summary
    finally:
        learner.agent._rng, motor.policy._rng = old_agent, old_motor
        np.random.set_state(old_numpy); random.setstate(old_python)


def run_learning(sim, motor, learner, sampler, *, output, root, contract, resume=False,
                deadline=float('inf'), task_id=None, query_cases=None, stop_updates=phase.MAX_UPDATES,
                panel_interval=phase.PANEL_INTERVAL, max_actions=phase.MAX_ACTIONS, stop_after_updates=None,
                storage_limit_bytes=phase.MAX_BYTES, storage_root=None):
    """New state-population driver; legacy fixed-case run_case remains unchanged.

    Smaller internal stop/panel/query values support CPU fixtures only. Public
    run CLI binds production constants. No checkpoint is taken mid-episode or
    mid-optimizer-burst; interrupted physical work remains charged in journals.
    """
    output, root = Path(output), Path(root)
    if task_id not in phase.CANDIDATES:
        raise ValueError('Learning task must be the frozen selected candidate')
    query_cases = phase.query_bank(task_id) if query_cases is None else query_cases
    storage_root = root if storage_root is None else Path(storage_root)
    output.mkdir(parents=True, exist_ok=True)
    state_path, contract_path = output / 'state.json', output / 'contract.json'
    if (stop_updates < 1 or panel_interval < 1 or not query_cases
            or stop_updates % panel_interval or any(c['task_id'] != task_id or c['initial_state'] not in range(40, 50) for c in query_cases)):
        raise ValueError('Require complete predeclared task57 panel schedule')
    science_resume = resume and state_path.exists()
    if science_resume:
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError('Task57 immutable scientific contract changed')
        progress = json.loads(state_path.read_text())
        model = progress['model']
        checkpoint = output / model['path']
        if onset.sha256(checkpoint) != model['sha256']:
            raise ValueError('Task57 full recovery checksum differs')
        saved = learner.restore(checkpoint, sampler, contract)
        onset._restore_motor(motor, saved['motor_rng'])
        if (saved['training'] != progress['training'] or learner.updates != model['updates']
                or len(learner.replay) != model['replay_rows']
                or onset._signature(learner) != model['signature']):
            raise ValueError('Atomic task57 cursor/model/training recovery differs')
    else:
        if state_path.exists() or (contract_path.exists() and not resume):
            raise ValueError('Fresh task57 actor requires a new immutable output identity')
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError('Task57 setup-only resume changed the scientific contract')
        if resume and any(json.loads(path.read_text()).get('role') in ('training', 'query')
                for path in (output / 'physical-journal').glob('*.json')):
            raise ValueError('Cannot restart scientific task57 data without its full recovery cursor')
        onset.write_json(contract_path, contract)
        progress = dict(format=phase.FORMAT, phase='initial_native', next_threshold=panel_interval,
            active_panel=None, panels=[], training=dict(episodes=0, chunks=0,
                primitive_steps=0, settling_steps=0, receipts=[], state_hashes={}),
            query_state_hashes={}, model=None)
    journal = onset.JournaledSimulator(sim, output, root, case_cap=max_actions, total_cap=max_actions)
    wrapped = OnlineSourceSimulator(journal, learner, sampler, onset.sac_observation,
        warmup=500, batch=256, utd=20)

    def save(*, full=False, stop_reason=None):
        if full:
            previous = progress['model']
            slot = 1 if previous and previous['path'] == 'recovery-0.pkl' else 0
            path = output / f'recovery-{slot}.pkl'
            learner.save(path, sampler, contract, training=progress['training'], motor_rng=onset._save_motor(motor))
            progress['model'] = dict(path=path.name, sha256=onset.sha256(path), updates=learner.updates,
                replay_rows=len(learner.replay), signature=onset._signature(learner))
        # Full replacement precedes the cursor. The other slot stays valid if
        # this process stops between writes; milestone hardlinks stay immutable.
        onset.write_json(state_path, progress)
        storage = qualified.storage_status(storage_root, output)
        value = dict(progress, completed=progress['phase'] == 'complete', data_role=phase.DATA_ROLE,
            target_only=True, source_assembly_admission=False, source_checkpoint_reused=False,
            contract_sha256=content_hash(contract), state_sha256=onset.sha256(state_path),
            target_optimizer_updates=learner.updates, replay_rows=len(learner.replay),
            costs=onset.costs(output), cost_breakdown_scope='learning case only; whole-phase charge includes all screening/confirmation/learning leaves',
            learning_case_action_charge=onset.charged(output),
            cumulative_whole_phase_charge=onset.joint_charge(root),
            total_action_charge=onset.joint_charge(root),
            maximum_total_actions=max_actions, output_bytes=storage['actual_output_bytes'], storage=storage,
            maximum_output_bytes=storage_limit_bytes, stop_reason=stop_reason,
            latest_recovery_sha256=progress['model']['sha256'])
        onset.write_json(output / 'index.json', value)
        return value

    def admitted():
        if time.monotonic() + 90 >= deadline:
            return 'complete_episode_time_reservation'
        if not journal.reserve_episode():
            return 'complete_episode_action_reservation'
        storage = qualified.storage_status(storage_root, output)
        if storage['actual_output_bytes'] + storage['next_episode_reservation_bytes'] > storage_limit_bytes:
            return 'complete_episode_storage_reservation'
        return None

    def start_panel(kind, threshold):
        name = f'{kind}-{threshold:06d}'
        archive_dir = output / 'milestones'
        archive_dir.mkdir(exist_ok=True)
        archive = archive_dir / f'{name}-recovery.pkl'
        if archive.exists():
            # A crash after hardlink creation but before cursor publication is
            # safe to recover only when the already immutable content matches.
            if onset.sha256(archive) != progress['model']['sha256']:
                raise ValueError('Immutable task57 milestone already exists with different content')
        else:
            os.link(output / progress['model']['path'], archive)
        progress['active_panel'] = dict(name=name, kind=kind, threshold_updates=threshold,
            actual_optimizer_updates=learner.updates, policy_signature=onset._signature(learner),
            recovery_path=str(archive.relative_to(output)), recovery_sha256=onset.sha256(archive),
            query_cursor=0, receipts=[])
        save()

    if not science_resume:
        save(full=True)
    while progress['phase'] != 'complete':
        current_phase = progress['phase']
        if current_phase == 'aborted_before_sgd':
            return save(stop_reason='initial_native_off_weak_band_no_task_substitution')
        if current_phase in ('initial_native', 'initial_tied', 'initial_sac', 'learned_panel', 'final_native'):
            if progress['active_panel'] is None:
                threshold = 0 if current_phase.startswith('initial_') else (
                    progress['next_threshold'] if current_phase == 'learned_panel' else stop_updates)
                start_panel('learned' if current_phase == 'learned_panel' else current_phase, threshold)
            panel = progress['active_panel']
            if learner.updates != panel['actual_optimizer_updates'] or onset._signature(learner) != panel['policy_signature']:
                raise RuntimeError('Task57 query panel actor lineage changed')
            while panel['query_cursor'] < len(query_cases):
                reason = admitted()
                if reason:
                    return save(stop_reason=reason)
                case = query_cases[panel['query_cursor']]
                state = case['initial_state']
                episode_id = f'{panel["name"]}-state-{state:02d}-draw-{case["noise_draw"]}'
                journal.role, journal.episode_id = 'query', episode_id
                rows, summary = control_query(journal, motor, learner, case, panel['kind'])
                state_hash = summary.get('initial_state_vector_sha256')
                previous_hash = progress['query_state_hashes'].get(str(state))
                if not state_hash or (previous_hash is not None and state_hash != previous_hash):
                    raise RuntimeError('Task57 matched query physical state changed across panels')
                progress['query_state_hashes'][str(state)] = state_hash
                qualified._annotate(rows, episode_id=episode_id, state=state, updates=learner.updates)
                for row in rows:
                    row.update(data_role=phase.DATA_ROLE, noise_draw=case['noise_draw'],
                        physical_reset_seed=case['scene_seed'], private_noise_seed=summary['episode_random_seed'])
                summary.update(data_role=phase.DATA_ROLE, case=case, panel_name=panel['name'],
                    threshold_updates=panel['threshold_updates'], actual_optimizer_updates=learner.updates,
                    policy_signature=panel['policy_signature'], query_data_added_to_replay=False,
                    underlying_actor_optimizer_updates=learner.updates)
                panel['receipts'].append(onset.save_trace(output, episode_id, rows, summary, contract))
                panel['query_cursor'] += 1
                save()  # Query restores all training RNG; no duplicate replay checkpoint.
                print(json.dumps(dict(event='query', panel=panel['name'], completed_queries=panel['query_cursor'],
                    updates=learner.updates, actions=journal.total_charge)), flush=True)
            progress['panels'].append(panel)
            progress['active_panel'] = None
            if current_phase == 'initial_native':
                count = sum(bool(row['success']) for row in panel['receipts'])
                if len(query_cases) == 20 and not 2 <= count <= 8:
                    progress['phase'] = 'aborted_before_sgd'
                    return save(stop_reason='initial_native_off_weak_band_no_task_substitution')
                progress['phase'] = 'initial_tied'
            elif current_phase == 'initial_tied':
                progress['phase'] = 'initial_sac'
            elif current_phase == 'initial_sac':
                progress['phase'] = 'training'
            elif current_phase == 'final_native':
                progress['phase'] = 'complete'
            else:
                progress['next_threshold'] += panel_interval
                if progress['next_threshold'] <= stop_updates and learner.updates >= progress['next_threshold']:
                    progress['phase'] = 'learned_panel'
                else:
                    progress['phase'] = 'final_native' if learner.updates >= stop_updates else 'training'
            save()
            continue
        if learner.updates >= progress['next_threshold']:
            progress['phase'] = 'learned_panel'
            save()
            continue
        if stop_after_updates is not None and learner.updates >= stop_after_updates:
            return save(stop_reason='requested_invocation_update_pause')
        reason = admitted()
        if reason:
            return save(stop_reason=reason)
        training = progress['training']
        episode = training['episodes']
        state, episode_id = qualified.TRAIN_STATES[episode % len(qualified.TRAIN_STATES)], f'training-{episode:06d}'
        case = phase.target(task_id, state, bank='training')
        journal.role, journal.episode_id = 'training', episode_id
        before, before_updates = onset._signature(learner), learner.updates
        motor.policy._rng = onset._key(onset._seed(case, 'training_motor', episode))

        def choose(observation):
            noise = wrapped.choose(observation)
            journal.context = dict(wrapped.pending, native_initial_noise=onset.expand_noise(noise, 10),
                policy_signature=before, episode_id=episode_id, data_role=phase.DATA_ROLE)
            return noise

        rows, summary = onset.run_episode(wrapped, motor, choose, initial_state=state,
            scene_seed=case['scene_seed'], chunk=10, bounded=False, record_features=False)
        if (onset._signature(learner) != before or learner.updates != before_updates
                or not rows or not rows[-1]['done'] or len(rows) != len(wrapped.provenance)
                or summary['primitive_steps'] > 400):
            raise RuntimeError('Task57 requires complete episodes with fixed behavior weights')
        state_hash = summary.get('initial_state_vector_sha256')
        old_hash = training['state_hashes'].get(str(state))
        if state_hash in set(progress['query_state_hashes'].values()):
            raise RuntimeError('Actual training/query physical initial states overlap before SGD')
        if not state_hash or (old_hash is not None and state_hash != old_hash):
            raise RuntimeError('Task57 declared physical initial-state identity changed')
        metrics = wrapped.finish_episode()
        expected = before_updates + (20 * len(rows) if len(learner.replay) > 500 else 0)
        if learner.updates != expected or any(not np.isfinite(value) for value in metrics.values()):
            raise RuntimeError('Task57 published SAC update count/finite metric invariant failed')
        for row, provenance in zip(rows, wrapped.provenance, strict=True):
            row.update({key.replace('source_', 'target_', 1): value for key, value in provenance.items()})
            row['target_task_key'] = row.pop('source_task_key')
        qualified._annotate(rows, episode_id=episode_id, state=state, updates=before_updates)
        for row in rows:
            row.update(data_role=phase.DATA_ROLE)
        summary.update(data_role=phase.DATA_ROLE, completed=True, training_episode=episode,
            target_optimizer_updates_before=before_updates, target_optimizer_updates=learner.updates,
            replay_rows=len(learner.replay), update_metrics=metrics, actor_fixed_within_episode=True,
            behavior='warmup_gaussian' if before_updates == 0 else 'target_sac', policy_signature=before)
        training['receipts'].append(onset.save_trace(output, episode_id, rows, summary, contract))
        training['episodes'] += 1
        training['chunks'] += len(rows)
        training['primitive_steps'] += int(summary['primitive_steps'])
        training['settling_steps'] += 10
        training['state_hashes'][str(state)] = state_hash
        save(full=True)
        print(json.dumps(dict(event='training', episodes=training['episodes'], state=state,
            chunks=training['chunks'], updates=learner.updates, actions=journal.total_charge,
            metrics=metrics)), flush=True)
    return save()



def endpoint_summary(result):
    if not result['completed']:
        return dict(available=False, reason='Fixed endpoint and final native panel incomplete')
    panels = result['panels']
    learned = [p for p in panels if p['kind'] == 'learned']
    final_native = next(p for p in panels if p['kind'] == 'final_native')
    initialized = next(p for p in panels if p['kind'] == 'initial_sac')
    count = lambda p: sum(bool(row['success']) for row in p['receipts'])
    baseline = count(final_native)
    return dict(available=True, fixed_endpoint_optimizer_updates=learned[-1]['actual_optimizer_updates'],
        episodes_per_panel=len(final_native['receipts']), final_learned_successes=count(learned[-1]),
        final_native_successes=baseline, initialized_sac_successes=count(initialized),
        primary_paired_success_count_gain=count(learned[-1]) - baseline,
        final_vs_initialized_sac_gain=count(learned[-1]) - count(initialized),
        last_three_learned_successes=[count(p) for p in learned[-3:]],
        predeclared_six_of_twenty_gain_and_last_three_persistence=(len(learned) >= 3 and
            all(count(p) - baseline >= 6 for p in learned[-3:])),
        interpretation='One development task/seed; fixed500k endpoint primary, all milestones reported; native gain without initialized-SAC gain does not establish learning benefit')


def run_selected(motor, server_factory, args, contract, deadline):
    root = Path(args.output)
    path = root / 'frozen-selection.json'
    if not args.selection_sha256 or onset.sha256(path) != args.selection_sha256:
        raise ValueError('Learning requires the exact frozen pre-SGD task/query selection SHA')
    selection = json.loads(path.read_text())
    producer_plan = json.loads((root / 'plan.json').read_text())
    if producer_plan != contract:
        admission = json.loads((root.parent / 'screen-to-learning-source.json').read_text())
        if (admission['screen_source'] != args.screen_source or admission['learning_source'] != args.source_commit
                or admission['original_screen_plan_sha256'] != onset.sha256(root / 'plan.json')
                or admission['frozen_selection_sha256'] != args.selection_sha256
                or {k: v for k, v in producer_plan.items() if k != 'source_commit'} !=
                   {k: v for k, v in contract.items() if k != 'source_commit'}):
            raise ValueError('Explicit frozen screen-to-learning code ancestry differs')
    if (selection['format'] != phase.FORMAT or not selection['selected_before_any_sgd']
            or selection['optimizer_updates'] != 0 or selection['task_id'] not in phase.CANDIDATES
            or selection['query_bank'] != phase.query_bank(selection['task_id'])
            or selection['plan_sha256'] != onset.sha256(root / 'plan.json')):
        raise ValueError('Selected task/query bank/producer plan differs')
    task = selection['task_id']
    learning_contract = dict(contract, selected_task=task, frozen_selection_sha256=args.selection_sha256,
        original_screen_plan_sha256=onset.sha256(root / 'plan.json'),
        frozen_query_bank=selection['query_bank'], no_selection_after_sgd=True,
        initial_native_readmission_band=[2, 8], initial_native_off_band='stop without SGD or task substitution',
        primary_endpoint='Fixed500k learned success count minus final native on matched20 query cases',
        persistence_rule='At least6/20 gain over final native at each of the final3 learned panels; no best-checkpoint selection')
    np.random.seed(0); random.seed(0)
    learner, sampler = onset.OfficialSAC(args.dsrl_root, 0), np.random.default_rng(0)
    server = server_factory(task)
    try:
        output = root / f'case-learn-{task:02d}'
        result = run_learning(server, motor, learner, sampler, output=output, root=root,
            contract=learning_contract, resume=args.resume, task_id=task,
            query_cases=selection['query_bank'], deadline=deadline, storage_root=root.parent)
        result['endpoint_analysis'] = endpoint_summary(result)
        onset.write_json(root / 'learning-index.json', result)
        return result
    finally:
        server.close()
