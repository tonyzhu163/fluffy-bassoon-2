"""Instrument the native RLinf learner without replacing PPO or sampling.

Each episode uses a fresh deterministic native constructor. This makes the
next complete-iteration boundary recoverable without pickling MuJoCo state.
Actor/Adam/LR/RNG use RLinf DCP; sidecars retain rollout RNG and case cursors.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker

from .rlinf_native_ppo import ActionJournal, cpu_tree, restore_rng, snapshot_rng, tensor_hash, write
from .rlinf_native_ppo import json_metrics, sha
from .rlinf_native_ppo_runtime import PilotActor, PilotRollout, PilotRunner, save_tensor
from .rlinf_robocasa_flow_ppo import (GlobalActionJournal, RESET_CONTROL_RESERVE, checkpoint_weight_export, first_success_reward,
    flow_contract, flow_episode_ids, flow_logprob_parity, flow_value_parity,
    keep_frozen_image_encoder_in_eval, train_case_seeds, verify_grpo_group_reset)
from .rlinf_robocasa_ppo_env import counted_native_factory, reconfigure_or_create_subproc_env
from .robocasa_pairing import canonical_case, native_fingerprint, public_snapshot, stable_counter_regions


def _write_chunk_archive(path, payload):
    payload['current_observation_sha256'] = tensor_hash(payload['current_observation'])
    with gzip.open(path, 'xb', compresslevel=1) as stream:
        torch.save(payload, stream)


def unwrapped(env):
    while '_flow_round' not in vars(env):
        env = env.env
    return env


def fresh_factory(factory, seed, physics_tick_limit):
    def create():
        random.seed(seed)
        np.random.seed(seed)
        stable_counter_regions()
        env = counted_native_factory(factory, physics_tick_limit=physics_tick_limit)()
        reset = env.reset

        def recorded_reset(*args, **kwargs):
            result = reset(*args, **kwargs)
            env._flow_physical = native_fingerprint(env)
            return result
        env.reset = recorded_reset
        return env
    return create


def native_class(original, owner):
    class FlowRoboCasa(original):
        def __init__(self, *args, **kwargs):
            self._flow_round = -1
            super().__init__(*args, **kwargs)

        def _init_env(self):
            self._refresh_task_context()  # construct only actual episode banks

        def _calc_step_reward(self, terminations):
            return first_success_reward(terminations, self.success_once, self.cfg.reward_coef)

        def reset(self, env_idx=None, options=None):
            if env_idx is not None or options is not None:
                raise ValueError('Flow PPO uses complete fixed-horizon vector resets')
            if hasattr(owner, '_flush_flow_io'):
                owner._flush_flow_io()
            self._flow_round += 1
            is_eval = bool(self.cfg.is_eval)
            contract = owner._flow_contract
            # Fresh TRAIN episodes for every round of every iteration; one fixed eval bank.
            if is_eval:
                self.env_seeds = [contract['eval_seed'] + owner._rank * self.num_envs + i
                                  for i in range(self.num_envs)]
            else:
                self.env_seeds = train_case_seeds(contract, iteration=owner._flow_iteration,
                    round_index=self._flow_round, rank=owner._rank, local_envs=self.num_envs)
            self._flow_episode_ids = flow_episode_ids(contract['task'], self.env_seeds,
                group_size=1 if is_eval else contract['group_size'])
            self.task_ids = np.zeros(self.num_envs, dtype=np.int32)
            self._refresh_task_context()
            self._is_start = True
            token = owner._flow_journal.begin(self.num_envs * RESET_CONTROL_RESERVE,
                phase=owner._flow_phase + '/native_constructor_reset')
            fns = [fresh_factory(fn, seed, contract['physics_tick_limit']) for fn, seed in
                zip(super().get_env_fns(), self.env_seeds, strict=True)]
            current = getattr(self, 'env', None)
            if is_eval and owner.cfg.native_pilot.get('fresh_eval_processes', False) and current is not None:
                current.close()
                current = None
            self.env = reconfigure_or_create_subproc_env(current, fns)
            observation, info = super().reset()
            counts = self.env.get_env_attr('_flora_accounting')
            owner._flow_journal.finish(token, sum(c['reset_controls'] for c in counts))
            self._flow_steps = 0
            self._flow_current = cpu_tree(observation)
            self._flow_first_success = [None] * self.num_envs
            phase = owner._flow_phase
            self._flow_key = f'{phase}-i{owner._flow_iteration:03d}-round{self._flow_round:02d}'
            self._flow_root = Path(owner.cfg.native_pilot.output) / 'environment' / f'rank-{owner._rank:02d}'
            case = dict(task=contract['task'], seeds=self.env_seeds, split=str(self.cfg.split),
                        horizon=contract['horizon'])
            snapshot = public_snapshot(observation, case, self.env.get_env_attr('_flow_physical'))
            if not is_eval and contract['group_size'] > 1:
                verify_grpo_group_reset(snapshot, contract['group_size'])
            initial = self._flow_root / (self._flow_key + '-initial.pt')
            self._flow_initial = save_tensor(initial, snapshot)
            if is_eval:
                canonical, report = canonical_case(Path(owner.cfg.native_pilot.output) / 'canonical-eval' /
                    f'rank-{owner._rank:02d}', snapshot, save=save_tensor,
                    load=lambda p: torch.load(p, map_location='cpu', weights_only=False),
                    source=dict(source=os.environ['TASK_CODE_PIN'], policy_iteration=owner._flow_iteration))
                write(initial.with_suffix('.json'), dict(canonical=str(canonical), report=report,
                      initial=self._flow_initial))
            return observation, info

        def step(self, actions=None, auto_reset=True):
            result = super().step(actions, auto_reset=False)
            if actions is not None:
                self._flow_steps += 1
                for i, succeeded in enumerate(self.success_once):
                    if succeeded and self._flow_first_success[i] is None:
                        self._flow_first_success[i] = self._flow_steps
            horizon = owner._flow_contract['horizon']
            if self._flow_steps == horizon:
                counts = self.env.get_env_attr('_flora_accounting')
                assert all(c['controls'] - c['reset_controls'] == horizon for c in counts)
                finals = [bool(info.get('success', False)) for info in self.current_info_list]
                rows = [dict(episode_id=episode_id, seed=seed, success_once=first is not None,
                    first_success_control=first,
                    success_at_end=final, native_counts=count)
                    for episode_id, seed, first, final, count in zip(
                        self._flow_episode_ids, self.env_seeds, self._flow_first_success,
                        finals, counts, strict=True)]
                write(self._flow_root / (self._flow_key + '-episodes.json'),
                    dict(completed=True, phase=owner._flow_phase, iteration=owner._flow_iteration,
                         rank=owner._rank, initial=self._flow_initial, rows=rows))
            return result

        def chunk_step(self, actions):
            if time.time() > float(owner.cfg.native_pilot.deadline_unix) - 180:
                raise TimeoutError('Preserve last full boundary before allocation deadline')
            requested = int(actions.shape[0] * actions.shape[1])
            token = owner._flow_journal.begin(requested, phase=owner._flow_phase)
            before = self._flow_current
            result = super().chunk_step(actions)
            owner._flow_journal.finish(token, requested)
            contract = owner._flow_contract
            after = cpu_tree(result[0][-1])
            last = self._flow_steps == contract['horizon']
            path = self._flow_root / (self._flow_key + f'-chunk{self._flow_steps // contract["chunk"]:02d}.pt.gz')
            payload = dict(phase=owner._flow_phase, iteration=owner._flow_iteration,
                round=self._flow_round, seeds=list(self.env_seeds),
                episode_ids=list(self._flow_episode_ids),
                current_observation=before,
                decoded_actions=cpu_tree(actions), rewards=cpu_tree(result[1]),
                terminations=cpu_tree(result[2]), truncations=cpu_tree(result[3]),
                final_info=cpu_tree(result[4][-1]),
                # The horizon observation is never a decision input; value targets need it.
                terminal_observation=after if last else None)
            if hasattr(owner, '_submit_flow_io'):
                owner._submit_flow_io(_write_chunk_archive, path, payload)
            else:
                _write_chunk_archive(path, payload)
            self._flow_current = after
            return result
    return FlowRoboCasa


class FlowActor(PilotActor):
    def save_checkpoint(self, save_path, step=0):
        final = int(self.cfg.runner.max_epochs)
        maximum = int(self.cfg.runner.get('max_steps', -1))
        if maximum >= 0:
            final = min(final, maximum)
        with checkpoint_weight_export(self._cfg.fsdp_config, step=int(step), final_step=final,
                export_steps=self.cfg.native_pilot.get('full_model_export_steps')):
            # The upstream writer always saves complete model/Adam/LR/RNG DCP.
            # Its optional full_weights.pt is an additional inference copy.
            return super().save_checkpoint(save_path, step)

    def train_micro_batch(self, micro_batch, metrics, *, is_last):
        if self._flow_first_micro_checked:
            return super().train_micro_batch(micro_batch, metrics, is_last=is_last)
        keep_frozen_image_encoder_in_eval(self.model)

        def check(module, inputs, output):
            report = flow_logprob_parity(output['logprobs'], micro_batch['prev_logprobs'],
                action_dim=int(self.cfg.actor.model.action_dim))
            value_head = bool(self.cfg.actor.model.get('add_value_head', True))
            if value_head and output.get('values') is not None and micro_batch.get('prev_values') is not None:
                value = flow_value_parity(output['values'], micro_batch['prev_values'])
                report = dict(report, value=value, passed=report['passed'] and value['passed'])
            write(Path(self.cfg.native_pilot.output) / 'logprob-parity' /
                f'iteration-{self.version + 1:03d}-rank-{self._rank:02d}.json',
                dict(report, value_head_enabled=value_head,
                     optimizer_steps=self.optimizer_steps, before_backward=True))
            self._flow_first_micro_checked = True
            if not report['passed']:
                raise RuntimeError(f'Unchanged-policy logprob parity failed before SGD: {report}')

        hook = self.model.register_forward_hook(check)
        try:
            return super().train_micro_batch(micro_batch, metrics, is_last=is_last)
        finally:
            hook.remove()

    def run_training(self):
        started = time.monotonic()
        before = self.optimizer_steps
        self._flow_first_micro_checked = False
        batch_shape = tuple(self.rollout_batch['prev_logprobs'].shape)
        metrics = EmbodiedFSDPActor.run_training(self)
        expected = flow_contract(self.cfg)['optimizer_steps_per_update']
        assert self.optimizer_steps - before == expected, 'Global PPO optimizer schedule changed'
        losses = {k: float(v) for k, v in metrics.items() if 'loss' in k or 'grad_norm' in k}
        assert losses and all(np.isfinite(v) for v in losses.values()), losses
        unchanged = self._pilot_frozen_before == self._pilot_frozen_signature()
        assert unchanged, 'Frozen VLM changed during action-expert PPO'
        write(Path(self.cfg.native_pilot.output) / 'updates' /
            f'iteration-{self.version + 1:03d}-rank-{self._rank:02d}.json',
            dict(completed=True, rank=self._rank, producing_policy_iteration=self.version,
                 optimizer_steps_before=before, optimizer_steps_after=self.optimizer_steps,
                 batch_shape=batch_shape, losses=losses, metrics={k: float(v) for k, v in metrics.items()},
                 training_seconds=time.monotonic() - started,
                 cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                 cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                 frozen_vlm_unchanged=unchanged))
        return metrics


class FlowRollout(PilotRollout):
    def pilot_training_rng(self, *, restore=False):
        self._flush_pilot_io()
        if restore:
            restore_rng(self._flow_training_rng)
        else:
            self._flow_training_rng = snapshot_rng()
        return dict(role=self.pilot_role, rank=self._rank, restored=restore)

    def pilot_restore(self, directory):
        self._flush_pilot_io()
        result = super().pilot_restore(directory)
        state = torch.load(Path(directory) / f'rollout-{self._rank:02d}.pt',
                           map_location='cpu', weights_only=False)
        if 'migration_rng_seed' in state:
            seed = state['migration_rng_seed']
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            self._pilot_rng = snapshot_rng()
            result['new_world_rng_seed'] = seed
        return result


class FlowEnvironment(EnvWorker):
    def init_worker(self):
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)
        self._flow_phase = 'initialization'
        self._flow_iteration = 0
        self._flow_contract = flow_contract(self.cfg)
        self._flow_io_pool = ThreadPoolExecutor(max_workers=2)
        self._flow_io_futures = []
        journal = GlobalActionJournal if self.cfg.native_pilot.get('global_action_escrow', False) else ActionJournal
        self._flow_journal = journal(self.cfg.native_pilot.accounting_root,
            rank=self._rank, world=self._world_size, attempt=self.cfg.native_pilot.attempt,
            maximum_actions=self._flow_contract['maximum_actions'])
        return super().init_worker()

    def _submit_flow_io(self, fn, *args):
        pool = getattr(self, '_flow_io_pool', None)
        if pool is None:
            return fn(*args)
        pending = []
        for future in self._flow_io_futures:
            if future.done():
                future.result()
            else:
                pending.append(future)
        pending.append(pool.submit(fn, *args))
        self._flow_io_futures = pending

    def _flush_flow_io(self):
        for future in getattr(self, '_flow_io_futures', ()):
            future.result()
        self._flow_io_futures = []

    def _setup_env_and_wrappers(self, env_cls, env_cfg, num_envs_per_stage):
        assert self.stage_num == 1
        return super()._setup_env_and_wrappers(native_class(env_cls, self), env_cfg, num_envs_per_stage)

    def set_global_step(self, step):
        self._flush_flow_io()
        self._flow_iteration = int(step)
        for wrapped in self.env_list:
            unwrapped(wrapped)._flow_round = -1
        return super().set_global_step(step)

    def pilot_phase(self, phase, *, policy_iteration=None):
        self._flush_flow_io()
        self._flow_phase = phase
        if policy_iteration is not None:
            self._flow_iteration = int(policy_iteration)
        for wrapped in self.eval_env_list:
            unwrapped(wrapped)._flow_round = -1
        return dict(rank=self._rank, phase=phase, charged=self._flow_journal.charged())

    def pilot_rng(self, *, restore=False):
        self._flush_flow_io()
        if restore:
            restore_rng(self._flow_rng)
        else:
            self._flow_rng = snapshot_rng()
        return dict(rank=self._rank, restored=restore)

    def pilot_training_rng(self, *, restore=False):
        self._flush_flow_io()
        if restore:
            restore_rng(self._flow_training_rng)
        else:
            self._flow_training_rng = snapshot_rng()
        return dict(rank=self._rank, restored=restore)

    def pilot_capture(self, directory, **kwargs):
        self._flush_flow_io()
        samplers = []
        for mode, envs in (('train', self.env_list), ('eval', self.eval_env_list)):
            for ordinal, wrapped in enumerate(envs):
                env = unwrapped(wrapped)
                samplers.append(dict(mode=mode, ordinal=ordinal, seed=env.seed,
                    generator=deepcopy(env._generator.bit_generator.state),
                    round=env._flow_round, task_ids=env.task_ids.copy(),
                    ordered_task_cursor=env._ordered_task_cursor))
        return save_tensor(Path(directory) / f'env-{self._rank:02d}.pt',
            dict(role='env', rank=self._rank, rng=snapshot_rng(), phase=self._flow_phase,
                 iteration=self._flow_iteration, trajectory_step=self._trajectory_step,
                 samplers=samplers, paired_eval_rng=getattr(self, '_flow_rng', None),
                 recovery='Whole completed iteration; fresh seeded native constructors at next reset'))

    def pilot_restore(self, directory):
        self._flush_flow_io()
        state = torch.load(Path(directory) / f'env-{self._rank:02d}.pt', map_location='cpu', weights_only=False)
        assert state['rank'] == self._rank
        self._flow_phase, self._flow_iteration = state['phase'], state['iteration']
        self._trajectory_step = state['trajectory_step']
        self._flow_rng = state['paired_eval_rng']
        for row in state['samplers']:
            envs = self.env_list if row['mode'] == 'train' else self.eval_env_list
            env = unwrapped(envs[row['ordinal']])
            assert env.seed == row['seed']
            env._generator.bit_generator.state = row['generator']
            env._flow_round, env.task_ids = row['round'], row['task_ids']
            env._ordered_task_cursor = row['ordered_task_cursor']
            env._refresh_task_context()
        self._prefetched_train_bootstrap = None
        restore_rng(state['rng'])
        return dict(role='env', rank=self._rank, restored=True)


class FlowRunner(PilotRunner):
    def _periodic_evaluation(self, output):
        self.rollout.pilot_training_rng().wait()
        self.env.pilot_training_rng().wait()
        self.update_rollout_weights()
        self.rollout.pilot_rng(restore=True).wait()
        self.env.pilot_rng(restore=True).wait()
        phase = f'eval_step_{self.global_step:03d}'
        self.env.pilot_phase(phase, policy_iteration=self.global_step).wait()
        self.rollout.pilot_evaluation_iteration(self.global_step).wait()
        periodic = self.evaluate()
        write(output / 'evaluations' / f'step-{self.global_step:03d}.json',
            dict(completed=True, global_step=self.global_step, phase=phase, metrics=json_metrics(periodic)))
        self.rollout.pilot_training_rng(restore=True).wait()
        self.env.pilot_training_rng(restore=True).wait()
        self.env.pilot_phase('training', policy_iteration=self.global_step).wait()

    def _log_step_metrics(self, *args, **kwargs):
        result = EmbodiedRunner._log_step_metrics(self, *args, **kwargs)
        output = Path(self.cfg.native_pilot.output)
        requested = (output / 'CHECKPOINT_AND_EXIT').exists()
        first_after_restore = self.global_step == getattr(self, '_flow_resume_step', 0) + 1
        contract = flow_contract(self.cfg)
        interval = contract['checkpoint_interval']
        if requested or first_after_restore or self.global_step % interval == 0 or self.global_step == self.max_steps:
            self._capture_boundary()
        eval_interval = contract['eval_interval']
        if not requested and 0 < eval_interval < self.max_steps and 0 < self.global_step < self.max_steps and self.global_step % eval_interval == 0:
            self._periodic_evaluation(output)
        write(output / 'progress.json',
            dict(completed_iterations=self.global_step, target_iterations=self.max_steps,
                 checked_unix=time.time(), operator_checkpoint_exit=requested))
        if requested:
            write(output / 'operator-checkpoint-exit.json', dict(completed=True,
                global_step=self.global_step, requested_marker='CHECKPOINT_AND_EXIT',
                scope='All update handles completed; native DCP and all worker sidecars saved; no next rollout'))
            self._finish_run()
            raise SystemExit(0)
        return result

    def run(self):
        self._flow_resume_step = self.global_step
        contract = flow_contract(self.cfg)
        output = Path(self.cfg.native_pilot.output)
        write(output / 'flow-contract.json', contract)
        write(output / 'resolved-config.json', OmegaConf.to_container(self.cfg, resolve=True))
        if self.cfg.runner.get('resume_dir'):
            reference_path = self.cfg.native_pilot.before_reference
            if not reference_path or sha(reference_path) != self.cfg.native_pilot.before_reference_sha256:
                raise ValueError('Resume requires the verified original before panel')
            reference = json.loads(Path(reference_path).read_text())
            assert reference['completed'] and reference['cases'] == contract['eval_cases']
            assert Path(reference['checkpoint']) == Path(self.cfg.runner.resume_dir)
            assert .05 <= reference['metrics']['success_once'] <= .70
            for row in reference['episode_refs']:
                assert sha(row['path']) == row['sha256']
            shutil.copytree(reference['canonical_directory'], output / 'canonical-eval')
            write(output / 'before-evaluation.json', dict(completed=True,
                metrics=reference['metrics'], reused_reference=dict(path=reference_path,
                    sha256=self.cfg.native_pilot.before_reference_sha256), new_evaluation_controls=0))
            for group in (self.actor, self.rollout, self.env):
                group.pilot_restore(str(Path(self.cfg.runner.resume_dir) / 'pilot')).wait()
            pending = reference.get('pending_evaluation_step')
            if pending is not None:
                if (pending != self.global_step or not 0 < pending < self.max_steps
                        or contract['eval_interval'] <= 0 or pending % contract['eval_interval']):
                    raise ValueError('Pending DEV evaluation must match a scheduled restored boundary')
                self._periodic_evaluation(output)
        else:
            self.update_rollout_weights()
            for group in (self.actor, self.rollout, self.env):
                group.pilot_rng().wait()
            self._capture_boundary()
            self.env.pilot_phase('before', policy_iteration=0).wait()
            self.rollout.pilot_evaluation_iteration(0).wait()
            before = self.evaluate()
            write(output / 'before-evaluation.json', dict(completed=True, metrics=json_metrics(before)))
            success = float(before['success_once'])
            if not .05 <= success <= .70:
                raise RuntimeError(f'Before-panel task admission requires human decision: success_once={success}')
            for group in (self.actor, self.rollout, self.env):
                group.pilot_rng(restore=True).wait()
        self.env.pilot_phase('training').wait()
        EmbodiedRunner.run(self)
        final = contract['iterations']
        assert self.global_step == final
        self.update_rollout_weights()
        self.rollout.pilot_rng(restore=True).wait()
        self.env.pilot_rng(restore=True).wait()
        self.env.pilot_phase('after', policy_iteration=final).wait()
        self.rollout.pilot_evaluation_iteration(final).wait()
        after = self.evaluate()
        write(output / 'after-evaluation.json', dict(completed=True, global_step=final, metrics=json_metrics(after)))
