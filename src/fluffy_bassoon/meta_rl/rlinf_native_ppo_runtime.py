"""Narrow worker/runner extensions of pinned official RLinf native PPO.

No PPO, action sampling, reward, GAE or loss implementation is replaced.
Sidecars are whole-iteration recovery, not an in-flight physics checkpoint.
"""
import gzip
import hashlib
import os
from pathlib import Path
import time

import torch
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

from .rlinf_native_ppo import (ActionJournal, cpu_tree, restore_rng, sha,
    snapshot_rng, tensor_hash, validate_recipe, write)


def save_tensor(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary); temporary.replace(path)
    return dict(path=str(path), sha256=sha(path))


def base_env(env):
    while 'reset_state_ids' not in vars(env):
        if 'env' not in vars(env):
            raise RuntimeError('Unexpected wrapper; cannot identify native RoboTwin sampler')
        env = env.env
    return env


def producing_iteration(worker):
    # Official Actor.set_global_step writes `version`, not `global_step`.
    # Prefer our explicit logical tag; use the actual native version as fallback.
    return int(getattr(worker, '_pilot_producing_policy_iteration',
        getattr(worker, 'version', getattr(worker, 'global_step', 0))))


class SidecarMixin:
    pilot_role = None

    def pilot_capture(self, directory, *, retain_rollout=False):
        """Called only after all iteration handles have completed."""
        state = dict(format='rlinf_native_ppo_worker_v1', role=self.pilot_role,
            rank=self._rank, rng=snapshot_rng(), global_step=getattr(self, 'global_step', None),
            version=getattr(self, 'version', None), paired_eval_rng=getattr(self, '_pilot_rng', None))
        state['producing_policy_iteration'] = producing_iteration(self)
        if self.pilot_role == 'env':
            if self._prefetched_train_bootstrap is not None:
                raise RuntimeError('Cannot checkpoint an in-flight environment bootstrap')
            state['trajectory_step'] = self._trajectory_step
            state['initial_reset_records'] = cpu_tree(self._pilot_first_resets)
            state['samplers'] = []
            for mode, envs in (('train', self.env_list), ('eval', self.eval_env_list)):
                for ordinal, wrapped in enumerate(envs):
                    env = base_env(wrapped)
                    state['samplers'].append(dict(mode=mode, ordinal=ordinal,
                        reset_state_ids=env.reset_state_ids.clone(),
                        current_seed_index=env._current_seed_index,
                        generator=env._generator.get_state(),
                        success_seeds=env.success_seeds, seed=env.seed))
            # auto_reset=False source collection begins each round with reset;
            # previous physics/cached observations must never be consumed.
            state['last_obs_list'] = cpu_tree(self.last_obs_list)
            state['last_intervened_info_list'] = cpu_tree(self.last_intervened_info_list)
            state['train_prev_done'] = cpu_tree(getattr(self, 'train_prev_done', None))
            state['eval_prev_done'] = cpu_tree(getattr(self, 'eval_prev_done', None))
        if self.pilot_role == 'actor':
            state['rollout_batch'] = cpu_tree(getattr(self, 'rollout_batch', None)) if retain_rollout else None
            state['optimizer_boundary'] = True
            state['grad_scaler'] = self.grad_scaler.state_dict()
            state['optimizer_steps'] = self.optimizer_steps
        return save_tensor(Path(directory) / f'{self.pilot_role}-{self._rank:02d}.pt', state)

    def pilot_restore(self, directory):
        state = torch.load(Path(directory) / f'{self.pilot_role}-{self._rank:02d}.pt',
            map_location='cpu', weights_only=False)
        if state['role'] != self.pilot_role or state['rank'] != self._rank:
            raise ValueError('Worker recovery role/rank differs')
        if state['global_step'] is not None:
            self.global_step = state['global_step']
        if state['version'] is not None:
            self.version = state['version']
        self._pilot_producing_policy_iteration = state.get('producing_policy_iteration',
            state['version'] if state['version'] is not None else 0)
        if state['paired_eval_rng'] is not None:
            self._pilot_rng = state['paired_eval_rng']
        if self.pilot_role == 'env':
            self._trajectory_step = state['trajectory_step']
            self._pilot_first_resets = state['initial_reset_records']
            for row in state['samplers']:
                envs = self.env_list if row['mode'] == 'train' else self.eval_env_list
                env = base_env(envs[row['ordinal']])
                same_seeds = (env.success_seeds is None and row['success_seeds'] is None) or (
                    env.success_seeds is not None and row['success_seeds'] is not None
                    and torch.equal(env.success_seeds, row['success_seeds']))
                if env.seed != row['seed'] or not same_seeds:
                    raise ValueError('Source/evaluation seed inventory changed on restore')
                env.reset_state_ids = row['reset_state_ids'].clone()
                env._current_seed_index = row['current_seed_index']
                env._generator.set_state(row['generator'])
            # Explicit source reset at next bootstrap renders the restored IDs.
            self._prefetched_train_bootstrap = None
        if self.pilot_role == 'actor':
            self.grad_scaler.load_state_dict(state['grad_scaler'])
            self.optimizer_steps = state['optimizer_steps']
        restore_rng(state['rng'])  # Last: initialization/restoration consumes no saved draws.
        return dict(role=self.pilot_role, rank=self._rank, restored=True,
            snapshot_sha256=sha(Path(directory) / f'{self.pilot_role}-{self._rank:02d}.pt'))

    def pilot_rng(self, *, restore=False):
        if restore:
            restore_rng(self._pilot_rng)
        else:
            self._pilot_rng = snapshot_rng()
        return dict(role=self.pilot_role, rank=self._rank, restored=restore)


class PilotActor(SidecarMixin, EmbodiedFSDPActor):
    pilot_role = 'actor'

    def set_global_step(self, global_step):
        self._pilot_producing_policy_iteration = int(global_step)
        return super().set_global_step(global_step)

    def compute_advantages_and_returns(self):
        metrics = super().compute_advantages_and_returns()
        # Official run_training subsequently shuffles/flattens and discards
        # final bootstrap rows. Capture the actual complete batch BEFORE it.
        policy_iteration = producing_iteration(self)
        path = Path(self.cfg.native_pilot.output) / 'training-rollouts' / (
            f'iteration-{policy_iteration+1:04d}-policy-{policy_iteration:04d}-rank-{self._rank:02d}.pt')
        if path.exists() or path.with_suffix('.json').exists():
            raise RuntimeError('Chronological rollout archive already exists; never overwrite a policy version')
        value = dict(format='rlinf_native_ppo_chronological_rollout_v1',
            collection_iteration=policy_iteration+1, producing_policy_iteration=policy_iteration,
            worker_policy_version=getattr(self, 'version', None), rank=self._rank,
            optimizer_steps_before=self.optimizer_steps, rng_before_sgd=snapshot_rng(),
            rollout_epoch=int(self.cfg.env.train.rollout_epoch),
            layout='time-major [chunk_index, round_index * local_env_count + env_index, ...]; terminal bootstrap rows retained',
            rollout_batch=cpu_tree(self.rollout_batch),
            evaluation_data=False, source6_pretraining_admission=False)
        receipt = save_tensor(path, value)
        from .rlinf_native_ppo_data import FORMAT, shape_tree
        write(path.with_suffix('.json'), dict(receipt, completed=True, capture_format=FORMAT,
            collection_iteration=policy_iteration+1, producing_policy_iteration=policy_iteration,
            rank=self._rank, optimizer_steps_before=self.optimizer_steps,
            fields=sorted(self.rollout_batch), field_shapes=shape_tree(self.rollout_batch),
            evaluation_data=False, resolved_config_path=str(Path(self.cfg.native_pilot.output)/'resolved-native-config.json'),
            retention='unshuffled/untrimmed original batch plus computed advantages/returns'))
        return metrics

    def _pilot_frozen_signature(self):
        digest, count, elements = hashlib.sha256(), 0, 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                array = parameter.detach().cpu().contiguous()
                digest.update(name.encode()); digest.update(str(array.dtype).encode())
                digest.update(str(tuple(array.shape)).encode())
                digest.update(array.reshape(-1).view(torch.uint8).numpy().tobytes())
                count += 1; elements += array.numel()
        if not elements:
            raise RuntimeError('No frozen VLM parameters are visible in the native FSDP actor')
        return dict(sha256=digest.hexdigest(), local_parameter_tensors=count, local_elements=elements)

    def init_worker(self):
        result = super().init_worker()
        self._pilot_frozen_before = self._pilot_frozen_signature()
        write(Path(self.cfg.native_pilot.output) / 'actor-invariants' /
            f'rank-{self._rank:02d}-before.json', self._pilot_frozen_before)
        self._pilot_gradient_recorded = False
        return result

    def train_micro_batch(self, *args, **kwargs):
        result = super().train_micro_batch(*args, **kwargs)
        if not self._pilot_gradient_recorded:
            rows = []
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    rows.append(dict(name=name, original_names=getattr(parameter, '_fqns', None),
                        local_gradient_l2=float(parameter.grad.detach().float().norm().item()),
                        elements=parameter.grad.numel()))
            write(Path(self.cfg.native_pilot.output) / 'actor-invariants' /
                f'rank-{self._rank:02d}-first-backward.json', dict(parameters=rows,
                    attribution='Read original_names to distinguish expert-1/policy from value-head; no aggregate-only inference'))
            self._pilot_gradient_recorded = True
        return result

    def run_training(self):
        mask = self.rollout_batch.get('loss_mask')
        metrics = dict(rank=self._rank, producing_policy_iteration=producing_iteration(self),
            gradient_accumulation=self.gradient_accumulation,
            local_valid_chunk_slots=None if mask is None else int(mask.sum().item()),
            local_total_mask_slots=None if mask is None else int(mask.numel()),
            official_loss_weighting='Unmodified c605 effective loss reducer; masked_mean_ratio and rank-local GAE normalization, not a claimed global valid-token mean')
        write(Path(self.cfg.native_pilot.output) / 'loss-masks' /
            f'step-{producing_iteration(self)}-rank-{self._rank:02d}.json', metrics)
        result = super().run_training()
        after = self._pilot_frozen_signature()
        unchanged = self._pilot_frozen_before == after
        write(Path(self.cfg.native_pilot.output) / 'actor-invariants' /
            f'rank-{self._rank:02d}-step-{producing_iteration(self)}-after.json',
            dict(before=self._pilot_frozen_before, after=after, unchanged=unchanged))
        if not unchanged:
            raise RuntimeError('Frozen native VLM parameters changed during PPO')
        return result


class PilotRollout(SidecarMixin, MultiStepRolloutWorker):
    pilot_role = 'rollout'

    def init_worker(self):
        result = super().init_worker()
        import rlinf.models.embodiment.openpi.sampling.rl_sampler as sampler
        from .rlinf_native_ppo_data import record_gaussian_parameters
        original = sampler.sample_mean_var
        self._pilot_sampling_steps = None
        self._pilot_prediction_ordinal = 0
        self._pilot_terminal_only = False
        self._pilot_eval_iteration = 0

        sampler.sample_mean_var = record_gaussian_parameters(original,
            lambda: self._pilot_sampling_steps, cpu_tree)
        return result

    def set_global_step(self, global_step):
        self._pilot_producing_policy_iteration = int(global_step)
        return super().set_global_step(global_step)

    def pilot_evaluation_iteration(self, iteration):
        self._pilot_eval_iteration = int(iteration)
        return dict(rank=self._rank, evaluation_policy_iteration=int(iteration))

    def _publish_env_part(self, env_part, *args, **kwargs):
        previous = self._pilot_terminal_only
        self._pilot_terminal_only = bool(env_part.requires_inference)
        try:
            return super()._publish_env_part(env_part, *args, **kwargs)
        finally:
            self._pilot_terminal_only = previous

    def _predict_rollout_actions(self, env_obs, mode='train', **kwargs):
        before = snapshot_rng()
        self._pilot_sampling_steps = []
        try:
            actions, result = super()._predict_rollout_actions(env_obs, mode=mode, **kwargs)
            after = snapshot_rng()
            inputs = result.get('forward_inputs', {})
            policy_iteration = (self._pilot_eval_iteration if mode == 'eval'
                else producing_iteration(self))
            record = dict(format='rlinf_native_ppo_actual_sampling_v1', mode=mode,
                evaluation_data=mode == 'eval', terminal_value_only=self._pilot_terminal_only,
                collection_iteration=policy_iteration+1 if mode == 'train' else None,
                producing_policy_iteration=policy_iteration,
                rank=self._rank, prediction_ordinal=self._pilot_prediction_ordinal,
                public_observation_sha256=tensor_hash(env_obs),
                actions_sha256=tensor_hash(actions), rng_before=before, rng_after=after,
                actual_gaussian_parameters=self._pilot_sampling_steps,
                forward_sampling_fields=cpu_tree({key: inputs[key] for key in
                    ('chains', 'denoise_inds', 'action', 'model_action') if key in inputs}),
                old_logprobs=cpu_tree(result.get('prev_logprobs')),
                old_values=cpu_tree(result.get('prev_values')),
                model_actions=cpu_tree(result.get('model_actions')),
                initial_gaussian_path='forward_sampling_fields.chains[:,0]',
                noise_semantics='Actual mean/std objects and full realized chain; later Gaussian draws recoverable from saved sampler RNG/source, no RNG-consuming replay performed',
                frozen_sft_steering_target_compatible=False)
            path = Path(self.cfg.native_pilot.output) / 'sampling' / f'rank-{self._rank:02d}' / (
                f'{self._pilot_prediction_ordinal:05d}.pt.gz')
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, 'xb', compresslevel=1) as handle:
                torch.save(record, handle)
            self._pilot_prediction_ordinal += 1
            return actions, result
        finally:
            self._pilot_sampling_steps = None


class PilotEnvironment(SidecarMixin, EnvWorker):
    pilot_role = 'env'

    def set_global_step(self, global_step):
        self._pilot_producing_policy_iteration = int(global_step)
        self._pilot_next_source_round = 0
        return super().set_global_step(global_step)

    def init_worker(self):
        # Install only in this owned EnvWorker process, before native creation.
        import rlinf.envs.sim.robotwin.robotwin_env as module
        original = module.RoboTwinEnv
        owner = self
        self._pilot_phase = 'initialization'
        self._pilot_reset_ordinal = 0
        self._pilot_trace_ordinal = 0
        self._pilot_first_resets = {}
        self._pilot_root = Path(self.cfg.native_pilot.output)
        self._pilot_journal = ActionJournal(self.cfg.native_pilot.accounting_root,
            rank=self._rank, world=int(self.cfg.native_pilot.world_size),
            attempt=self.cfg.native_pilot.attempt)

        class RecordedRoboTwin(original):
            def __init__(env, *args, **kwargs):
                # Native constructor starts the vector simulator. Reserve ten
                # steps/scene as explicit unmeasured setup allowance; never
                # claim these conservative allowances are measured actions.
                num = kwargs.get('num_envs', args[1] if len(args) > 1 else None)
                slot = owner._pilot_journal.begin(int(num) * 10, phase='constructor-allowance')
                super().__init__(*args, **kwargs)
                owner._pilot_journal.finish(slot, int(num) * 10)
                env._pilot_episode_serial = torch.zeros(int(num), dtype=torch.long)
                env._pilot_primitive_time = torch.zeros(int(num), dtype=torch.long)
                env._pilot_current_observation = None
                env._pilot_chunk_index = 0
                env._pilot_round_index = None

            def reset(env, env_idx=None, env_seeds=None):
                count = env.num_envs if env_idx is None else len(env_idx)
                slot = owner._pilot_journal.begin(int(count) * 10,
                    phase=owner._pilot_phase + '/reset-allowance')
                result = super().reset(env_idx=env_idx, env_seeds=env_seeds)
                owner._pilot_journal.finish(slot, int(count) * 10)
                indices = list(range(env.num_envs)) if env_idx is None else list(env_idx)
                env._pilot_episode_serial[indices] += 1
                env._pilot_primitive_time[indices] = 0
                env._pilot_current_observation = cpu_tree(result[0])
                if env_idx is None:
                    env._pilot_chunk_index = 0
                    if not env.cfg.is_eval:
                        env._pilot_round_index = getattr(owner, '_pilot_next_source_round', 0)
                        owner._pilot_next_source_round = env._pilot_round_index + 1
                reset = dict(task_key=env.task_name, evaluation_data=bool(env.cfg.is_eval),
                    producing_policy_iteration=producing_iteration(owner),
                    round_index=env._pilot_round_index, rank=owner._rank,
                    reset_env_indices=indices, reset_state_ids=env.reset_state_ids.clone(),
                    episode_serials=env._pilot_episode_serial.clone(),
                    initial_observation=cpu_tree(result[0]),
                    public_observation_sha256=tensor_hash(result[0]),
                    reset_rng=snapshot_rng(), sampler_generator=env._generator.get_state(),
                    sampler_cursor=env._current_seed_index)
                if env.cfg.is_eval and env_idx is None:
                    phase = owner._pilot_phase
                    owner._pilot_first_resets.setdefault(phase, []).append(reset)
                prefix = owner._pilot_phase if env.cfg.is_eval else 'source'
                save_tensor(owner._pilot_root / 'initial-states' /
                    f'{prefix}-rank-{owner._rank:02d}-{owner._pilot_reset_ordinal:04d}.pt', reset)
                owner._pilot_reset_ordinal += 1
                return result

            def chunk_step(env, actions):
                owner._pilot_deadline()
                requested = int(actions.shape[0] * actions.shape[1])
                slot = owner._pilot_journal.begin(requested, phase=owner._pilot_phase)
                current = env._pilot_current_observation
                if current is None:
                    raise RuntimeError('A motor chunk has no recorded initial/current public observation')
                scene_ids = env.reset_state_ids.clone()
                episode_ids = env._pilot_episode_serial.clone()
                times = env._pilot_primitive_time.clone()
                result = super().chunk_step(actions)
                mask = result[-1][-1].get('valid_action_mask')
                executed = int(mask.sum().item()) if mask is not None else requested
                owner._pilot_journal.finish(slot, executed)
                duration = (mask.detach().cpu().sum(dim=-1).long() if mask is not None else
                    torch.full((env.num_envs,), int(actions.shape[1]), dtype=torch.long))
                trace = dict(phase=owner._pilot_phase, journal_cursor=slot,
                    evaluation_data=bool(env.cfg.is_eval),
                    task_key=env.task_name, rank=owner._rank, pipeline_stage=0,
                    local_env_indices=list(range(env.num_envs)),
                    global_env_indices=list(range(owner._rank*env.num_envs, (owner._rank+1)*env.num_envs)),
                    producing_policy_iteration=(owner._pilot_eval_iteration if env.cfg.is_eval else producing_iteration(owner)),
                    collection_iteration=producing_iteration(owner)+1 if not env.cfg.is_eval else None,
                    round_index=env._pilot_round_index, chunk_index=env._pilot_chunk_index,
                    episode_serials=episode_ids, primitive_time_before=times,
                    executed_or_conservative_duration=duration, reset_state_ids=scene_ids,
                    current_observation=current,
                    proposed_motor_actions=cpu_tree(actions), outputs=cpu_tree(result),
                    measured_valid_mask=mask is not None,
                    executed_or_conservative=executed)
                path = owner._pilot_root / 'traces' / f'rank-{owner._rank:02d}' / f'{owner._pilot_trace_ordinal:05d}.pt.gz'
                path.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(path, 'xb', compresslevel=1) as handle:
                    torch.save(trace, handle)
                owner._pilot_trace_ordinal += 1
                env._pilot_chunk_index += 1
                env._pilot_current_observation = cpu_tree(result[0][-1])
                reset_during_chunk = env._pilot_episode_serial != episode_ids
                env._pilot_primitive_time = torch.where(reset_during_chunk,
                    torch.zeros_like(times), times + duration)
                return result

        module.RoboTwinEnv = RecordedRoboTwin
        return super().init_worker()

    def _pilot_deadline(self):
        if time.time() >= float(self.cfg.native_pilot.deadline_unix) - 90:
            raise TimeoutError('Native PPO phase deadline; keep last complete boundary and all charged prefixes')

    def pilot_phase(self, phase, *, policy_iteration=None):
        self._pilot_deadline()
        self._pilot_phase = phase
        self._pilot_eval_iteration = producing_iteration(self) if policy_iteration is None else int(policy_iteration)
        return dict(rank=self._rank, phase=phase, action_charge=self._pilot_journal.charged())

    def pilot_pair_receipt(self):
        before = self._pilot_first_resets.get('before', [])
        after = self._pilot_first_resets.get('after', [])
        if len(before) != len(after) or not before:
            raise ValueError('Missing paired before/after reset records')
        rows = []
        for left, right in zip(before, after):
            same_ids = torch.equal(left['reset_state_ids'], right['reset_state_ids'])
            if not same_ids:
                raise ValueError('Evaluation changed its predeclared reset-state IDs')
            rows.append(dict(reset_state_ids=left['reset_state_ids'].tolist(), same_ids=True,
                same_public_observation_bytes=left['public_observation_sha256'] == right['public_observation_sha256'],
                before_observation_sha256=left['public_observation_sha256'],
                after_observation_sha256=right['public_observation_sha256']))
        return dict(rank=self._rank, pairs=rows, action_charge=self._pilot_journal.charged())


class PilotRunner(EmbodiedRunner):
    def _pilot_write(self, name, value):
        write(Path(self.cfg.native_pilot.output) / name, value)

    def _capture_boundary(self):
        self._save_checkpoint()  # official actor/Adam/scheduler DCP, all ranks
        root = Path(self.cfg.runner.logger.log_path) / self.cfg.runner.logger.experiment_name / 'checkpoints' / f'global_step_{self.global_step}'
        rows = []
        for group in (self.actor, self.rollout, self.env):
            rows.extend(group.pilot_capture(str(root / 'pilot'), retain_rollout=True).wait())
        actor_files = [dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
            for path in sorted((root / 'actor').rglob('*')) if path.is_file()]
        write(root / 'pilot-boundary.json', dict(completed=True, global_step=self.global_step,
            workers=rows, actor_checkpoint_files=actor_files,
            exact_resume_scope='whole completed PPO iteration; fresh source reset',
            mid_iteration_physics_resume=False, source_admission=False))

    def _maybe_eval_and_checkpoint(self, step):
        # The upstream loop and learner remain untouched. Suppress upstream
        # terminal auto-evaluation; explicit before/after are the only queries.
        return {}

    def _log_step_metrics(self, *args, **kwargs):
        # Upstream logging waits env_handle, which commits the sampler cursor.
        # Capturing earlier (after actor/rollout handles only) could race it.
        result = super()._log_step_metrics(*args, **kwargs)
        self._capture_boundary()
        return result

    def run(self):
        from omegaconf import OmegaConf
        from .rlinf_native_ppo_data import collection_contract
        recipe = validate_recipe(self.cfg)
        self._pilot_write('collection-contract.json', collection_contract())
        if OmegaConf.is_config(self.cfg):
            self._pilot_write('resolved-native-config.json', OmegaConf.to_container(self.cfg, resolve=True))
        self._pilot_write('resolved-pilot-contract.json', dict(recipe,
            mode=self.cfg.native_pilot.mode, role='target-online/native-action-expert PPO',
            source6_pretraining_admission=False, paper_numeric_reproduction=False,
            scaler_disabled=True, initialization_revision=self.cfg.native_pilot.initializer_revision))
        resume = self.cfg.runner.get('resume_dir')
        if resume:
            receipts = []
            for group in (self.actor, self.rollout, self.env):
                receipts.extend(group.pilot_restore(str(Path(resume) / 'pilot')).wait())
            self._pilot_write('operational-restore.json', dict(global_step=self.global_step, workers=receipts))
        if self.cfg.native_pilot.mode == 'evaluate_existing':
            self.update_rollout_weights()
            self.rollout.pilot_evaluation_iteration(0).wait()
            self.env.pilot_phase('existing').wait()
            metrics = self.evaluate()
            self._pilot_write('existing-evaluation.json', dict(completed=True, metrics=metrics,
                optimizer_updates=0, model_path=self.cfg.actor.model.model_path))
            self._finish_run()
            return
        if not resume:
            self.update_rollout_weights()
            for group in (self.actor, self.rollout, self.env):
                group.pilot_rng().wait()
            self._capture_boundary()  # baseline actor/Adam/scheduler plus sampler/RNG
            for group in (self.actor, self.rollout, self.env):
                group.pilot_rng(restore=True).wait()
            self.env.pilot_phase('before').wait()
            self.rollout.pilot_evaluation_iteration(0).wait()
            before = self.evaluate()
            self._pilot_write('before-evaluation.json', dict(completed=True, global_step=0, metrics=before))
            # Isolate collection RNG from the extra baseline evaluation. The
            # same rollout sampling RNG is restored for the after panel below.
            for group in (self.actor, self.rollout, self.env):
                group.pilot_rng(restore=True).wait()
        self.env.pilot_phase('training').wait()
        super().run()
        # super closes the asynchronous logger; write our own immutable JSON.
        if self.global_step != 2:
            raise RuntimeError('Pilot ended without two complete published-batch iterations')
        self.rollout.pilot_rng(restore=True).wait()
        self.env.pilot_rng(restore=True).wait()
        self.update_rollout_weights()
        self.env.pilot_phase('after', policy_iteration=self.global_step).wait()
        self.rollout.pilot_evaluation_iteration(self.global_step).wait()
        after = self.evaluate()
        pairing = self.env.pilot_pair_receipt().wait()
        self._pilot_write('after-evaluation.json', dict(completed=True, global_step=2, metrics=after,
            pairing=pairing, model_selection='fixed iteration2', source6_pretraining_admission=False))
