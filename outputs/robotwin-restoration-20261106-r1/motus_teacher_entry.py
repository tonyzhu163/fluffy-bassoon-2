"""Native Motus PPO/GRPO with common cases and shared train/eval engineering.

No Sidney transforms or numerical patches. Flow-SDE, GAE, losses and reduction
stay in pinned RLinf; the explicit OpenPi probability-reference hook preserves
its actor partitioning while handling continuous action coordinates.
"""
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
from types import MethodType

import hydra
import numpy as np
from omegaconf import OmegaConf
import ray
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'robotwin-sidney-matched-baselines-20261106-r1'))
from case_schedule import CaseSchedule, boundary_ordinal, install as install_cases
from gradient_audit import audit as audit_gradients
from optimizer_evidence import evidence
from recovery_signature import actor_signature
from optimizer_semantics import semantic_digest
from motus_configuration import validate
from motus_learner_reference import audit as audit_learner
from profile_checkpoint import verify as verify_archive
from fluffy_bassoon.meta_rl import rlinf_native_ppo_runtime as runtime
from fluffy_bassoon.rl.robotwin_native_setup import worker_setup, configure_environment
from fluffy_bassoon.meta_rl.rlinf_native_ppo import json_metrics, restore_rng, sha, snapshot_rng, write
from fluffy_bassoon.meta_rl.rlinf_robocasa_flow_ppo import keep_frozen_image_encoder_in_eval
from rlinf.config import validate_cfg
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement


def initialize_rng(worker, role):
    seed = {'actor': 22000, 'rollout': 23000}[role]+worker._rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    write(Path(worker.cfg.native_pilot.output)/'policy-rng'/f'{role}-{worker._rank:02d}.json',
        dict(seed=seed, initialized_after_model=True,
             protocol='native rank/batch Flow-SDE streams; selected denoising index shared within a batch',
             ode_case_noise_not_used=True))


class MotusActor(runtime.PilotActor):
    def model_provider_func(self):
        model = super().model_provider_func()
        assert model.action_horizon == model.action_chunk == 32 and model.action_env_dim == 14
        assert model.rl_cfg.train_expert_only and model.rl_cfg.noise_method == 'flow_sde'
        assert model.rl_cfg.noise_level == .3 and not any(p.requires_grad for p in model.img.parameters())
        native_train = model.train
        def train(instance, mode=True):
            result = native_train(mode)
            keep_frozen_image_encoder_in_eval(instance)
            return result
        model.train = MethodType(train, model)
        model.eval()
        return model

    def init_worker(self):
        worker_setup(self, 'actor')
        result = super().init_worker()
        initialize_rng(self, 'actor')
        self._motus_first_replay = False
        self._motus_gradient_ordinal = 0
        return result

    def compute_advantages_and_returns(self):
        result = super().compute_advantages_and_returns()
        report = audit_learner(self.rollout_batch, self.cfg.native_pilot.comparison_algorithm)
        write(Path(self.cfg.native_pilot.output)/'learner-reference'/
            f'iteration-{runtime.producing_iteration(self)+1:04d}-rank-{self._rank:02d}.json', report)
        assert report['passed'], 'Actual saved learner batch differs from independent reference'
        return result

    def run_training(self):
        torch.cuda.reset_peak_memory_stats(self.device)
        started = time.monotonic()
        assert not self.cfg.algorithm.recompute_logprobs and self.cfg.native_pilot.recompute_openpi_logprobs
        # c605 validation correctly restricts its flattened-token diagnostic to
        # OpenVLA. Opt in only during this owned actor loop, whose hook below
        # preserves the native split/order and handles continuous OpenPi shapes.
        self.cfg.algorithm.recompute_logprobs = True
        try:
            result = super().run_training()
        finally:
            self.cfg.algorithm.recompute_logprobs = False
        free,total = torch.cuda.mem_get_info(self.device)
        write(Path(self.cfg.native_pilot.output)/'actor-memory'/
            f'iteration-{runtime.producing_iteration(self)+1:04d}-rank-{self._rank:02d}.json',
            dict(seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device),free_bytes_at_end=free,
                total_bytes=total,micro_batch=256,accumulation=2,global_batch=2048,
                includes_recompute_and_five_native_epochs=True))
        return result

    def recompute_prev_logprobs(self,batch_size_per_rank):
        from fluffy_bassoon.meta_rl.rlinf_openpi_replay import recompute_openpi_logprobs
        return recompute_openpi_logprobs(self,batch_size_per_rank)

    def train_micro_batch(self, micro_batch, metrics, *, is_last):
        critic = bool(self.cfg.actor.model.add_value_head)
        if not self._motus_first_replay:
            from rlinf.utils.nested_dict_process import put_tensor_device
            batch = put_tensor_device(micro_batch, self.device)
            rng = snapshot_rng()
            try:
                with torch.no_grad(), self.amp_context:
                    replay = self.model(forward_inputs=batch['forward_inputs'],compute_logprobs=True,
                        compute_entropy=False,compute_values=critic,use_cache=False)
                before, after = batch['prev_logprobs'].float(), replay['logprobs'].float()
                assert before.shape == after.shape and torch.isfinite(before).all() and torch.isfinite(after).all()
                error = float((before-after).abs().max())
                assert error <= 1e-3, 'Native recomputed reference differs before first SGD'
                if critic:
                    assert torch.isfinite(batch['prev_values']).all() and torch.isfinite(replay['values']).all()
                else:
                    assert not any('value_head' in name for name,_ in self.model.named_parameters())
                write(Path(self.cfg.native_pilot.output)/'first-replay'/f'rank-{self._rank:02d}.json',
                    dict(passed=True,logprob_max_abs=error,micro_batch=256,optimizer_steps=self.optimizer_steps,
                         reference='native once-per-iteration recomputation before all5 epochs',critic_present=critic))
            finally:
                restore_rng(rng)
            self._motus_first_replay = True
        result = super().train_micro_batch(micro_batch,metrics,is_last=is_last)
        # Qualify both an unsynchronized accumulation and its synchronization.
        if self.optimizer_steps == 0:
            audit_gradients(self,is_last=is_last,ordinal=self._motus_gradient_ordinal)
            self._motus_gradient_ordinal += 1
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad and parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(), name
        return result

    def pilot_capture(self,directory,*,retain_rollout=False):
        result = super().pilot_capture(directory,retain_rollout=retain_rollout)
        layout = evidence(self)
        write(Path(directory)/f'motus-optimizer-layout-{self._rank:02d}.json',layout)
        write(Path(directory)/f'motus-actor-signature-{self._rank:02d}.json',
              actor_signature(self,snapshot_rng(),optimizer_layout=layout))
        return result

    def pilot_restore(self,directory):
        result = super().pilot_restore(directory)
        expected = json.loads((Path(directory)/f'motus-actor-signature-{self._rank:02d}.json').read_bytes())
        saved = json.loads((Path(directory)/f'motus-optimizer-layout-{self._rank:02d}.json').read_bytes())
        assert saved['optimizer_sha256'] == expected['optimizer_sha256']
        assert semantic_digest(saved) == expected['optimizer_semantic_sha256']
        actual = actor_signature(self,snapshot_rng(),optimizer_layout=evidence(self))
        same = {k:v for k,v in actual.items() if k != 'optimizer_sha256'} == {
            k:v for k,v in expected.items() if k != 'optimizer_sha256'}
        write(Path(self.cfg.native_pilot.output)/'recovery-signatures'/f'actor-{self._rank:02d}.json',
              dict(expected=expected,actual=actual,owned_state_exact=same,
                   empty_unowned_state_rule='opt01-measured-v1'))
        assert same and self._pilot_frozen_signature() == self._pilot_frozen_before
        return result


class MotusRollout(runtime.PilotRollout):
    def init_worker(self):
        worker_setup(self,'rollout')
        result = super().init_worker()
        initialize_rng(self,'rollout')
        self._motus_value_audited = False
        return result

    def _predict_rollout_actions(self,env_obs,mode='train',**kwargs):
        check = mode == 'train' and bool(self.cfg.actor.model.add_value_head) and not self._motus_value_audited
        values = []
        hook = (self.hf_model.value_head.register_forward_hook(
            lambda module,inputs,output: values.append(output.detach().clone())) if check else None)
        try:
            actions, result = super()._predict_rollout_actions(env_obs,mode=mode,**kwargs)
        finally:
            if hook is not None: hook.remove()
        if check:
            rng = snapshot_rng()
            try:
                assert len(values) == 5
                raw = torch.stack(values,dim=1).float().squeeze(-1)
                batch = len(raw)
                picked = raw[torch.arange(batch,device=raw.device),result['forward_inputs']['denoise_inds'][:,0]]
                with torch.no_grad():
                    replay = self.hf_model(forward_inputs=result['forward_inputs'],compute_logprobs=True,
                        compute_entropy=False,compute_values=True,use_cache=False)
                mean = result['prev_values'].float().reshape(batch)
                selected = replay['values'].float().reshape(batch)
                errors = dict(logprobs=float((replay['logprobs'].float()-result['prev_logprobs'].float()).abs().max()),
                    rollout_mean_values=float((mean-raw.mean(dim=1)).abs().max()),
                    selected_step_values=float((selected-picked).abs().max()))
                passed = errors['logprobs'] <= 1e-3 and max(errors[k] for k in (
                    'rollout_mean_values','selected_step_values')) <= 1e-4
                write(Path(self.cfg.native_pilot.output)/'native-values'/f'rank-{self._rank:02d}.json',
                    dict(passed=passed,batch=batch,max_abs_error=errors,
                         mean_vs_selected_max_abs=float((mean-selected).abs().max()),
                         estimands_preserved=True,extra_audit_rng_restored=True))
                assert passed, 'Native Motus mean/selected critic or likelihood audit failed'
                self._motus_value_audited = True
            finally:
                restore_rng(rng)
        return actions, result


class MotusEnvironment(runtime.PilotEnvironment):
    def init_worker(self):
        root = configure_environment(self)
        result = super().init_worker()
        self._paired_reset_ordinal = 0
        self._motus_schedule = CaseSchedule(self.cfg.native_pilot.case_manifest_path,
                                           self.cfg.native_pilot.case_manifest_sha256)
        assert len(self.env_list) == len(self.eval_env_list) == 1
        train, evaluation = (runtime.base_env(group[0]) for group in (self.env_list,self.eval_env_list))
        self._install_bindings(train,root)
        self._install_bindings(evaluation,root)
        install_cases(self,train,self._motus_schedule,root)
        dev = json.loads(Path(self.cfg.native_pilot.dev_manifest_path).read_bytes())['seeds']
        seeds = dev[self._rank*4:(self._rank+1)*4]
        assert len(seeds) == evaluation.num_envs == 4
        original = evaluation.reset
        def reset(env,env_idx=None,env_seeds=None):
            indices = list(range(env.num_envs)) if env_idx is None else list(env_idx)
            selected = [seeds[int(index)] for index in indices]
            env.reset_state_ids[indices] = torch.tensor(selected,dtype=torch.long,device=env.reset_state_ids.device)
            return original(env_idx=env_idx,env_seeds=selected)
        evaluation.reset = MethodType(reset,evaluation)
        return result

    def _install_bindings(self,physical,root):
        original = physical.reset
        owner = self
        def reset(env,env_idx=None,env_seeds=None):
            result = original(env_idx=env_idx,env_seeds=env_seeds)
            indices = range(env.num_envs) if env_idx is None else env_idx
            raw = result[0]
            rows = []
            for index in indices:
                sub = env.venv.envs[index]
                assert int(sub.env_seed) == int(sub.task.ep_num)
                assert sub.episode_info_list == [sub.task.get_info()]
                assert sub.task.get_instruction() == raw['task_descriptions'][index]
                rows.append(dict(env_index=int(index),actual_seed=int(sub.task.ep_num),
                    scene_info=sub.task.get_info(),instruction=sub.task.get_instruction()))
            for key in ('main_images','wrist_images'):
                images = torch.as_tensor(raw[key])
                assert images.dtype == torch.uint8 and images.shape[-1] == 3
                spread = images.reshape(-1,*images.shape[-3:])[:,::16,::16].float().flatten(1).std(dim=1)
                assert bool((spread > 1).all()), f'Blank native camera: {key}'
            assert torch.isfinite(torch.as_tensor(raw['states'])).all()
            write(root/'scene-bindings'/f'rank-{owner._rank:02d}-reset-{owner._pilot_reset_ordinal:05d}.json',
                  dict(evaluation_data=bool(env.cfg.is_eval),phase=owner._pilot_phase,bindings=rows,
                       three_camera_nonblank=True,controller_passively_measured=True))
            return result
        physical.reset = MethodType(reset,physical)

    def pilot_capture(self,directory,*,retain_rollout=False):
        assert self._paired_reset_ordinal == boundary_ordinal(directory,4)
        result = super().pilot_capture(directory,retain_rollout=retain_rollout)
        write(Path(directory)/f'motus-case-schedule-env-{self._rank:02d}.json',dict(
            manifest_sha256=self._motus_schedule.sha256,next_reset_ordinal=self._paired_reset_ordinal))
        return result

    def pilot_restore(self,directory):
        state = json.loads((Path(directory)/f'motus-case-schedule-env-{self._rank:02d}.json').read_bytes())
        assert state['manifest_sha256'] == self._motus_schedule.sha256
        assert state['next_reset_ordinal'] == boundary_ordinal(directory,4)
        result = super().pilot_restore(directory)
        self._paired_reset_ordinal = state['next_reset_ordinal']
        return result


class MotusRunner(runtime.PilotRunner):
    def _capture_boundary(self):
        started = time.monotonic()
        super()._capture_boundary()
        checkpoint = Path(self.cfg.runner.logger.log_path)/self.cfg.runner.logger.experiment_name/'checkpoints'/f'global_step_{self.global_step}'
        manifest = dict(global_step=self.global_step,source_commit=os.environ['TASK_SOURCE'],
            files={str(path.relative_to(checkpoint)):dict(bytes=path.stat().st_size,sha256=sha(path))
                   for path in sorted(checkpoint.rglob('*')) if path.is_file()})
        write(checkpoint/'MANIFEST.json',manifest)
        cold = Path(self.cfg.native_pilot.checkpoint_archive)/checkpoint.name
        assert not cold.exists()
        shutil.copytree(checkpoint,cold)
        verify_archive(cold,manifest)
        stageback = Path(os.environ['TASK_LOCAL'])/f'stageback-{self.global_step}'
        shutil.copytree(cold,stageback)
        verify_archive(stageback,manifest)
        assert sha(stageback/'MANIFEST.json') == sha(cold/'MANIFEST.json') == sha(checkpoint/'MANIFEST.json')
        shutil.rmtree(stageback)  # Two complete verified retained copies remain.
        self._pilot_write(f'checkpoint-receipts/step-{self.global_step:04d}.json',dict(
            completed=True,cold=str(cold),manifest_sha256=sha(cold/'MANIFEST.json'),
            full_cold_readback=True,full_local_stageback=True,global_step=self.global_step,
            data_bytes=sum(row['bytes'] for row in manifest['files'].values()),
            seconds=time.monotonic()-started,independent_process_restore_still_required=True))

    def _dev(self,iteration):
        for group in (self.actor,self.rollout,self.env): group.pilot_rng().wait()
        self.update_rollout_weights()
        self.env.pilot_phase(f'dev-{iteration:04d}',policy_iteration=iteration).wait()
        self.rollout.pilot_evaluation_iteration(iteration).wait()
        metrics = self.evaluate()
        assert int(metrics['num_trajectories']) == 16, 'Incomplete native DEV panel'
        # Drain canonical train/eval writers without creating another checkpoint.
        for group in (self.rollout,self.env): group.set_global_step(iteration).wait()
        self._pilot_write(f'dev/iteration-{iteration:04d}.json',dict(completed=True,
            producing_policy_iteration=iteration,metrics=json_metrics(metrics),episodes=16,optimizer_updates_during_eval=0,
            checkpoint_selection='fixed iteration2; this is a source DEV progress panel'))
        for group in (self.actor,self.rollout,self.env): group.pilot_rng(restore=True).wait()

    def _log_step_metrics(self,*args,**kwargs):
        # PilotRunner waits native environment completion before saving samplers.
        result = super()._log_step_metrics(*args,**kwargs)
        self._dev(self.global_step)
        self.env.pilot_phase('training').wait()
        return result

    def run(self):
        contract = validate(self.cfg)
        self._pilot_write('resolved-native-config.json',OmegaConf.to_container(self.cfg,resolve=True))
        self._pilot_write('resolved-pilot-contract.json',contract)
        resume = self.cfg.runner.get('resume_dir')
        if resume:
            rows = []
            for group in (self.actor,self.rollout,self.env):
                rows.extend(group.pilot_restore(str(Path(resume)/'pilot')).wait())
            self._pilot_write('operational-restore.json',dict(global_step=self.global_step,workers=rows))
        else:
            self.update_rollout_weights()
            self._capture_boundary()
            self._dev(0)
        self.env.pilot_phase('training').wait()
        # This boundary stop preserves the two-iteration scientific contract;
        # a later job resumes the same full global batches and constant schedule.
        self.max_steps = int(self.cfg.native_pilot.stop_after_iteration)
        EmbodiedRunner.run(self)
        assert self.global_step == self.max_steps
        self._pilot_write('RESULT.json',dict(status='completed_admitted_prefix',
            completed_iterations=self.global_step,intended_iterations=2,
            native_adam_updates_per_rank=15*self.global_step,algorithm=self.cfg.native_pilot.comparison_algorithm,
            scientific_source_admission=False,complete_six_task_campaign=False))


@hydra.main(version_base='1.1',config_path=str(ROOT.parents[1]/'configs/native_ppo'),config_name='motus_phone_two')
def main(cfg):
    torch.multiprocessing.set_start_method('spawn',force=True)
    validate(cfg)
    cfg = validate_cfg(cfg)
    cluster = Cluster(cluster_cfg=cfg.cluster,distributed_log_dir=cfg.runner.per_worker_log_path)
    placement = HybridComponentPlacement(cfg,cluster)
    groups = []
    try:
        for role,worker in (('actor',MotusActor),('rollout',MotusRollout),('env',MotusEnvironment)):
            group = worker.create_group(cfg).launch(cluster,name=cfg[role].group_name,
                placement_strategy=placement.get_strategy(role))
            groups.append(group)
        runner = MotusRunner(cfg=cfg,actor=groups[0],rollout=groups[1],env=groups[2],reward=None)
        runner.init_workers()
        runner.run()
    finally:
        for group in groups: group._close()
        ray.shutdown()


if __name__ == '__main__': main()
