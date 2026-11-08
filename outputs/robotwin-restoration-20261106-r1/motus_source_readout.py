"""Read-only full-checkpoint restore and paired native-B2 source evaluation.

No training loop is invoked. The native ODE sampler is observed in place;
its actual initial Gaussian is recorded without consuming an additional draw.
"""
import json
import os
from pathlib import Path
import random
import sys
from types import MethodType
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import ray
import torch

import motus_teacher_entry as teacher
from motus_optimized_entry import require_reserve
from case_schedule import CaseSchedule, install as install_cases
from optimizer_evidence import evidence
from recovery_signature import actor_signature, digest
from fluffy_bassoon.meta_rl import rlinf_native_ppo_runtime as runtime
from fluffy_bassoon.meta_rl.rlinf_native_ppo import cpu_tree, json_metrics, sha, snapshot_rng, tensor_hash, write
from fluffy_bassoon.meta_rl.tensor_archive import read_tensor
from rlinf.scheduler import Channel
from rlinf.utils.metric_utils import compute_evaluate_metrics
from motus_curve_readout import selected_steps, allowed_steps, original_sft_restore

ROOT = Path(__file__).resolve().parent


def selected_iterations(plan):
    return selected_steps(plan)


def selected_seeds(bank, panel, rank):
    assert len(bank) == 64 and len(set(bank)) == 64
    assert 0 <= panel < 4 and 0 <= rank < 8
    return bank[panel*16+rank*2:panel*16+rank*2+2]


def observe_native_ODE(original, action_dim, *args, **kwargs):
    assert kwargs.get('noise') is None, 'Native readout must draw its own noise'
    draws = []
    native_randn = torch.randn
    def record(*shape, **options):
        value = native_randn(*shape,**options)
        draws.append(value.detach().cpu().clone())
        return value
    with patch.object(torch,'randn',record):
        value = original(*args,**kwargs)
    assert len(draws) == 1 and list(draws[0].shape) == [2,32,action_dim]
    assert torch.isfinite(draws[0]).all()
    return value, draws[0]


class ReadoutActor(teacher.MotusActor):
    def readout_signature(self, label):
        assert all(p.grad is None for p in self.model.parameters())
        layout = evidence(self)
        write(Path(self.cfg.native_pilot.output)/'readout-actor'/f'{label}-rank-{self._rank:02d}.json',layout)
        return dict(rank=self._rank,**actor_signature(self,snapshot_rng(),optimizer_layout=layout))


class ReadoutRollout(teacher.MotusRollout):
    def init_worker(self):
        result = super().init_worker()
        original = self.hf_model.sample_actions
        owner = self
        def observed(instance, *args, **kwargs):
            # Patch is confined to this worker's synchronous native sampler.
            # The original generator, arguments, draw location and result are retained.
            value, noise = observe_native_ODE(original,instance.action_dim,*args,**kwargs)
            owner._readout_actual_noise = noise
            return value
        self.hf_model.sample_actions = MethodType(observed,self.hf_model)
        self._readout_noise = []
        return result

    def readout_begin(self, step, panel):
        self._flush_pilot_io()
        assert 0 <= panel < 4 and step in allowed_steps(self.cfg)
        seed = 31000+100*panel+self._rank
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        self._readout_label = f'policy-{step:04d}-panel-{panel:02d}'
        self._readout_noise = []
        self._readout_frozen = digest(dict(self.hf_model.named_parameters()))
        self._pilot_eval_iteration = step
        return dict(rank=self._rank,seed=seed,native_batch=2,initial_rng_sha256=digest(snapshot_rng()))

    def _predict_rollout_actions(self,env_obs,mode='train',**kwargs):
        require_reserve(self)
        assert mode == 'eval'
        first_panel = (self.cfg.native_pilot.readout_bank == 'dev'
            and self._readout_label == 'policy-0000-panel-00')
        if first_panel:
            index = len(self._readout_noise)
            reference = Path(self.cfg.native_pilot.readout_reference_root)/'sampling'/f'rank-{self._rank:02d}'/f'{index:05d}.pt.gz'
            source = read_tensor(reference)
            assert digest(snapshot_rng()) == digest(source['rng_before'])
            assert tensor_hash(cpu_tree(env_obs)) == source['public_observation_sha256']
        result = super()._predict_rollout_actions(env_obs,mode=mode,**kwargs)
        if first_panel:
            actual = torch.as_tensor(result[0]).detach().cpu().reshape(2,-1)
            expected = source['forward_sampling_fields']['action']
            assert actual.dtype == expected.dtype and torch.equal(actual,expected)
        assert hasattr(self,'_readout_actual_noise')
        noise = self._readout_actual_noise
        row = dict(prediction_ordinal=self._pilot_prediction_ordinal-1,
            initial_gaussian_sha256=digest(noise),shape=list(noise.shape),
            per_row_initial_gaussian_sha256=[digest(v) for v in noise],extra_random_draws=0)
        row['source_SFT_actions_pixels_RNG_exact'] = True if first_panel else None
        path = Path(self.cfg.native_pilot.output)/'actual-eval-noise'/(
            f'{self._readout_label}-rank-{self._rank:02d}-chunk-{len(self._readout_noise):02d}.pt.gz')
        row['raw_tensor'] = runtime.save_tensor(path,dict(initial_gaussian=noise,
            rank=self._rank,prediction_ordinal=row['prediction_ordinal'],panel=self._readout_label))
        self._readout_noise.append(row)
        return result

    def readout_finish(self):
        self._flush_pilot_io()
        assert len(self._readout_noise) == 6
        assert not any(p.grad is not None for p in self.hf_model.parameters())
        assert digest(dict(self.hf_model.named_parameters())) == self._readout_frozen
        row = dict(rank=self._rank,samples=self._readout_noise,optimizer_updates=0,
            frozen_parameters_unchanged=True,native_ODE_sampler_unchanged=True)
        write(Path(self.cfg.native_pilot.output)/'actual-eval-noise'/
            f'{self._readout_label}-rank-{self._rank:02d}.json',row)
        return row


class ReadoutEnvironment(teacher.MotusEnvironment):
    def pilot_restore(self,directory):
        state=json.loads((Path(directory)/f'motus-case-schedule-env-{self._rank:02d}.json').read_bytes())
        if original_sft_restore(self.cfg,self._motus_schedule,state,directory):
            # Retain the original native sampler/RNG restore. Only this explicit
            # C0 readout accepts the old manifest; training is never invoked.
            result=runtime.PilotEnvironment.pilot_restore(self,directory)
            self._paired_reset_ordinal=0
            write(Path(self.cfg.native_pilot.output)/'readout-case-migration'/f'env-{self._rank:02d}.json',dict(
                original_SFT_checkpoint0=True,read_only_DEV=True,next_reset_ordinal=0,
                original_manifest_sha256=state['manifest_sha256'],
                extended_manifest_sha256=self._motus_schedule.sha256,
                exact256_group_prefix_verified=True,native_sampler_and_RNG_restore_retained=True))
            return result
        return super().pilot_restore(directory)

    def init_worker(self):
        root = teacher.configure_environment(self)
        result = runtime.PilotEnvironment.init_worker(self)
        self._paired_reset_ordinal = 0
        self._motus_schedule = CaseSchedule(self.cfg.native_pilot.case_manifest_path,
            self.cfg.native_pilot.case_manifest_sha256)
        assert len(self.env_list) == len(self.eval_env_list) == 1
        train = runtime.base_env(self.env_list[0]);evaluation = runtime.base_env(self.eval_env_list[0])
        self._install_bindings(train,root);self._install_bindings(evaluation,root)
        install_cases(self,train,self._motus_schedule,root)
        assert evaluation.num_envs == 2
        original = evaluation.reset
        owner = self
        def reset(env,env_idx=None,env_seeds=None):
            indices = list(range(env.num_envs)) if env_idx is None else list(env_idx)
            seeds = [owner._readout_selected[int(i)] for i in indices]
            env.reset_state_ids[indices] = torch.tensor(seeds,dtype=torch.long,device=env.reset_state_ids.device)
            return original(env_idx=env_idx,env_seeds=seeds)
        evaluation.reset = MethodType(reset,evaluation)
        return result

    def readout_begin(self,bank,panel,step):
        require_reserve(self)
        self._readout_selected = selected_seeds(bank,panel,self._rank)
        self.pilot_phase(f'readout-policy-{step:04d}-panel-{panel:02d}',policy_iteration=step)
        return dict(rank=self._rank,seeds=self._readout_selected,native_batch=2)


def same_signature(a,b):
    return {k:v for k,v in a.items() if k!='optimizer_sha256'} == {
        k:v for k,v in b.items() if k!='optimizer_sha256'}


def configuration(plan,output):
    cfg = OmegaConf.create(json.loads(Path(plan['original_config']).read_bytes()))
    method = plan['algorithm']
    extended=plan.get('extended_source_curve_readout',False)
    if extended:
        selected_iterations(plan)
        assert cfg.native_pilot.extended_source_curve and method=='grpo'
    case_name=('MOTUS-CURVE-GRPO-CASES.json' if extended else
        'MOTUS-PPO-GROUP1-CASES.json' if method=='ppo' else 'MOTUS-TRAIN-CASES.json')
    for key,name in [('case',case_name),
            ('dev','MOTUS-DEV-CASES.json')]:
        cfg.native_pilot[f'{key}_manifest_path'] = str(ROOT/name)
        assert sha(ROOT/name) == cfg.native_pilot[f'{key}_manifest_sha256']
    teacher.validate(cfg)
    assert cfg.native_pilot.require_behavior_likelihood_match and cfg.actor.micro_batch_size == 32
    assert cfg.native_pilot.world_size in (8,'8') and cfg.native_pilot.comparison_algorithm == method
    if not extended:assert not cfg.runner.resume_dir
    cfg.runner.resume_dir=None  # This entry restores only the two explicit read-only inputs.
    cfg.runner.logger.log_path = str(Path(os.environ['TASK_LOCAL'])/'logs')
    cfg.runner.logger.experiment_name = plan['output_identity']
    cfg.runner.per_worker_log_path = str(output/'workers')
    cfg.native_pilot.output = str(output)
    cfg.native_pilot.accounting_root = str(output/'action-accounting')
    cfg.native_pilot.attempt = plan['output_identity']
    cfg.native_pilot.deadline_unix = int(os.environ['TASK_DEADLINE'])
    cfg.native_pilot.maximum_actions = 32768
    cfg.native_pilot.passive_controller_source = os.environ['TASK_CONTROLLER']
    cfg.native_pilot.passive_controller_sha256 = os.environ['TASK_CONTROLLER_SHA']
    cfg.native_pilot.readout_reference_root = (plan['source_SFT_reference_state'] if extended else plan['source_state'])
    cfg.native_pilot.readout_bank = plan['bank']
    if extended:
        cfg.native_pilot.extended_source_curve_readout=True
        cfg.native_pilot.readout_selected_iteration=plan['selected_training_iteration']
    assert plan['bank'] in ('dev','confirmation')
    assert cfg.env.eval.total_num_envs == 16 and cfg.env.eval.rollout_epoch == 1
    return cfg


def main():
    plan = json.loads(Path(os.environ['TASK_READOUT_PLAN']).read_bytes())
    assert plan['status'] == 'admitted' and os.uname().nodename.startswith('job-')
    torch.multiprocessing.set_start_method('spawn',force=True)
    output = Path(os.environ['TASK_COLD'])/'state';output.mkdir()
    cfg = configuration(plan,output)
    # Full native schema was qualified before the explicit read-only action cap.
    cfg = teacher.validate_cfg(cfg)
    cases = json.loads((ROOT/'MOTUS-SOURCE-READOUT-CASES.json').read_bytes())
    bank = cases[plan['bank']]
    assert plan['bank'] in ('dev','confirmation') and len(bank) == 64
    if plan['bank'] == 'confirmation':assert plan['selection_committed_before_confirmation']
    cluster = teacher.Cluster(cluster_cfg=cfg.cluster,distributed_log_dir=cfg.runner.per_worker_log_path)
    placement = teacher.HybridComponentPlacement(cfg,cluster)
    groups = []
    report = dict(status='starting',algorithm=plan['algorithm'],bank=plan['bank'],panels=[],restores=[],
        optimizer_updates=0,source_commit=os.environ['TASK_SOURCE'],gain_claim=False)
    if plan.get('extended_source_curve_readout',False):
        report.update(extended_source_curve_readout=True,input_checkpoints=plan['checkpoints'],
            original_SFT_reference_state=plan['source_SFT_reference_state'],
            fixed_DEV_only=True,new_confirmation=False)
    try:
        for role,worker in [('actor',ReadoutActor),('rollout',ReadoutRollout),('env',ReadoutEnvironment)]:
            groups.append(worker.create_group(cfg).launch(cluster,name=cfg[role].group_name,
                placement_strategy=placement.get_strategy(role)))
        actor,rollout,env = groups
        runner = teacher.MotusRunner(cfg=cfg,actor=actor,rollout=rollout,env=env,reward=None)
        runner.init_workers()
        for step in selected_iterations(plan):
            checkpoint = Path(os.environ['TASK_LOCAL'])/'checkpoints'/f'global_step_{step}'
            assert sha(checkpoint/'MANIFEST.json') == plan['checkpoints'][str(step)]['manifest_sha256']
            actor.load_checkpoint(str(checkpoint/'actor')).wait()
            restored = []
            for group in groups:restored.extend(group.pilot_restore(str(checkpoint/'pilot')).wait())
            assert len(restored) == 24 and all(r['restored'] for r in restored)
            for group in groups:group.set_global_step(step).wait()
            runner.global_step = step
            before = sorted(actor.readout_signature(f'policy-{step}-before').wait(),key=lambda r:r['rank'])
            assert len(before) == 8 and all(r['optimizer_steps'] == 15*step for r in before)
            actor.pilot_rng().wait()
            runner.update_rollout_weights()
            report['restores'].append(dict(step=step,workers=restored,actor_before=before))
            for panel in range(4):
                environment = env.readout_begin(bank,panel,step).wait()
                sampling = rollout.readout_begin(step,panel).wait()
                env_channel = Channel.create(f'EnvP{step}R{panel}')
                rollout_channel = Channel.create(f'RolloutP{step}R{panel}')
                handle = env.evaluate(input_channel=env_channel,rollout_channel=rollout_channel)
                predictions = rollout.evaluate(input_channel=rollout_channel,output_channel=env_channel)
                results = handle.wait();predictions.wait()
                metrics = compute_evaluate_metrics([r for r in results if r is not None])
                assert int(metrics['num_trajectories']) == 16
                frozen = rollout.readout_finish().wait()
                for group in (rollout,env):group.set_global_step(step).wait()
                row = dict(step=step,panel=panel,metrics=json_metrics(metrics),
                    environments=environment,sampling=sampling,frozen_rollouts=frozen)
                report['panels'].append(row)
                write(output/f'panel-policy-{step:04d}-{panel:02d}.json',row)
            actor.pilot_rng(restore=True).wait()
            after = sorted(actor.readout_signature(f'policy-{step}-after').wait(),key=lambda r:r['rank'])
            assert all(same_signature(a,b) for a,b in zip(before,after,strict=True))
            report['restores'][-1]['actor_after'] = after
        report.update(status='complete_readonly_full_restore_paired_source_readout',episodes=128,
            checkpoints=list(selected_iterations(plan)),full24worker_restores_each_policy=True,actor_Adam_scheduler_scaler_RNG_unchanged=True,
            independent_trace_noise_and_pair_readback_required=True,source_effectiveness_not_yet_claimed=True)
    finally:
        write(output/'RESULT.json',report)
        failed = sys.exc_info()[0] is not None;errors = []
        for group in groups:
            if ray.is_initialized():
                try:group._close()
                except Exception as error:errors.append(str(error))
        ray.shutdown()
        if errors:
            write(Path(os.environ['TASK_OUT'])/'cleanup-errors.json',dict(errors=errors,primary_failure_preserved=failed))
            if not failed:raise RuntimeError('Owned readout cleanup failed')


if __name__=='__main__':main()
