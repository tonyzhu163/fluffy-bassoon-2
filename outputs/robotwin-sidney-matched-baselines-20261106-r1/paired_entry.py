"""Run-scoped common-case PPO/GRPO wrapper around the pinned native learner."""
import importlib.util
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

SHARED=Path(__file__).resolve().parents[1]/'robotwin-sidney-rlinf-ppo-20261105-r1'
sys.path.insert(0,str(SHARED))
import train_entry as shared
from case_schedule import CaseSchedule,boundary_ordinal,install
from paired_recipe import validate_pair
from recovery_signature import actor_signature
from optimizer_evidence import evidence
from optimizer_semantics import semantic_digest
from gradient_audit import audit
from fluffy_bassoon.meta_rl.rlinf_native_ppo import snapshot_rng,restore_rng,tensor_hash,write

runtime=shared.runtime


def initialize_rng(worker,role):
    # These independent streams are initialized after optional critic creation.
    # A complete recovery subsequently restores captured states, without reseeding.
    seed={'actor':22000,'rollout':23000}[role]+worker._rank
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    write(Path(worker.cfg.native_pilot.output)/'paired-rng'/f'{role}-{worker._rank:02d}.json',dict(
        role=role,rank=worker._rank,seed=seed,initialized_after_model=True,
        state_sha256=tensor_hash(snapshot_rng()),complete_physics_rng_capture=False))


class PairedActor(shared.SidneyActor):
    def init_worker(self):
        result=super().init_worker();initialize_rng(self,'actor')
        self._pair_first_replay=False
        self._pair_audit_micro=0
        return result

    def pilot_capture(self,directory,*,retain_rollout=False):
        result=super().pilot_capture(directory,retain_rollout=retain_rollout)
        layout=evidence(self)
        write(Path(directory)/f'paired-optimizer-layout-{self._rank:02d}.json',layout)
        write(Path(directory)/f'paired-actor-signature-{self._rank:02d}.json',
            actor_signature(self,snapshot_rng(),optimizer_layout=layout))
        return result

    def pilot_restore(self,directory):
        import json
        result=super().pilot_restore(directory)
        frozen=self._pilot_frozen_signature()
        write(Path(self.cfg.native_pilot.output)/'recovery-frozen'/f'actor-{self._rank:02d}.json',dict(
            expected=self._pilot_frozen_before,actual=frozen,unchanged=frozen==self._pilot_frozen_before))
        assert frozen==self._pilot_frozen_before,'Frozen VLM differs after native checkpoint readback'
        expected=json.loads((Path(directory)/f'paired-actor-signature-{self._rank:02d}.json').read_text())
        saved_layout=json.loads((Path(directory)/f'paired-optimizer-layout-{self._rank:02d}.json').read_text())
        assert expected['optimizer_semantic_version']==1
        assert saved_layout['optimizer_sha256']==expected['optimizer_sha256']
        assert semantic_digest(saved_layout)==expected['optimizer_semantic_sha256']
        layout=evidence(self)
        write(Path(self.cfg.native_pilot.output)/'recovery-optimizer-layout'/f'actor-{self._rank:02d}.json',layout)
        actual=actor_signature(self,snapshot_rng(),optimizer_layout=layout)
        semantic_identity={k:v for k,v in expected.items() if k!='optimizer_sha256'}=={
            k:v for k,v in actual.items() if k!='optimizer_sha256'}
        write(Path(self.cfg.native_pilot.output)/'recovery-signatures'/f'actor-{self._rank:02d}.json',dict(
            expected=expected,actual=actual,exact_raw_identity=expected==actual,
            semantic_identity=semantic_identity,empty_unowned_state_rule='opt01-measured-v1'))
        assert semantic_identity,'Native model/owned Adam/groups/scheduler/scaler/RNG restoration differs'
        return result

    def train_micro_batch(self,micro_batch,metrics,*,is_last):
        critic=bool(self.cfg.actor.model.add_value_head)
        first=not self._pair_first_replay
        if first:
            from rlinf.utils.nested_dict_process import put_tensor_device
            batch=put_tensor_device(micro_batch,self.device);rng=snapshot_rng()
            try:
                with torch.no_grad(),self.amp_context:
                    replay=self.model(forward_inputs=batch['forward_inputs'],compute_logprobs=True,
                        compute_entropy=False,compute_values=critic,use_cache=False)
                old,new=batch['prev_logprobs'].float(),replay['logprobs'].float()
                assert old.shape==new.shape and torch.isfinite(old).all() and torch.isfinite(new).all()
                error=float((new-old).abs().max());assert error<=1e-3
                if critic:
                    assert torch.isfinite(batch['prev_values']).all() and torch.isfinite(replay['values']).all()
                else:
                    assert not any('value_head' in name for name,_ in self.model.named_parameters())
                write(Path(self.cfg.native_pilot.output)/'first-replay'/f'rank-{self._rank:02d}.json',dict(
                    passed=True,logprob_max_abs=error,critic_present=critic,optimizer_steps=self.optimizer_steps))
            finally:restore_rng(rng)
            self._pair_first_replay=True
        # Skip only SidneyActor's unconditional critic replay guard. The shared
        # PilotActor and original RLinf loss/backward/optimizer remain in use.
        result=runtime.PilotActor.train_micro_batch(self,micro_batch,metrics,is_last=is_last)
        if self.optimizer_steps==0 and self._pair_audit_micro<8:
            audit(self,is_last=is_last,ordinal=self._pair_audit_micro)
            self._pair_audit_micro+=1
        if first:
            names=[]
            for name,param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    assert torch.isfinite(param.grad).all(),name
                    names.append(name)
            assert names and any('action_' in name or '.1.' in name for name in names)
            assert any('value_head' in name for name in names)==critic
            write(Path(self.cfg.native_pilot.output)/'paired-gradients'/f'rank-{self._rank:02d}.json',dict(
                finite=True,frozen_parameters_without_nonzero_gradients=True,
                frozen_parameters_excluded_from_optimizer=True,critic_present=critic,parameters=names))
        return result


class PairedRollout(shared.SidneyRollout):
    def init_worker(self):
        result=super().init_worker();initialize_rng(self,'rollout');return result


class PairedEnvironment(shared.SidneyEnvironment):
    def init_worker(self):
        result=super().init_worker();self._paired_reset_ordinal=0
        assert len(self.env_list)==1
        self._paired_schedule=CaseSchedule(self.cfg.native_pilot.case_manifest_path,
            self.cfg.native_pilot.case_manifest_sha256)
        install(self,runtime.base_env(self.env_list[0]),self._paired_schedule,
            Path(self.cfg.native_pilot.output))
        return result

    def pilot_capture(self,directory,*,retain_rollout=False):
        assert self._paired_reset_ordinal==boundary_ordinal(directory,int(self.cfg.env.train.rollout_epoch))
        receipt=super().pilot_capture(directory,retain_rollout=retain_rollout)
        write(Path(directory)/f'case-schedule-env-{self._rank:02d}.json',dict(
            manifest_sha256=self._paired_schedule.sha256,next_reset_ordinal=self._paired_reset_ordinal))
        return receipt

    def pilot_restore(self,directory):
        import json
        saved=json.loads((Path(directory)/f'case-schedule-env-{self._rank:02d}.json').read_text())
        assert saved['manifest_sha256']==self._paired_schedule.sha256
        assert saved['next_reset_ordinal']==boundary_ordinal(directory,int(self.cfg.env.train.rollout_epoch))
        receipt=super().pilot_restore(directory)
        self._paired_reset_ordinal=saved['next_reset_ordinal']
        return receipt


class PairedRunner(shared.SidneyRunner):
    def _capture_boundary(self):
        import time
        started=time.perf_counter();result=super()._capture_boundary()
        self._pilot_write(f'boundary-timings/step-{self.global_step:04d}.json',dict(
            global_step=self.global_step,seconds=time.perf_counter()-started,
            includes_native_DCP_worker_sidecars_and_SHA256=True))
        return result

    def run(self):
        if self.cfg.native_pilot.qualification_only:
            return self.qualify()
        result=super().run()
        self._pilot_write('RESULT.json',dict(status='complete',rollout_iterations=2,
            algorithm=str(self.cfg.native_pilot.comparison_algorithm),
            case_manifest_sha256=self.cfg.native_pilot.case_manifest_sha256,
            baseline_reference='published Sidney SFT',new_sft_evaluation_episodes=0,
            source_task='place_phone_stand',group_size=8,
            original_initializer_reproduction=False,paper_numeric_reproduction=False))
        return result

    def qualify(self):
        # A separately charged one-round/one-update engineering fixture. It is
        # never admitted as an arm of the scientific two-iteration comparison.
        from omegaconf import OmegaConf
        from rlinf.runners.embodied_runner import EmbodiedRunner
        validate_pair(self.cfg)
        self._pilot_write('resolved-native-config.json',OmegaConf.to_container(self.cfg,resolve=True))
        resume=self.cfg.runner.get('resume_dir')
        if resume:
            rows=[]
            for group in (self.actor,self.rollout,self.env):
                rows.extend(group.pilot_restore(str(Path(resume)/'pilot')).wait())
            self._pilot_write('operational-restore.json',dict(global_step=self.global_step,workers=rows))
        else:
            self.update_rollout_weights();self._capture_boundary()
        if not resume:
            self.env.pilot_phase('training').wait()
            EmbodiedRunner.run(self)
        assert self.global_step==1
        self._pilot_write('RESULT.json',dict(status='complete',qualification_only=True,
            algorithm='grpo',rollout_iterations=1,optimizer_updates=1,
            recovery_only=bool(resume),scientific_source_admission=False))


def main():
    shared.validate_sidney=validate_pair;runtime.validate_recipe=validate_pair
    import rlinf.workers.actor.embodied_fsdp_actor_worker as actor_module
    actor_module.EmbodiedFSDPActor=PairedActor
    path=Path(os.environ['EMBODIED_PATH'])/'train_embodied_agent.py'
    spec=importlib.util.spec_from_file_location('sidney_paired_entry',path)
    entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    entry.EmbodiedRunner=PairedRunner;entry.EnvWorker=PairedEnvironment
    entry.MultiStepRolloutWorker=PairedRollout
    entry.main()


if __name__=='__main__':main()
