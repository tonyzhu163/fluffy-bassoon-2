"""Actor-only replay of an immutable closed native GRPO fixture; no simulator."""
import json
import os
from pathlib import Path
import time

import hydra
import ray
import torch
from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement

from paired_entry import PairedActor
from paired_recipe import validate_pair
from optimizer_evidence import evidence
from recovery_signature import actor_signature
from fluffy_bassoon.meta_rl.rlinf_native_ppo import snapshot_rng
from fluffy_bassoon.meta_rl.rlinf_native_ppo_runtime import SidecarMixin
from fluffy_bassoon.meta_rl.rlinf_native_ppo import restore_rng,sha,write


class ProbeActor(PairedActor):
    def verify_fixture_initializer(self,directory):
        # Fresh original SFT must exactly match the fixture's pre-SGD boundary,
        # including its legacy raw optimizer hash. Do not rewrite old payloads.
        expected=json.loads((Path(directory)/f'paired-actor-signature-{self._rank:02d}.json').read_text())
        actual=actor_signature(self,snapshot_rng())
        projected={k:actual[k] for k in expected}
        report=dict(rank=self._rank,expected=expected,actual=projected,exact_identity=expected==projected,
            original_SFT_freshly_initialized=True,fixture_checkpoint_mutated=False)
        write(Path(self.cfg.native_pilot.output)/'fixture-initializer'/f'actor-{self._rank:02d}.json',report)
        assert expected==projected,'Fresh original Sidney model/Adam/scheduler/scaler/RNG differs from fixture'
        return report

    def describe_optimizer(self,label):
        value=evidence(self)
        write(Path(self.cfg.native_pilot.output)/'optimizer-layout'/f'{label}-rank-{self._rank:02d}.json',value)
        return dict(rank=self._rank,optimizer_sha256=value['optimizer_sha256'],state_entries=value['state_entries'])

    def restore_for_diagnosis(self,directory):
        # Diagnostic only: preserve the failing raw-hash comparison and collect
        # its full state/group delta. This method never claims qualification.
        result=SidecarMixin.pilot_restore(self,directory)
        expected=json.loads((Path(directory)/f'paired-actor-signature-{self._rank:02d}.json').read_text())
        actual=actor_signature(self,snapshot_rng())
        report=dict(rank=self._rank,expected=expected,actual=actual,
            equality={k:expected[k]==actual[k] for k in expected},
            frozen_unchanged=self._pilot_frozen_signature()==self._pilot_frozen_before,
            optimizer_semantic_qualified=False,new_optimizer_updates=0)
        write(Path(self.cfg.native_pilot.output)/f'diagnostic-restore-rank-{self._rank:02d}.json',report)
        return report

    def replay_fixture(self,source):
        source=Path(source)
        receipt=source/'training-rollouts'/f'iteration-0001-policy-0000-rank-{self._rank:02d}.json'
        ref=json.loads(receipt.read_text());path=Path(ref['path'])
        assert ref['completed'] and ref['rank']==self._rank and ref['optimizer_steps_before']==0
        assert sha(path)==ref['sha256']
        record=torch.load(path,map_location='cpu',weights_only=False)
        assert record['producing_policy_iteration']==0 and record['optimizer_steps_before']==0
        assert self.optimizer_steps==0 and not self.cfg.actor.model.add_value_head
        self.rollout_batch=record['rollout_batch']
        self._pilot_producing_policy_iteration=0
        restore_rng(record['rng_before_sgd'])
        del record
        before=self.optimizer_steps;started=time.perf_counter()
        self.run_training()
        assert self.optimizer_steps==before+1 and self._pair_audit_micro==8
        result=dict(rank=self._rank,status='complete',optimizer_steps=self.optimizer_steps,
            seconds=time.perf_counter()-started,fixture_path=str(path),fixture_sha256=ref['sha256'],
            new_simulator_controls=0,source_corpus_admission=False)
        write(Path(self.cfg.native_pilot.output)/f'probe-rank-{self._rank:02d}.json',result)
        return result


@hydra.main(version_base='1.1',config_path=str(Path(os.environ['TASK_CODE'])/'configs/native_ppo'),config_name='sidney_phone_two')
def main(cfg):
    cfg=validate_cfg(cfg);validate_pair(cfg)
    assert cfg.native_pilot.qualification_only and cfg.algorithm.adv_type=='grpo'
    root=Path(cfg.native_pilot.output);source=Path(os.environ['TASK_FIXTURE'])/'train'
    mode=os.environ['TASK_PROBE_MODE'];assert mode in ('train','restore')
    cluster=Cluster(cluster_cfg=cfg.cluster,distributed_log_dir=cfg.runner.per_worker_log_path)
    placement=HybridComponentPlacement(cfg,cluster)
    actor=ProbeActor.create_group(cfg).launch(cluster,name=cfg.actor.group_name,
        placement_strategy=placement.get_strategy('actor'))
    try:
        actor.init_worker().wait()
        diagnostic=os.environ.get('TASK_OPTIMIZER_DIAGNOSTIC')=='1'
        if diagnostic and mode=='train':
            before=actor.describe_optimizer('before-save').wait()
            checkpoint=root/'checkpoint0'
            actor.save_checkpoint(str(checkpoint/'actor'),0).wait()
            workers=actor.pilot_capture(str(checkpoint/'pilot'),retain_rollout=False).wait()
            after=actor.describe_optimizer('after-save').wait()
            write(root/'RESULT.json',dict(status='diagnostic_complete_unqualified',before=before,after=after,
                workers=workers,new_optimizer_updates=0,new_simulator_controls=0))
        elif diagnostic:
            checkpoint=Path(os.environ['TASK_PROBE_CHECKPOINT'])
            before=actor.describe_optimizer('before-load').wait()
            actor.load_checkpoint(str(checkpoint/'actor')).wait()
            after=actor.describe_optimizer('after-load').wait()
            restored=actor.restore_for_diagnosis(str(checkpoint/'pilot')).wait()
            write(root/'RESULT.json',dict(status='diagnostic_complete_unqualified',before=before,after=after,
                restored=restored,new_optimizer_updates=0,new_simulator_controls=0,
                optimizer_semantic_qualified=False))
        elif mode=='train':
            initial=source/'sidney-paired-grpo/checkpoints/global_step_0'
            restored=actor.verify_fixture_initializer(str(initial/'pilot')).wait()
            ranks=actor.replay_fixture(str(source)).wait()
            checkpoint=root/'checkpoint1';started=time.perf_counter()
            actor.save_checkpoint(str(checkpoint/'actor'),1).wait()
            workers=actor.pilot_capture(str(checkpoint/'pilot'),retain_rollout=False).wait()
            files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
                for p in sorted(checkpoint.rglob('*')) if p.is_file()]
            write(checkpoint/'BOUNDARY.json',dict(completed=True,actor_only=True,workers=workers,files=files))
            write(root/'checkpoint-timing.json',dict(seconds=time.perf_counter()-started,
                includes_native_DCP_sidecars_signatures_and_SHA256=True))
            write(root/'RESULT.json',dict(status='complete',actor_only=True,mode=mode,ranks=ranks,
                initial_actors_exactly_match_fixture=restored,optimizer_updates=1,new_simulator_controls=0,
                full_grouped_runtime_recovery_qualified=False))
        else:
            checkpoint=Path(os.environ['TASK_PROBE_CHECKPOINT'])
            actor.load_checkpoint(str(checkpoint/'actor')).wait()
            restored=actor.pilot_restore(str(checkpoint/'pilot')).wait()
            assert len(restored)==4 and all(r['restored'] for r in restored)
            write(root/'RESULT.json',dict(status='complete',actor_only=True,mode=mode,
                actors_restored=restored,new_simulator_controls=0,
                full_grouped_runtime_recovery_qualified=False))
    finally:
        actor._close();ray.shutdown()


if __name__=='__main__':main()
