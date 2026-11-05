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
from fluffy_bassoon.meta_rl.rlinf_native_ppo import restore_rng,sha,write


class ProbeActor(PairedActor):
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
        if mode=='train':
            initial=source/'sidney-paired-grpo/checkpoints/global_step_0'
            actor.load_checkpoint(str(initial/'actor')).wait()
            restored=actor.pilot_restore(str(initial/'pilot')).wait()
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
                initial_actors_restored=restored,optimizer_updates=1,new_simulator_controls=0,
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
