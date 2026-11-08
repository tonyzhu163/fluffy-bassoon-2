"""Continue the native GRPO learner at a verified whole-iteration boundary."""
import json
import os
from pathlib import Path
import sys

import ray
import torch

import motus_teacher_entry as teacher
from motus_grpo_entry import CapacityActor, CapacityRollout, CapacityEnvironment
from motus_curve import configuration
from fluffy_bassoon.meta_rl.rlinf_native_ppo import write,snapshot_rng,tensor_hash,cpu_tree
from fluffy_bassoon.meta_rl.rlinf_native_ppo_runtime import producing_iteration
from fluffy_bassoon.meta_rl.tensor_archive import read_tensor
from recovery_signature import digest


class CurveActor(CapacityActor):
    def recompute_prev_logprobs(self,batch_size_per_rank):
        metrics=super().recompute_prev_logprobs(batch_size_per_rank)
        assert metrics['actor/original_behavior_logprob_max_abs']==0
        assert metrics['actor/original_behavior_ratio_outside_clip_fraction']==0
        step=producing_iteration(self)
        assert self.optimizer_steps==15*step
        write(Path(self.cfg.native_pilot.output)/'behavior-reference'/f'policy-{step:04d}-rank-{self._rank:02d}.json',
            dict(rank=self._rank,producing_policy_iteration=step,optimizer_steps_before=self.optimizer_steps,
                micro_batch=32,accumulation=8,original_behavior_exact=True,metrics=metrics))
        return metrics


class CurveRollout(CapacityRollout):
    def init_worker(self):
        result=super().init_worker()
        self._curve_reference_count=0
        return result

    def _predict_rollout_actions(self,env_obs,mode='train',**kwargs):
        check=(mode=='eval' and self._pilot_eval_iteration==int(self.cfg.native_pilot.segment_start_iteration))
        if check:
            assert self._curve_reference_count<6
            ordinal=int(self.cfg.native_pilot.curve_reference_first_ordinals[self._rank])+self._curve_reference_count
            path=Path(self.cfg.native_pilot.curve_reference_root)/f'rank-{self._rank:02d}'/f'{ordinal:05d}.pt.gz'
            source=read_tensor(path)
            assert source['mode']=='eval' and source['producing_policy_iteration']==int(self.cfg.native_pilot.segment_start_iteration)
            assert digest(snapshot_rng())==digest(source['rng_before'])
            assert tensor_hash(cpu_tree(env_obs))==source['public_observation_sha256']
        result=super()._predict_rollout_actions(env_obs,mode=mode,**kwargs)
        if check:
            actual=torch.as_tensor(result[0]).detach().cpu().reshape(2,-1)
            expected=source['forward_sampling_fields']['action']
            assert actual.dtype==expected.dtype and torch.equal(actual,expected)
            assert digest(snapshot_rng())==digest(source['rng_after'])
            self._curve_reference_count+=1
        return result

    def curve_reference_result(self):
        assert self._curve_reference_count==6
        return dict(rank=self._rank,source_predictions=6,actions_pixels_and_both_RNG_states_exact=True,
            actual_model_resume_still_under_actor_state_guards=True)


class CurveRunner(teacher.MotusRunner):
    def _dev(self,iteration):
        result=super()._dev(iteration)
        if iteration==int(self.cfg.native_pilot.segment_start_iteration):
            rows=self.rollout.curve_reference_result().wait()
            assert len(rows)==8 and all(r['source_predictions']==6 for r in rows)
            self._pilot_write('resume-baseline-reference.json',dict(all48_predictions_exact=True,workers=rows,
                no_additional_random_draws=True,source_iteration=iteration))
        return result


def main():
    assert os.uname().sysname=='Linux' and os.uname().nodename.startswith('job-')
    plan=json.loads(Path(os.environ['TASK_CURVE_PLAN']).read_bytes())
    assert plan['status']=='admitted' and plan['budget_extension_approved']
    torch.multiprocessing.set_start_method('spawn',force=True)
    cfg=configuration(plan)
    teacher.validate(cfg)
    cfg=teacher.validate_cfg(cfg)
    cluster=teacher.Cluster(cluster_cfg=cfg.cluster,distributed_log_dir=cfg.runner.per_worker_log_path)
    placement=teacher.HybridComponentPlacement(cfg,cluster)
    groups=[]
    try:
        for role,worker in [('actor',CurveActor),('rollout',CurveRollout),('env',CapacityEnvironment)]:
            groups.append(worker.create_group(cfg).launch(cluster,name=cfg[role].group_name,
                placement_strategy=placement.get_strategy(role)))
        runner=CurveRunner(cfg=cfg,actor=groups[0],rollout=groups[1],env=groups[2],reward=None)
        runner.init_workers()
        assert runner.global_step==plan['start_iteration']
        runner.run()
    finally:
        failed=sys.exc_info()[0] is not None;errors=[]
        for group in groups:
            if ray.is_initialized():
                try:group._close()
                except Exception as error:errors.append(str(error))
        ray.shutdown()
        if errors:
            write(Path(os.environ['TASK_OUT'])/'cleanup-errors.json',dict(errors=errors,primary_failure_preserved=failed))
            if not failed:raise RuntimeError('Owned continuation cleanup failed')


if __name__=='__main__':main()
