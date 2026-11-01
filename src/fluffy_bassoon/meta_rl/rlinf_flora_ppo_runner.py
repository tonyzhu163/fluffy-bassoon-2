"""Serial official PPO loop with an all-rank terminal-label barrier.

No custom GAE/loss/optimizer. A label failure leaves the original on-policy
batch and preceding complete model/Adam checkpoint intact, before any SGD.
"""
import json
from pathlib import Path
import time

from rlinf.runners.embodied_runner import EmbodiedRunner

from .rlinf_flora_ppo import digest,freeze_batch,commit_labels,flora_jobs,sha,write_new,INITIALIZER_REPO,INITIALIZER_REVISION,INITIALIZER_WEIGHT_SHA,NORM_SHA
from .rlinf_flora_ppo_runtime import read_ref,validate_pending_checkpoint
from .rlinf_flora_ppo_monitor import ProgressLog,stopping_decision,whole_iteration_admission
from .rlinf_flora_ppo_config import validate_config


class MatchedRunner(EmbodiedRunner):
    def init_workers(self):
        self.contract=validate_config(self.cfg)
        self.profile=read_ref(dict(path=self.cfg.flora_ppo.profile_path,sha256=self.cfg.flora_ppo.profile_sha256))
        self._check_profile()
        if self.cfg.runner.only_eval:
            # The upstream entry creates the group; do not initialize a model,
            # value head or optimizer on that actor for a no-SGD qualification.
            handles=[self.rollout.init_worker(),self.env.init_worker()]
            for handle in handles:handle.wait()
        else:super().init_workers()

    def _check_profile(self):
        p=self.profile
        if (p['arm']!=self.cfg.flora_ppo.arm or p['phase']!=self.cfg.flora_ppo.phase
                or p['world_size']!=self.contract['world'] or not 1<=p['allocated_gpu_count']<=8
                or p['allocated_gpu_count']<self.contract['world']
                or not 0<p['scheduler_wall_seconds']<8*3600 or p['deadline_unix']<=time.time()
                or not p.get('allocation_id') or not p.get('actual_gpu_probe')
                or not p.get('initializer_artifact_receipt') or p.get('source_commit') is None):
            raise ValueError('Concrete fresh allocation/probe/initializer/finite-clock profile required')
        for key in ('actual_gpu_probe','initializer_artifact_receipt'):read_ref(p[key])
        if p['phase']=='train':
            for key in ('cpu_qualification','loading_qualification','baseline_validation'):
                ref=read_ref(p[key])
                if ref.get('completed',ref.get('go')) is not True:raise ValueError(f'Unqualified {key}')
            if not read_ref(p['baseline_validation'])['go']:raise ValueError('Fresh64 baseline did not pass gates')
            loaded=read_ref(p['loading_qualification'])
            if (loaded.get('repository_id')!=INITIALIZER_REPO or loaded.get('revision')!=INITIALIZER_REVISION
                    or loaded.get('model_sha256')!=INITIALIZER_WEIGHT_SHA or loaded.get('normalization_sha256')!=NORM_SHA
                    or loaded.get('backbone_missing_keys')!=[] or loaded.get('unexpected_keys')!=[]
                    or loaded.get('value_head_seed')!=42):
                raise ValueError('Actual strict published-release backbone load and common value-head seed required')
            if not 0<p['estimated_iteration_seconds'] or p['closure_reserve_seconds']<180:
                raise ValueError('Finite whole-iteration reservation and closure time required')

    def _root(self,phase,iteration):
        return Path(self.cfg.flora_ppo.exchange_root)/self.cfg.flora_ppo.arm/phase/f'iteration-{iteration:04d}'

    def _emit(self,log,**updates):
        # Owner canonical cost view, read only. No second allocation ledger.
        view=json.loads(Path(self.profile['cost_status_path']).read_text())
        known=view['gpu_seconds_actual']
        updates.update(aggregate_gpu_seconds=known,closed_actual_gpu_seconds=known,
            gpu_seconds_reserved=view['gpu_seconds_reserved'],
            this_open_job_elapsed_gpu_seconds=max(0,time.time()-self.profile['allocated_start_unix'])*self.profile['allocated_gpu_count'],
            gpu_charge_semantics='canonical owner ledger actual; open-job elapsed and reserved capacities shown separately',
            elapsed_seconds=time.time()-self.profile['allocated_start_unix'],
            remaining_wall_seconds=max(0,self.profile['deadline_unix']-time.time()))
        return log.emit(**updates)

    def _set_step(self):
        for group in (self.actor,self.rollout,self.env):group.set_global_step(self.global_step).wait()

    def _capture_boundary(self):
        self._save_checkpoint() # Author DCP: model, critic, Adam and LR schedule, all ranks.
        root=Path(self.cfg.runner.logger.log_path)/self.cfg.runner.logger.experiment_name/'checkpoints'/f'global_step_{self.global_step}'
        rows=[]
        for group in (self.actor,self.rollout,self.env):rows.extend(group.flora_capture(str(root/'comparison')).wait())
        files=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in sorted((root/'actor').rglob('*')) if p.is_file()]
        value=dict(completed=True,iteration=self.global_step,model_optimizer_files=files,workers=rows,
            profile_sha256=self.cfg.flora_ppo.profile_sha256,scope='closed full rollout/update/evaluation boundary')
        # Initial boundary and post-evaluation boundaries each have distinct steps.
        write_new(root/'comparison-boundary.json',value)
        return dict(path=str(root/'comparison-boundary.json'),sha256=sha(root/'comparison-boundary.json'))

    def _records(self,evaluation=False):
        rows=self.env.flora_records(evaluation=evaluation).wait()
        episodes=[e for row in rows for e in row['episodes']]
        return episodes,sum(row['physical_actions'] for row in rows)

    def _evaluate(self,phase):
        self.env.flora_phase(phase,self.global_step).wait()
        self.rollout.flora_evaluation_rng().wait()
        try:metrics=self.evaluate() # Author action sampler and actual simulator.
        finally:self.rollout.flora_evaluation_rng(restore_training=True).wait()
        episodes,actions=self._records(evaluation=True)
        side=Path(self.cfg.flora_ppo.monitoring_root)/self.cfg.flora_ppo.arm/phase/f'iteration-{self.global_step:04d}'
        monitor=[r for path in sorted(side.glob('rank-*-round-*.json')) for r in json.loads(path.read_text())]
        expected=50 if phase=='DEV' else (64 if phase=='baseline' else 1)
        scored=[r for r in monitor if r['case']['counted']]
        if len(scored)!=expected or len(monitor)!=len(episodes):raise ValueError('Incomplete actual evaluation vector')
        root=self._root(phase,self.global_step)
        write_new(root/'episodes.json',episodes)
        with (root/'jobs.jsonl').open('x') as stream:
            for row in flora_jobs(episodes):stream.write(json.dumps(row)+'\n')
        result=dict(completed=True,metric_source='simulator_DEV_final_success',iteration=self.global_step,
            episodes=expected,successes=sum(r['simulator_final_success'] for r in scored),
            unique_nominal_states=len({r['case']['nominal_state'] for r in scored}),
            padding_episodes=len(monitor)-expected,physical_actions=actions,official_metrics=metrics)
        write_new(root/'evaluation.json',result)
        return result,episodes,actions

    def _label_barrier(self,previous_checkpoint,pending_resume=False):
        root=self._root('train',self.global_step+1)
        if pending_resume:
            batch=json.loads((root/'batch.json').read_text())
            validate_pending_checkpoint(batch,previous_checkpoint)
        else:
            episodes,_=self._records()
            pending=self.actor.capture_pending_for_labels().wait()
            recovery=[]
            for group in (self.actor,self.rollout,self.env):recovery.extend(group.flora_capture(str(root/'recovery')).wait())
            previous=read_ref(previous_checkpoint)
            batch=freeze_batch(root/'batch.json',arm=self.cfg.flora_ppo.arm,iteration=self.global_step+1,
                episodes=episodes,rollout_ref=dict(producer_iteration=self.global_step,checkpoint=previous_checkpoint),
                recovery_refs=recovery,rank_pending_receipts={str(r['rank']):r for r in pending},
                profile_sha256=self.cfg.flora_ppo.profile_sha256,policy_sha256=digest(previous['model_optimizer_files']))
        ready_path=root/'ready.json'
        if not ready_path.exists() and self.cfg.flora_ppo.arm=='S':
            side=Path(self.cfg.flora_ppo.monitoring_root)/'S'/'train'/f'iteration-{self.global_step+1:04d}'
            rows=[r for path in sorted(side.glob('rank-*-round-*.json')) for r in json.loads(path.read_text())]
            labels={r['id']:int(r['simulator_final_success']) for r in rows}
            commit_labels(ready_path,batch,labels,label_receipt=dict(completed=True,input_batch_sha256=digest(batch),
                source='isolated simulator final-state labels'),baseline_validation_sha256=self.profile['baseline_validation']['sha256'])
        if not ready_path.exists():
            request=root/'label-request.json'
            if not request.exists():write_new(request,dict(batch=dict(path=str(root/'batch.json'),sha256=sha(root/'batch.json')),
                baseline_validation=self.profile['baseline_validation'],phase='train',completed=False))
            # The separately allocated Flora process consumes this bounded request.
            # No inference dependencies, native truth or fallback enter the actor.
            while not ready_path.exists():
                if time.time()+self.profile['closure_reserve_seconds']>=self.profile['deadline_unix']:
                    raise TimeoutError('Labels incomplete: raw pending batch preserved; no GAE or SGD')
                time.sleep(1)
        batch_ref=dict(path=str(root/'batch.json'),sha256=sha(root/'batch.json'))
        ready_ref=dict(path=str(ready_path),sha256=sha(ready_path))
        self.actor.install_reward_refs(batch_ref,ready_ref).wait()

    def run(self):
        log=ProgressLog(Path(self.cfg.flora_ppo.output)/'progress',self.cfg.flora_ppo.arm,
            contract_sha256=self.cfg.flora_ppo.profile_sha256).start()
        try:
            if self.cfg.runner.only_eval:
                self._emit(log,phase=self.cfg.flora_ppo.phase)
                result,_,actions=self._evaluate(self.cfg.flora_ppo.phase)
                self._emit(log,phase='complete',episodes_complete=result['episodes'],physical_actions=actions,
                    evaluation_successes=f'{result["successes"]}/{result["episodes"]}')
                return
            self._set_step()
            resume=self.cfg.runner.get('resume_dir')
            if resume:
                previous=dict(path=str(Path(resume)/'comparison-boundary.json'),sha256=sha(Path(resume)/'comparison-boundary.json'))
                for group in (self.actor,self.rollout,self.env):group.flora_restore(str(Path(resume)/'comparison')).wait()
            else:previous=self._capture_boundary()
            pending=bool(self.cfg.flora_ppo.get('pending_resume',False))
            if pending:
                root=self._root('train',self.global_step+1)
                if list(root.glob('rank-*/UPDATE-STARTED.json')):raise ValueError('Partially updated batch needs owner recovery, never automatic replay')
                validate_pending_checkpoint(json.loads((root/'batch.json').read_text()),previous)
                for group in (self.actor,self.rollout,self.env):group.flora_restore(str(root/'recovery'),pending_directory=str(root)).wait()
            evaluations=[]
            if self.cfg.flora_ppo.get('evaluation_history'):
                evaluations=read_ref(self.cfg.flora_ppo.evaluation_history)['evaluations']
            baseline=read_ref(self.profile['baseline_validation'])['initial_success_rate']
            while self.global_step<self.max_steps:
                _,charged=self._records()
                remaining=self.profile['deadline_unix']-time.time()
                if (remaining<self.profile['estimated_iteration_seconds']+self.profile['closure_reserve_seconds']
                        or not whole_iteration_admission(charged_actions=charged+self.profile['common_actions_attribution'],
                            projected_actions=self.contract['estimated_iteration_actions'],charged_gpu_seconds=0,projected_gpu_seconds=0)):
                    self._emit(log,phase='stopped',stop_reason='hard_limit_before_next_iteration',last_full_recovery=previous);break
                self._set_step();start=time.monotonic()
                if not pending:
                    self.env.flora_phase('train',self.global_step+1).wait()
                    self.update_rollout_weights()
                    self._emit(log,phase='rollout',iteration=self.global_step+1)
                    env_handle=self.env.interact(input_channel=self.env_channel,rollout_channel=self.rollout_channel,reward_channel=None)
                    rollout_handle=self.rollout.generate(input_channel=self.rollout_channel,output_channel=self.env_channel,actor_channel=self.actor_channel)
                    self.actor.recv_rollout_trajectories(input_channel=self.actor_channel).wait()
                    rollout_handle.wait();env_handle.wait()
                rollout_seconds=time.monotonic()-start
                self._emit(log,phase='label_barrier',episodes_complete=128,phase_seconds=dict(rollout=rollout_seconds))
                label_start=time.monotonic()
                self._label_barrier(previous,pending_resume=pending);pending=False
                label_seconds=time.monotonic()-label_start
                self.actor.compute_advantages_and_returns().wait()
                self._emit(log,phase='official_PPO_update',labels_complete=128)
                update_start=time.monotonic()
                diagnostics=self.actor.run_training().wait()
                update_seconds=time.monotonic()-update_start
                self.global_step+=1;self._set_step();self.update_rollout_weights()
                self._emit(log,phase='DEV_evaluation')
                eval_start=time.monotonic()
                result,_,charged=self._evaluate('DEV')
                eval_seconds=time.monotonic()-eval_start
                previous=self._capture_boundary()
                result.update(full_boundary_checkpoint=True,checkpoint_sha256=previous['sha256'],health='healthy')
                evaluations.append(result)
                decision=stopping_decision(evaluations,baseline_success_rate=baseline)
                write_new(Path(self.cfg.flora_ppo.output)/f'iteration-{self.global_step:04d}.json',
                    dict(evaluations=evaluations,decision=decision,checkpoint=previous,diagnostics=diagnostics,elapsed_seconds=time.monotonic()-start))
                self._emit(log,phase='iteration_complete',iteration=self.global_step,physical_actions=charged,
                    last_full_recovery=previous,stop_reason=decision['reason'],ppo_diagnostics=diagnostics,
                    evaluation_successes=f'{result["successes"]}/50',best_success_rate=max(baseline,max(r['successes']/50 for r in evaluations)),
                    plateau_progress=decision.get('window_improvement_percentage_points'),gpu_memory=self.actor.flora_memory().wait(),
                    phase_seconds=dict(rollout=rollout_seconds,flora=label_seconds,update=update_seconds,eval=eval_seconds))
                if decision['stop']:break
            self._emit(log,phase='closed',last_full_recovery=previous)
            service=Path(self.cfg.flora_ppo.exchange_root)/self.cfg.flora_ppo.arm/'SERVICE-CLOSED.json'
            if not service.exists():write_new(service,dict(completed=True,iteration=self.global_step,checkpoint=previous))
        except BaseException as error:
            log.emit(phase='failed',error=f'{type(error).__name__}: {error}',stop_reason='failure_not_convergence')
            raise
        finally:
            log.close();self._finish_run()
