"""Native LIBERO adapter: bounded placements, clean videos, isolated truth.

The author environment still performs every reset, physical action and native
termination/mask calculation. No success termination or auto-reset is enabled.
"""
from copy import deepcopy
import json
from pathlib import Path

from .rlinf_flora_ppo import TASK, digest, sha, write_new


def strip_training_truth(infos):
    """Only transition validity survives; simulator labels never reach F's actor."""
    return {key:value for key,value in infos.items() if key=='valid_action_mask'}


def libero_class(original, worker):
    import numpy as np
    import torch
    from .rlinf_native_ppo import ActionJournal, tensor_hash

    class MatchedLibero(original):
        def get_env_fn_params(self,*args,**kwargs):
            params=super().get_env_fn_params(*args,**kwargs)
            for row in params:
                if row.get('control_freq')!=20:raise ValueError('Bind actual simulator constructor control_freq20')
            return params

        def get_reset_state_ids_all(self):
            offset=int(self.cumsum_trial_id_bins[43])
            if self.is_eval:
                schedule=worker._flora_schedule
                ids=np.asarray([offset+r['nominal_state'] for r in schedule],dtype=np.int64)
                if len(ids)%self.total_num_processes:raise ValueError('Explicit padded evaluation layout required')
                return ids.reshape(-1,self.total_num_processes).T.copy()
            ids=np.arange(offset,offset+40,dtype=np.int64)
            self._generator_ordered.shuffle(ids)
            return ids.reshape(self.total_num_processes,-1)

        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self._flora_round=-1;self._flora_rows=[];self._flora_writers=[]
            self._flora_context=None
            self._flora_journal=worker._flora_journal
            if self.auto_reset or not self.ignore_terminations:raise ValueError('Both arms require full-horizon episodes')

        def reset(self,*args,**kwargs):
            if self._flora_writers:raise ValueError('Cannot drop an incomplete physical episode/video')
            intent=self._flora_journal.begin(self.num_envs*15,phase=f'{worker._flora_phase}/settles')
            result=super().reset(*args,**kwargs)
            self._flora_journal.finish(intent,self.num_envs*15)
            self._flora_round+=1
            if np.any(self.task_ids!=44):raise ValueError('Actual reset escaped task44')
            trials=self.trial_ids.tolist()
            allowed=set(range(40,50)) if self.is_eval and worker._flora_phase=='DEV' else set(range(40))
            if not set(trials)<=allowed:raise ValueError('Actual reset escaped frozen nominal-state partition')
            from rlinf.envs.sim.libero.utils import get_libero_image
            import imageio.v2 as imageio
            # c605's child getattr cannot unwrap OffScreenRenderEnv.env. Bind
            # the real constructor value, checked above before construction.
            self._flora_fps=[20.]*self.num_envs;self._flora_steps=0
            root=worker._flora_root()/f'env-rank-{worker._rank:02d}'
            root.mkdir(parents=True,exist_ok=True)
            self._flora_episode_meta=[]
            for local_env,raw in enumerate(self.current_raw_obs):
                key=f'{worker._flora_phase}-i{worker._flora_iteration:04d}-r{worker._rank:02d}-e{self._flora_round:02d}-{local_env:02d}'
                video=root/f'{key}.mp4'
                if video.exists():raise ValueError('Never overwrite or recollect an existing episode')
                writer=imageio.get_writer(str(video),fps=self._flora_fps[local_env],codec='libx264',
                    quality=8,macro_block_size=None)
                writer.append_data(np.ascontiguousarray(get_libero_image(raw)))
                self._flora_writers.append(writer)
                self._flora_episode_meta.append(dict(id=key,video=video,nominal_state=int(trials[local_env]),
                    actual_reset_state_id=int(self.reset_state_ids[local_env]),
                    initial_observation_sha256=tensor_hash(result[0]),
                    initial_raw_observation_sha256=tensor_hash(raw)))
            return result

        def step(self,*args,**kwargs):
            from rlinf.envs.sim.libero.utils import get_libero_image
            obs,rewards,terms,truncs,infos=super().step(*args,**kwargs)
            self._flora_steps+=1
            if self._flora_steps>400 or torch.any(terms):raise ValueError('Unexpected termination/overshoot')
            for writer,raw in zip(self._flora_writers,self.current_raw_obs):
                writer.append_data(np.ascontiguousarray(get_libero_image(raw)))
            if self._flora_steps==400:
                if not torch.all(truncs):raise ValueError('Incomplete horizon batch cannot be scored')
                truth=infos['episode']['success_at_end'].detach().cpu().tolist()
                for writer in self._flora_writers:writer.close()
                self._flora_writers=[]
                rows=[];monitor=[]
                for local_env,meta in enumerate(self._flora_episode_meta):
                    fps=self._flora_fps[local_env]
                    row=dict(id=meta['id'],task=TASK,camera_order=['agentview'],info_on_video=False,
                        extra_info_on_video=[],raw_rgb_only=True,success_termination_disabled=True,
                        horizon=400,completed=True,video=dict(path=str(meta['video']),sha256=sha(meta['video'])),
                        frame_clock=dict(frame_count=401,native_fps=fps,encoded_fps=fps,resampled=False,
                            physical_steps=400,initial_frame_included=True,
                            timestamp_sha256=digest([i/fps for i in range(401)])),
                        rank=worker._rank,round=self._flora_round,local_env=local_env)
                    rows.append(row)
                    case=(worker._flora_schedule[worker._rank+local_env*worker._world_size]
                        if self.is_eval else dict(nominal_state=meta['nominal_state'],counted=True,draw=None))
                    if case['nominal_state']!=meta['nominal_state']:raise ValueError('Actual initial case differs from schedule')
                    monitor.append(dict(id=meta['id'],case=case,simulator_final_success=bool(truth[local_env]),
                        initial_raw_observation_sha256=meta['initial_raw_observation_sha256'],
                        initial_observation_sha256=meta['initial_observation_sha256'],
                        actual_reset_state_id=meta['actual_reset_state_id'],actual_scene_seed=self.seed*self.num_envs+local_env,
                        native_clock_source='simulator_constructor_control_freq20',
                        physical_actions=415,video_sha256=row['video']['sha256']))
                write_new(worker._flora_root()/f'env-rank-{worker._rank:02d}'/f'round-{self._flora_round:02d}.json',rows)
                side=Path(worker.cfg.flora_ppo.monitoring_root)/worker.cfg.flora_ppo.arm/worker._flora_phase/f'iteration-{worker._flora_iteration:04d}'
                write_new(side/f'rank-{worker._rank:02d}-round-{self._flora_round:02d}.json',monitor)
                self._flora_rows.extend(rows)
            # Both arms receive only zero native rewards. S's final-state binary
            # label is installed by the same pre-GAE barrier as F's graded label.
            return obs,torch.zeros_like(rewards),terms,truncs,strip_training_truth(infos)

        def chunk_step(self,actions):
            intent=self._flora_journal.begin(int(actions.shape[0]*actions.shape[1]),phase=worker._flora_phase)
            result=super().chunk_step(actions)
            self._flora_journal.finish(intent,int(actions.shape[0]*actions.shape[1]))
            return result

    return MatchedLibero


def environment_class():
    import numpy as np
    from rlinf.workers.env.env_worker import EnvWorker
    from .rlinf_native_ppo import ActionJournal, cpu_tree, restore_rng, snapshot_rng
    from .rlinf_native_ppo_runtime import save_tensor

    class MatchedEnvironment(EnvWorker):
        def init_worker(self):
            self._flora_phase='unstarted';self._flora_iteration=0
            from .rlinf_flora_ppo_config import evaluation_schedule
            phase='DEV' if self.cfg.flora_ppo.phase=='train' else self.cfg.flora_ppo.phase
            self._flora_schedule=evaluation_schedule(self._world_size,phase)
            self._flora_journal=ActionJournal(self.cfg.flora_ppo.accounting_root,rank=self._rank,
                world=self._world_size,attempt=self.cfg.flora_ppo.attempt)
            return super().init_worker()

        def _setup_env_and_wrappers(self,env_cls,env_cfg,num_envs_per_stage):
            if self.stage_num!=1:raise ValueError('No pipeline or overlapping environment bootstrap')
            return super()._setup_env_and_wrappers(libero_class(env_cls,self),env_cfg,num_envs_per_stage)

        def _flora_root(self):
            return Path(self.cfg.flora_ppo.exchange_root)/self.cfg.flora_ppo.arm/self._flora_phase/f'iteration-{self._flora_iteration:04d}'

        def flora_phase(self,phase,iteration):
            self._flora_phase=phase;self._flora_iteration=int(iteration)
            for env in self.env_list+self.eval_env_list:
                if env._flora_writers:raise ValueError('Close current physical episode before phase transition')
                env._flora_round=-1;env._flora_rows=[]
            return dict(rank=self._rank,phase=phase,iteration=iteration)

        def flora_records(self,evaluation=False):
            envs=self.eval_env_list if evaluation else self.env_list
            return dict(rank=self._rank,episodes=[r for env in envs for r in env._flora_rows],
                physical_actions=self._flora_journal.charged())

        def flora_capture(self,directory,**kwargs):
            if self._prefetched_train_bootstrap is not None:raise ValueError('No in-flight bootstrap checkpoint')
            samplers=[]
            for mode,envs in [('train',self.env_list),('eval',self.eval_env_list)]:
                for ordinal,env in enumerate(envs):
                    if env._flora_writers:raise ValueError('Pending physics is not a complete batch boundary')
                    samplers.append(dict(mode=mode,ordinal=ordinal,start_idx=env.start_idx,seed=env.seed,
                        reset_state_ids=env.reset_state_ids.copy(),pool=env.reset_state_ids_all.copy(),
                        generator=deepcopy(env._generator.bit_generator.state),
                        ordered_generator=deepcopy(env._generator_ordered.bit_generator.state),
                        round=env._flora_round,records=deepcopy(env._flora_rows)))
            value=dict(rank=self._rank,role='env',rng=snapshot_rng(),samplers=samplers,
                phase=self._flora_phase,iteration=self._flora_iteration,trajectory_step=self._trajectory_step,
                last_obs_list=cpu_tree(self.last_obs_list),last_intervened_info_list=cpu_tree(self.last_intervened_info_list),
                train_prev_done=cpu_tree(getattr(self,'train_prev_done',None)),eval_prev_done=cpu_tree(getattr(self,'eval_prev_done',None)))
            ref=save_tensor(Path(directory)/f'env-{self._rank:02d}.pt',value)
            return dict(ref,role='env',rank=self._rank,complete=True,optimizer_sampler_RNG=True)

        def flora_restore(self,directory,**kwargs):
            import torch
            state=torch.load(Path(directory)/f'env-{self._rank:02d}.pt',map_location='cpu',weights_only=False)
            if state['rank']!=self._rank:raise ValueError('Environment rank changed on restore')
            self._flora_phase=state['phase'];self._flora_iteration=state['iteration'];self._trajectory_step=state['trajectory_step']
            for row in state['samplers']:
                env=(self.env_list if row['mode']=='train' else self.eval_env_list)[row['ordinal']]
                if env.seed!=row['seed']:raise ValueError('Native scene seed changed on restore')
                env.start_idx=row['start_idx'];env.reset_state_ids=row['reset_state_ids'];env.reset_state_ids_all=row['pool']
                env._generator.bit_generator.state=row['generator'];env._generator_ordered.bit_generator.state=row['ordered_generator']
                env._flora_round=row['round'];env._flora_rows=row['records']
            for key in ('last_obs_list','last_intervened_info_list','train_prev_done','eval_prev_done'):
                setattr(self,key,state[key])
            self._prefetched_train_bootstrap=None
            restore_rng(state['rng'])
            return dict(role='env',rank=self._rank,restored=True)

    return MatchedEnvironment
