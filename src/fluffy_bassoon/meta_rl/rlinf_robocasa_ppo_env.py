"""Native365 adapter for the existing terminal-reward worker and action journal.

Native create_env/reset/step, observations, action mapping and masks are used
once. A fresh seeded native child per episode makes held-out case replay exact.
Constructor/reset controls and MuJoCo ticks are observed inside that child;
unresolved parent intents remain conservatively charged by ActionJournal.
"""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import time

from .rlinf_flora_ppo import digest,sha,write_new
from .rlinf_flora_ppo_env import strip_training_truth
from .rlinf_robocasa_ppo import TASKS,case_bank,evaluation_task,protocol


def counted_native_factory(factory,*,reset_control_limit=128,physics_tick_limit):
    """Owned native child only. Instrument real base calls, including reset internals."""
    def create():
        from robosuite.environments.base import MujocoEnv
        from robosuite.utils.binding_utils import MjSim
        # first_success_control: 1-based policy control at which the native
        # success check first held (None if never). Monitoring-only truth used
        # to choose task horizons; it never reaches the learner or Flora.
        counts=dict(controls=0,reset_controls=0,physics_ticks=0,step1_calls=0,
            phase='reset',control_time=None,clock=None,first_success_control=None)
        step=MujocoEnv.step
        def counted_step(env,action):
            if counts['phase']=='reset' and counts['reset_controls']>=reset_control_limit:
                raise RuntimeError('Native constructor/reset exceeded its prospective control reserve')
            counts['controls']+=1
            if counts['phase']=='reset':counts['reset_controls']+=1
            result=step(env,action)
            counts['control_time']=float(env.cur_time)
            if (counts['phase']=='policy' and counts['first_success_control'] is None
                    and hasattr(env,'_check_success') and bool(env._check_success())):
                counts['first_success_control']=counts['controls']-counts['reset_controls']
            return result
        MujocoEnv.step=counted_step
        # lite_physics uses step1+step2, ordinary physics uses step. A completed
        # step2 and a completed step are each one integration tick, never two.
        for name in ('step','step1','step2'):
            original=getattr(MjSim,name)
            def counted_tick(sim,*args,_original=original,_name=name,**kwargs):
                if counts['physics_ticks']>=physics_tick_limit:raise RuntimeError('Native physics tick bound exceeded')
                if _name=='step1':counts['step1_calls']+=1
                else:counts['physics_ticks']+=1
                return _original(sim,*args,**kwargs)
            setattr(MjSim,name,counted_tick)
        env=factory()
        original_reset=env.reset
        def reset(*args,**kwargs):
            counts['phase']='reset';counts['first_success_control']=None
            try:result=original_reset(*args,**kwargs)
            finally:counts['phase']='policy'
            # RoboCasa defers time initialization until native reset. Read
            # actual initialized fields, without supplying a guessed frequency.
            clock=dict(control_freq=float(env.control_freq),control_timestep=float(env.control_timestep),
                model_timestep=float(env.model_timestep),lite_physics=bool(env.lite_physics))
            if (clock['control_freq']<=0 or clock['model_timestep']<=0
                    or abs(clock['control_timestep']*clock['control_freq']-1)>1e-9
                    or not 1<=int(clock['control_timestep']/clock['model_timestep'])<=128):
                raise ValueError('Actual native control/physics clock outside predeclared finite limits')
            counts['clock']=clock
            counts['control_time']=float(env.cur_time)
            return result
        env.reset=reset
        env._flora_accounting=counts
        return env
    return create


def robocasa_class(original,worker):
    import numpy as np
    import torch
    from .rlinf_native_ppo import cpu_tree,tensor_hash
    from .rlinf_native_ppo_runtime import save_tensor

    class MatchedRoboCasa(original):
        def __init__(self,*args,**kwargs):
            self._flora_round=-1;self._flora_rows=[];self._flora_writers=[]
            self._flora_current=None;self._flora_cases=[]
            super().__init__(*args,**kwargs)
            if self.auto_reset or not self.ignore_terminations:raise ValueError('Native full horizon/no success termination required')

        def _init_env(self):
            # Native __init__ would eagerly construct unused train/eval scenes.
            # Defer its unchanged factories until the actual bank is selected.
            self._refresh_task_context()

        def _counts(self):
            rows=self.env.get_env_attr('_flora_accounting')
            if len(rows)!=self.num_envs or any(not isinstance(r,dict) for r in rows):
                raise ValueError('Actual native child accounting missing')
            return rows

        def _public(self,obs):
            return {k:v for k,v in obs.items() if k in
                ('main_images','wrist_images','extra_view_images','states','task_descriptions')}

        def reset(self,env_idx=None,options=None):
            if env_idx is not None or options is not None:raise ValueError('Only whole native episode vectors may reset')
            if self._flora_writers:raise ValueError('Never abandon an incomplete native episode')
            if time.time()>=worker._flora_profile['deadline_unix']-worker._flora_profile['closure_reserve_seconds']:
                raise TimeoutError('Preserve closed boundary before a new physical episode')
            self._flora_round+=1
            phase=worker._flora_phase
            is_eval=bool(self.cfg.is_eval)
            if is_eval and phase=='train':raise ValueError('Evaluation reset cannot consume training bank')
            task=evaluation_task(str(worker.cfg.flora_ppo.task),phase)
            bank=case_bank(task,phase,iteration=worker._flora_iteration,round_index=self._flora_round)
            offset=self._flora_round*self.num_envs*worker._world_size if phase in ('baseline','smoke') else 0
            self._flora_cases=[bank[offset+worker._rank+i*worker._world_size] for i in range(self.num_envs)]
            if hasattr(self,'env'):self.env.close();del self.env
            self.cfg=deepcopy(self.cfg)
            self.cfg.task_filter=[task];self.cfg.task_soup=TASKS[task][1];self.cfg.task_mode=TASKS[task][2]
            self.cfg.max_episode_steps=TASKS[task][0];self.cfg.max_steps_per_rollout_epoch=TASKS[task][0]
            self.task_specs=self._load_task_specs();self.num_tasks=len(self.task_specs)
            if self.num_tasks!=1 or self.task_specs[0]['task_name']!=task or int(self.task_specs[0]['horizon'])!=TASKS[task][0]:
                raise ValueError('Actual native task registry/horizon differs from frozen bank')
            self.task_ids=np.zeros(self.num_envs,dtype=np.int32)
            self.env_seeds=[row['native_seed'] for row in self._flora_cases]
            self._refresh_task_context();self._is_start=True
            horizon=TASKS[task][0]
            import os
            root=worker._flora_root()/f'env-rank-{worker._rank:02d}'
            root.mkdir(parents=True,exist_ok=True)
            stat=os.statvfs(root)
            if stat.f_bavail*stat.f_frsize<16*(1<<30):raise RuntimeError('Insufficient shared free space before native data')
            # Parent intent is durable BEFORE create_env/reset can issue commands.
            intent=worker._flora_journal.begin(self.num_envs*128,phase=f'{phase}/native_constructor_reset')
            from rlinf.envs.sim.robocasa.venv import RobocasaSubprocEnv
            self.env=RobocasaSubprocEnv([counted_native_factory(fn,physics_tick_limit=(horizon+128)*128)
                for fn in super().get_env_fns()])
            result=super().reset()
            counts=self._counts()
            worker._flora_journal.finish(intent,sum(r['reset_controls'] for r in counts))
            self._flora_steps=0;self._flora_chunk=0;self._flora_task=task
            self._flora_meta=[];self._flora_current=self._public(result[0])
            import imageio.v2 as imageio
            from .robocasa_video import EpisodeVideos, raw_frames
            for local,(raw,case,counter) in enumerate(zip(self.current_raw_obs,self._flora_cases,counts)):
                key=f'{phase}-i{worker._flora_iteration:04d}-r{worker._rank:02d}-e{self._flora_round:02d}-{local:02d}'
                fps=counter['clock']['control_freq']
                writer=EpisodeVideos(root,key,fps=fps,writer_factory=imageio.get_writer)
                self._flora_writers.append(writer)
                writer.append(raw_frames(raw))
                meta=dict(id=key,recorder=writer,case=case,clock=counter['clock'],timestamps=[counter['control_time']],
                    initial_raw_sha256=tensor_hash(raw),initial_public_sha256=tensor_hash(self._flora_current),
                    constructor_reset_controls=counter['reset_controls'],reset_physics_ticks=counter['physics_ticks'])
                self._flora_meta.append(meta)
                save_tensor(root/f'{key}-initial.pt',dict(case=case,raw=cpu_tree(raw),
                    public=cpu_tree(self._flora_current),native_ep_meta=deepcopy(self.current_info_list[local]),
                    native_counts=deepcopy(counter)))
            write_new(root/f'round-{self._flora_round:02d}-clock.json',
                dict(task=task,cases=self._flora_cases,actual_native_counters=counts,
                    clock_source='actual native constructor fields and observed base controller/MjSim calls'))
            return self._flora_current,strip_training_truth(result[1])

        def step(self,actions=None,auto_reset=True):
            if actions is None or self.is_start:
                result=super().step(actions,auto_reset=False)
                return self._public(result[0]),torch.zeros_like(result[1]),result[2],result[3],strip_training_truth(result[4])
            horizon=TASKS[self._flora_task][0]
            if self._flora_steps>=horizon:raise ValueError('No action beyond complete native horizon')
            obs,rewards,terms,truncs,infos=super().step(actions,auto_reset=False)
            self._flora_steps+=1
            if torch.any(terms):raise ValueError('Success-driven termination is forbidden in both arms')
            counts=self._counts()
            from .robocasa_video import REWARD_CAMERAS, raw_frames
            for writer,raw,meta,counter in zip(self._flora_writers,self.current_raw_obs,self._flora_meta,counts):
                writer.append(raw_frames(raw))
                meta['timestamps'].append(counter['control_time'])
            public=self._public(obs)
            if self._flora_steps==horizon:
                if not torch.all(truncs):raise ValueError('Actual native horizon must close every column')
                truth=infos['episode']['success_at_end'].detach().cpu().tolist()
                for writer in self._flora_writers:writer.close()
                self._flora_writers=[]
                rows=[];monitor=[]
                for local,(meta,counter) in enumerate(zip(self._flora_meta,counts)):
                    timestamps=meta['timestamps'];fps=meta['clock']['control_freq']
                    if (len(timestamps)!=horizon+1 or any(abs((b-a)-1/fps)>1e-7
                            for a,b in zip(timestamps,timestamps[1:]))
                            or counter['controls']-counter['reset_controls']!=horizon):
                        raise ValueError('Observed native frame/control clock differs from full physical horizon')
                    row=dict(id=meta['id'],task=self.task_descriptions[local],camera_order=list(REWARD_CAMERAS),
                        info_on_video=False,extra_info_on_video=[],raw_rgb_only=True,success_termination_disabled=True,
                        horizon=horizon,completed=True,videos=meta['recorder'].refs(),
                        frame_clock=dict(frame_count=horizon+1,native_fps=fps,encoded_fps=fps,resampled=False,
                            physical_steps=horizon,initial_frame_included=True,timestamp_sha256=digest(timestamps)),
                        rank=worker._rank,round=self._flora_round,local_env=local)
                    rows.append(row)
                    monitor.append(dict(id=meta['id'],case=meta['case'],simulator_final_success=bool(truth[local]),
                        initial_raw_observation_sha256=meta['initial_raw_sha256'],
                        initial_observation_sha256=meta['initial_public_sha256'],
                        native_ep_meta=deepcopy(self.current_info_list[local].get('ep_meta')),
                        physical_actions=counter['controls'],native_counts=counter,video_refs_sha256=digest(row['videos'])))
                write_new(worker._flora_root()/f'env-rank-{worker._rank:02d}'/f'round-{self._flora_round:02d}.json',rows)
                side=Path(worker.cfg.flora_ppo.monitoring_root)/worker.cfg.flora_ppo.arm/worker._flora_phase/f'iteration-{worker._flora_iteration:04d}'
                write_new(side/f'rank-{worker._rank:02d}-round-{self._flora_round:02d}.json',monitor)
                self._flora_rows.extend(rows)
            return public,torch.zeros_like(rewards),terms,truncs,strip_training_truth(infos)

        def chunk_step(self,actions):
            requested=int(actions.shape[0]*actions.shape[1])
            intent=worker._flora_journal.begin(requested,phase=worker._flora_phase)
            before=self._flora_current
            result=super().chunk_step(actions)
            worker._flora_journal.finish(intent,requested)
            # Retain actually decoded/mapped commands separately from upstream
            # model-space actions/logp. This never mutates the caller's array.
            from rlinf.envs.sim.robocasa365.robocasa365_env import map_released_control_scores
            trace=dict(case=deepcopy(self._flora_cases),iteration=worker._flora_iteration,
                phase=worker._flora_phase,round=self._flora_round,chunk=self._flora_chunk,
                current_observation=cpu_tree(before),decoded_actions=cpu_tree(actions),
                physical_actions=cpu_tree(map_released_control_scores(actions,self.action_space_cfg)),
                outputs=cpu_tree(result),evaluation_data=bool(self.cfg.is_eval))
            path=worker._flora_root()/f'env-rank-{worker._rank:02d}'/f'round-{self._flora_round:02d}-chunk-{self._flora_chunk:03d}.pt.gz'
            with gzip.open(path,'xb',compresslevel=1) as stream:torch.save(trace,stream)
            self._flora_current=result[0][-1];self._flora_chunk+=1
            return result

    return MatchedRoboCasa


def environment_class():
    from rlinf.workers.env.env_worker import EnvWorker
    from .rlinf_native_ppo import ActionJournal,cpu_tree,restore_rng,snapshot_rng
    from .rlinf_native_ppo_runtime import save_tensor
    from .rlinf_flora_ppo_runtime import read_ref

    class MatchedEnvironment(EnvWorker):
        def init_worker(self):
            self._flora_phase='unstarted';self._flora_iteration=0
            self._flora_profile=read_ref(dict(path=self.cfg.flora_ppo.profile_path,sha256=self.cfg.flora_ppo.profile_sha256))
            self._flora_journal=ActionJournal(self.cfg.flora_ppo.accounting_root,rank=self._rank,
                world=self._world_size,attempt=self.cfg.flora_ppo.attempt)
            return super().init_worker()

        def _setup_env_and_wrappers(self,env_cls,env_cfg,num_envs_per_stage):
            if self.stage_num!=1:raise ValueError('No pipeline/overlapping bootstrap in matched PPO')
            return super()._setup_env_and_wrappers(robocasa_class(env_cls,self),env_cfg,num_envs_per_stage)

        def _flora_root(self):
            return Path(self.cfg.flora_ppo.exchange_root)/self.cfg.flora_ppo.arm/self._flora_phase/f'iteration-{self._flora_iteration:04d}'

        def flora_phase(self,phase,iteration):
            self._flora_phase=phase;self._flora_iteration=int(iteration)
            task=evaluation_task(str(self.cfg.flora_ppo.task),phase)
            if phase!='train':self.n_eval_chunk_steps=TASKS[task][0]//50
            for env in self.env_list+self.eval_env_list:
                if env._flora_writers:raise ValueError('Close actual episode before switching phase')
                env._flora_round=-1;env._flora_rows=[]
            return dict(rank=self._rank,phase=phase,iteration=iteration)

        def flora_records(self,evaluation=False):
            envs=self.eval_env_list if evaluation else self.env_list
            return dict(rank=self._rank,episodes=[r for env in envs for r in env._flora_rows],
                physical_actions=self._flora_journal.charged())

        def flora_capture(self,directory,**kwargs):
            if self._prefetched_train_bootstrap is not None:raise ValueError('No in-flight bootstrap recovery')
            samplers=[]
            for mode,envs in [('train',self.env_list),('eval',self.eval_env_list)]:
                for ordinal,env in enumerate(envs):
                    if env._flora_writers:raise ValueError('Only complete episode-boundary recovery supported')
                    samplers.append(dict(mode=mode,ordinal=ordinal,seed=env.seed,env_seeds=deepcopy(env.env_seeds),
                        generator=deepcopy(env._generator.bit_generator.state),task_ids=env.task_ids.copy(),
                        ordered_cursor=env._ordered_task_cursor,round=env._flora_round,records=deepcopy(env._flora_rows)))
            value=dict(role='env',rank=self._rank,rng=snapshot_rng(),samplers=samplers,
                phase=self._flora_phase,iteration=self._flora_iteration,trajectory_step=self._trajectory_step,
                last_obs_list=cpu_tree(self.last_obs_list),last_intervened_info_list=cpu_tree(self.last_intervened_info_list),
                train_prev_done=cpu_tree(getattr(self,'train_prev_done',None)),eval_prev_done=cpu_tree(getattr(self,'eval_prev_done',None)),
                resume_scope='complete episodes only; next native reset reconstructs the next frozen constructor-seed case')
            ref=save_tensor(Path(directory)/f'env-{self._rank:02d}.pt',value)
            return dict(ref,role='env',rank=self._rank,complete=True,optimizer_sampler_RNG=True)

        def flora_restore(self,directory,**kwargs):
            import torch
            state=torch.load(Path(directory)/f'env-{self._rank:02d}.pt',map_location='cpu',weights_only=False)
            if state['rank']!=self._rank:raise ValueError('No rank/layout migration during recovery')
            self.flora_phase(state['phase'],state['iteration']);self._trajectory_step=state['trajectory_step']
            for row in state['samplers']:
                env=(self.env_list if row['mode']=='train' else self.eval_env_list)[row['ordinal']]
                if env.seed!=row['seed']:raise ValueError('Native sampler initializer changed')
                env.env_seeds=row['env_seeds'];env.task_ids=row['task_ids'];env._ordered_task_cursor=row['ordered_cursor']
                env._generator.bit_generator.state=row['generator'];env._flora_round=row['round'];env._flora_rows=row['records']
            for key in ('last_obs_list','last_intervened_info_list','train_prev_done','eval_prev_done'):setattr(self,key,state[key])
            self._prefetched_train_bootstrap=None
            restore_rng(state['rng'])
            return dict(role='env',rank=self._rank,restored=True)

    return MatchedEnvironment
