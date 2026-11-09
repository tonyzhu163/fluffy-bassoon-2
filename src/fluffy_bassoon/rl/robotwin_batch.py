"""One batched frozen motor and process-owned measured RoboTwin simulators.

The same API serves residual training and evaluation. Physics never crosses a
worker's owning thread; model work stays in the parent. Native batching is a
declared numerical condition, not an assertion of singleton action equivalence.
"""
from dataclasses import asdict
import hashlib
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

from .robotwin_execution import chunk_execution,load_accounted_controller,install_accounted_controller
from .robotwin_motor import case_noise,stack_public_observations
from .robotwin_production_process import ProductionSimulatorPool,_NativeSimulator,_cpu_tree


REFERENCE_SHA256 = '6b3f74b038509ef989913b3f6f37c74e1d0b7156d201a1bf0100f9ccb6b0d46b'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def accounted_controller_source(robotwin_root, output, *, waypoint_controller=False):
    """Prepare a job-owned method copy; never edit the external native checkout."""
    if type(waypoint_controller) is not bool:
        raise ValueError('waypoint_controller must be a boolean')
    reference = Path(robotwin_root)/'envs/_base_task.py'
    if _sha(reference) != REFERENCE_SHA256:
        raise ValueError('Unqualified native controller preimage')
    output = Path(output)
    (output/'envs').mkdir(parents=True,exist_ok=False)
    target = output/'envs/_base_task.py';shutil.copyfile(reference,target)
    patch_name = ('robotwin-dca9ec68-waypoint-controller.patch'
                  if waypoint_controller else 'robotwin-dca9ec68-execution-accounting.patch')
    patch = Path(__file__).resolve().parents[3]/'patches'/patch_name
    subprocess.run(['patch','--fuzz=0','-p1','-d',str(output),'-i',str(patch)],check=True,
                   stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    return dict(controller_source=str(target.resolve()),controller_sha256=_sha(target),
                waypoint_controller=waypoint_controller)


class FastMeasuredSimulator(_NativeSimulator):
    """Native fast controller with separate measured corpus fields, no VLA."""
    def __init__(self,config):
        super().__init__(config)
        options = config['worker_options'];path = Path(options['controller_source'])
        self._controller = load_accounted_controller(path, options['controller_sha256'])

    def _environment_config(self,task,seed,limit):
        cfg = super()._environment_config(task,seed,limit)
        cfg.skip_intermediate_renders = True
        cfg.malloc_trim_reset_interval = 1
        cfg.task_config.timestep = 1/250
        cfg.task_config.camera.head_camera_type = 'Large_D435'
        cfg.task_config.camera.wrist_camera_type = 'Large_D435'
        return cfg

    def reset(self,task,seed):
        raw,info = super().reset(task,seed)
        native = self.env.venv.envs[0].task
        install_accounted_controller(native, self._controller)
        from .robotwin_noise import _observation_hash
        info.update(public_observation_sha256=_observation_hash(raw),scene=native.get_info(),
            controller='native fast with passive execution accounting',
            controller_sha256=self.config['worker_options']['controller_sha256'])
        self.requested = 0
        return raw,info

    def step(self,actions):
        from .robotwin_noise import _capture_sparse_feedback
        if not self.active or len(actions) > self.episode_limit-self.requested:
            raise RuntimeError('Require one full chunk within the active episode allowance')
        native = self.env.venv.envs[0].task
        started = time.monotonic()
        with _capture_sparse_feedback(native) as feedback:
            obs,rewards,terms,truncs,infos = self.env.chunk_step(self.torch.from_numpy(actions[None]))
        execution = chunk_execution(feedback['native_info'],requested_targets=len(actions),
                                    timestep_seconds=float(native.scene.get_timestep()))
        terminated,truncated = bool(self.torch.as_tensor(terms).any()),bool(self.torch.as_tensor(truncs).any())
        if not execution.delivered_targets and not (terminated or truncated):
            raise RuntimeError('A nonterminal decision delivered no target')
        self.requested += len(actions);self.elapsed += execution.delivered_targets
        if self.requested >= self.episode_limit and not (terminated or truncated):
            raise RuntimeError('Native wrapper omitted the declared episode boundary')
        self.active = not (terminated or truncated or self.requested >= self.episode_limit)
        reward = float(self.torch.as_tensor(rewards).sum())
        if not np.isfinite(reward): raise RuntimeError('Nonfinite native reward')
        from .robotwin_production_process import _cpu_observation
        return _cpu_observation(obs[-1]),reward,terminated,truncated,dict(
            execution=asdict(execution),duration_seconds=execution.duration_seconds,
            wall_seconds=time.monotonic()-started,worker_pid=__import__('os').getpid(),
            measured_valid_target_mask=np.asarray(execution.valid_target_mask,dtype=bool),
            native_valid_action_mask=_cpu_tree(infos[-1].get('valid_action_mask')),
            adapter_rewards=_cpu_tree(rewards),adapter_terminations=_cpu_tree(terms),
            adapter_truncations=_cpu_tree(truncs),
            success=bool(self.torch.as_tensor(infos[-1]['episode']['success_once']).any()),
            **_cpu_tree(feedback))

class RoboTwinBatchRuntime:
    """Case-addressed base proposals and measured physical transitions.

    Call reset → infer → external residual head → step. Every inference is one
    native batch/prefix and one explicit draw per episode/chunk. A pending
    proposal cannot be reused or invalidated by another reset. The caller owns
    learner/history state, action escrows, checkpoint boundaries and raw writes.
    With fixed_native_batch=True, inference retains initialized completed
    slots while physical dispatch still selects active lanes only.
    """
    def __init__(self,motor,pool,*,seed=0):
        self.motor,self.pool,self.seed = motor,pool,seed
        self.horizon = motor.noise_shape[0]
        self._raw = [None]*pool.num_workers
        self._cases = [None]*pool.num_workers
        self._chunks = [0]*pool.num_workers
        self._active = [False]*pool.num_workers
        self._seen_cases = set()
        self._pending,self.failed,self.closed = None,False,False
        self.native_batches = 0

    @classmethod
    def create(cls,motor,*,lanes,rlinf_root,robotwin_root,output,task_episode_limits,seed=0):
        output = Path(output);output.mkdir(parents=True,exist_ok=False)
        options = accounted_controller_source(robotwin_root,output/'accounted-controller')
        pool = ProductionSimulatorPool(lanes,rlinf_root=rlinf_root,robotwin_root=robotwin_root,
            output_dir=output/'simulators',task_episode_limits=task_episode_limits,
            final_observation_only=False,task_config_name='demo_clean',
            _simulator_factory=FastMeasuredSimulator,worker_options=options)
        return cls(motor,pool,seed=seed)

    def _check(self):
        if self.closed or self.failed: raise RuntimeError('Runtime closed/failed; no automatic physical replay')
        self.motor.check_frozen()

    def _indices(self,indices):
        rows = list(range(len(self._raw))) if indices is None else list(indices)
        if (not rows or any(type(i) is not int or not 0 <= i < len(self._raw) for i in rows)
                or len(set(rows)) != len(rows)):
            raise ValueError('Require distinct valid lane indices')
        return rows

    def reset(self,tasks,seeds,*,case_ids,indices=None):
        self._check();indices = self._indices(indices)
        if self._pending is not None or any(self._active[i] for i in indices):
            raise RuntimeError('Never abandon an active episode or prepared proposal')
        cases,seeds = list(case_ids),list(seeds)
        if (len(cases) != len(indices) or len(seeds) != len(indices)
                or any(not isinstance(c,str) or not c for c in cases) or len(set(cases)) != len(cases)
                or set(cases) & self._seen_cases):
            raise ValueError('Bind a unique complete episode address and seed per selected lane')
        try:
            raw,info = self.pool.reset(tasks,seeds,indices=indices)
            for i,observation,receipt,case,seed in zip(indices,raw,info,cases,seeds,strict=True):
                if receipt['actual_seed'] != seed or receipt['episode_limit'] % self.horizon:
                    raise ValueError('Require the exact reset identity and a full-chunk episode horizon')
                self._raw[i],self._cases[i],self._chunks[i],self._active[i] = observation,case,0,True
            self._seen_cases.update(cases)
            return info
        except BaseException:
            self.failed = True;raise

    def infer(self,*,indices=None,fixed_native_batch=False):
        self._check()
        if type(fixed_native_batch) is not bool or fixed_native_batch and indices is not None:
            raise ValueError('Fixed native slots require all initialized lanes in their original order')
        indices = [i for i,v in enumerate(self._active) if v] if indices is None else self._indices(indices)
        if self._pending is not None or not indices or any(not self._active[i] for i in indices):
            raise RuntimeError('Infer exactly once for selected active observations')
        proposed = list(range(len(self._raw))) if fixed_native_batch else indices
        if any(self._raw[i] is None for i in proposed):
            raise RuntimeError('Initialize every fixed native slot before inference')
        identities = [f'{self._cases[i]}/chunk-{self._chunks[i]}' for i in proposed]
        noise = case_noise(identities,seed=self.seed,horizon=self.horizon,
                          action_dim=self.motor.noise_shape[1],device=self.motor.model.device)
        actions,features = self.motor.infer_batch(stack_public_observations([self._raw[i] for i in proposed]),noise=noise)
        self.native_batches += 1
        metadata = dict(native_batch=self.native_batches,batch_size=len(proposed),lanes=list(proposed),
            active_lanes=list(indices),fixed_native_batch=fixed_native_batch,
            decision_ids=identities,numerical_reference='native batched ODE at the recorded actual batch and row')
        self._pending = dict(indices=list(indices),command_rows=[proposed.index(i) for i in indices],
            observations=[self._raw[i] for i in proposed],
            base_actions=actions.detach().clone(),public_features=features.detach().clone(),metadata=_cpu_tree(metadata))
        return actions,features,metadata

    def step(self,commands):
        self._check()
        if self._pending is None: raise RuntimeError('No current proposal to execute')
        pending = self._pending;indices = pending['indices']
        commands = _cpu_tree(commands)
        commands = np.asarray(commands,dtype=np.float32)
        if commands.shape != (len(pending['metadata']['lanes']),self.horizon,14) or not np.isfinite(commands).all():
            raise ValueError('Require one finite full native chunk per prepared row')
        try:
            results = self.pool.step(commands[pending['command_rows']],indices=indices)
            transitions = []
            for lane,row,result in zip(indices,pending['command_rows'],results,strict=True):
                observation,reward,terminated,truncated,info = result
                measured = info['execution']
                execution = chunk_execution({'fb2_execution':{k:v for k,v in measured.items() if k!='timestep_seconds'}},
                    requested_targets=self.horizon,timestep_seconds=measured['timestep_seconds'])
                if not np.isfinite(reward) or not execution.delivered_targets and not (terminated or truncated):
                    raise ValueError('Require finite feedback and actual execution or a terminal no-op')
                transitions.append(dict(case_id=self._cases[lane],chunk_index=self._chunks[lane],
                    observation=pending['observations'][row],next_observation=observation,
                    base_actions=_cpu_tree(pending['base_actions'][row]),
                    public_features=_cpu_tree(pending['public_features'][row]),requested_actions=commands[row].copy(),
                    reward=float(reward),terminated=bool(terminated),truncated=bool(truncated),
                    execution=asdict(execution),physics_duration_seconds=execution.duration_seconds,
                    measured_valid_target_mask=np.asarray(execution.valid_target_mask,dtype=bool),
                    native_batch=dict(pending['metadata'],row=row),audit=info))
                # Transitions belong to the caller; retain a private next input.
                self._raw[lane] = _cpu_tree(observation);self._chunks[lane] += 1
                self._active[lane] = not (terminated or truncated)
            self._pending = None
            return transitions
        except BaseException:
            self.failed = True;raise

    def close(self):
        if not self.closed:
            try: self.pool.close()
            finally: self.closed = True
