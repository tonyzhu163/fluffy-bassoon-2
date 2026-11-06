"""One batched frozen motor and process-owned measured RoboTwin simulators.

The same API serves residual training and evaluation. Physics never crosses a
worker's owning thread; model work stays in the parent. Native batching is a
declared numerical condition, not an assertion of singleton action equivalence.
"""
import ast
from dataclasses import asdict
import hashlib
from pathlib import Path
import shutil
import subprocess
import time
from types import MethodType

import numpy as np
import torch

from .robotwin_execution import chunk_execution
from .robotwin_motor import case_noise,stack_public_observations
from .robotwin_production_process import ProductionSimulatorPool,_NativeSimulator,_cpu_tree


REFERENCE_SHA256 = '6b3f74b038509ef989913b3f6f37c74e1d0b7156d201a1bf0100f9ccb6b0d46b'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def accounted_controller_source(robotwin_root, output):
    """Prepare a job-owned method copy; never edit the external native checkout."""
    reference = Path(robotwin_root)/'envs/_base_task.py'
    if _sha(reference) != REFERENCE_SHA256:
        raise ValueError('Unqualified native controller preimage')
    output = Path(output)
    (output/'envs').mkdir(parents=True,exist_ok=False)
    target = output/'envs/_base_task.py';shutil.copyfile(reference,target)
    patch = Path(__file__).resolve().parents[3]/'patches/robotwin-dca9ec68-execution-accounting.patch'
    subprocess.run(['patch','--fuzz=0','-p1','-d',str(output),'-i',str(patch)],check=True,
                   stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    return dict(controller_source=str(target.resolve()),controller_sha256=_sha(target))


class FastMeasuredSimulator(_NativeSimulator):
    """Native fast controller with separate measured corpus fields, no VLA."""
    def __init__(self,config):
        super().__init__(config)
        options = config['worker_options'];path = Path(options['controller_source'])
        if _sha(path) != options['controller_sha256']:
            raise ValueError('Job-owned controller source changed')
        nodes = [node for node in ast.walk(ast.parse(path.read_text()))
                 if isinstance(node,ast.FunctionDef) and node.name == 'gen_sparse_reward_data']
        if len(nodes) != 1:
            raise ValueError('Require exactly one qualified native controller method')
        self._controller = compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec')

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
        original = type(native).gen_sparse_reward_data
        namespace = dict(original.__globals__);exec(self._controller,namespace)
        native.gen_sparse_reward_data = MethodType(namespace['gen_sparse_reward_data'],native)
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

    def infer(self,*,indices=None):
        self._check()
        indices = [i for i,v in enumerate(self._active) if v] if indices is None else self._indices(indices)
        if self._pending is not None or not indices or any(not self._active[i] for i in indices):
            raise RuntimeError('Infer exactly once for selected active observations')
        identities = [f'{self._cases[i]}/chunk-{self._chunks[i]}' for i in indices]
        noise = case_noise(identities,seed=self.seed,horizon=self.horizon,
                          action_dim=self.motor.noise_shape[1],device=self.motor.model.device)
        actions,features = self.motor.infer_batch(stack_public_observations([self._raw[i] for i in indices]),noise=noise)
        self.native_batches += 1
        metadata = dict(native_batch=self.native_batches,batch_size=len(indices),lanes=list(indices),
            decision_ids=identities,numerical_reference='native batched ODE at the recorded actual batch and row')
        self._pending = dict(indices=list(indices),observations=[self._raw[i] for i in indices],
            base_actions=actions.detach().clone(),public_features=features.detach().clone(),metadata=_cpu_tree(metadata))
        return actions,features,metadata

    def step(self,commands):
        self._check()
        if self._pending is None: raise RuntimeError('No current proposal to execute')
        pending = self._pending;indices = pending['indices']
        commands = _cpu_tree(commands)
        commands = np.asarray(commands,dtype=np.float32)
        if commands.shape != (len(indices),self.horizon,14) or not np.isfinite(commands).all():
            raise ValueError('Require one finite full native chunk per prepared row')
        try:
            results = self.pool.step(commands,indices=indices)
            transitions = []
            for row,(lane,result) in enumerate(zip(indices,results,strict=True)):
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
