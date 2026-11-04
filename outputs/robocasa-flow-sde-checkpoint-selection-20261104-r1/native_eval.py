"""Qualification of eval-only RLinf orchestration on the unchanged source runtime.

Run only inside one scheduler GPU allocation. All model computation and native
simulator construction use the source runtime; no training workers are created.
"""
import argparse
import gc
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from fluffy_bassoon.meta_rl.rlinf_native_ppo import (
    ActionJournal, sha, snapshot_rng, tensor_hash, write,
)
from fluffy_bassoon.meta_rl.rlinf_robocasa_flow_ppo import flow_logprob_parity
from fluffy_bassoon.meta_rl.rlinf_robocasa_flow_ppo_runtime import (
    FlowEnvironment, unwrapped,
)
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class SelectionEnvironment(FlowEnvironment):
    def init_worker(self):
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)
        self._flow_phase, self._flow_iteration = 'initialization', 0
        # Derived and validated from the untouched source config before the
        # explicit eval-only world/case-bank migration below.
        self._flow_contract = OmegaConf.to_container(self.cfg.selection.contract, resolve=True)
        self._flow_journal = ActionJournal(self.cfg.native_pilot.accounting_root,
            rank=self._rank, world=self._world_size,
            attempt=self.cfg.native_pilot.attempt, maximum_actions=100000)
        return EnvWorker.init_worker(self)

    def select_bank(self, phase, step, case_seed):
        self._flow_phase, self._flow_iteration = phase, int(step)
        self._flow_contract['eval_seed'] = int(case_seed)
        for wrapped in self.eval_env_list:
            unwrapped(wrapped)._flow_round = -1
        return dict(phase=phase, step=step, seeds=list(range(case_seed, case_seed + 8)))

    def release_environments(self):
        for wrapped in self.eval_env_list:
            env = unwrapped(wrapped)
            if hasattr(env, 'env'):
                env.env.close()
        return dict(charged_controls=self._flow_journal.charged())


class SelectionRollout(MultiStepRolloutWorker):
    def select_checkpoint(self, path, expected_sha, step, phase, rng_seed):
        from rlinf.models.embodiment.openpi.checkpoint import (
            _normalize_state_dict, _convert_official_openpi_keys,
        )
        from rlinf.utils.ckpt_convertor.openpi._core import as_state_dict
        path = Path(path)
        assert sha(path) == expected_sha
        state = _convert_official_openpi_keys(_normalize_state_dict(as_state_dict(
            torch.load(path, map_location='cpu', weights_only=False))), path)
        self.hf_model.load_state_dict(state, strict=True)
        del state
        gc.collect()
        self.hf_model.eval().requires_grad_(False)
        self.hf_model.set_global_step(int(step))
        self._selection_phase = phase
        self._selection_calls = 0
        # Environment constructors live in another worker; resetting the
        # rollout RNG here fixes the same Gaussian noise sequence in every arm.
        random.seed(rng_seed)
        np.random.seed(rng_seed)
        torch.manual_seed(rng_seed)
        torch.cuda.manual_seed_all(rng_seed)
        return dict(step=step, model_sha256=expected_sha, strict_load=True, rng_seed=rng_seed)

    def initial_likelihood_parity(self, fixture_path, expected_sha):
        assert sha(fixture_path) == expected_sha
        fixture = torch.load(fixture_path, map_location='cpu', weights_only=False)
        inputs = {k: v.to(self.device) for k, v in fixture['forward_inputs'].items()}
        with torch.no_grad():
            result = self.hf_model(forward_inputs=inputs, compute_logprobs=True,
                compute_values=True, compute_entropy=False, use_cache=False)
        parity = flow_logprob_parity(result['logprobs'].detach().float().cpu(),
                                    fixture['old_logprobs'])
        assert parity['passed'], parity
        return parity

    def predict(self, env_obs, mode='train'):
        assert mode == 'eval' and self.only_eval
        before = snapshot_rng()
        actions, result = super().predict(env_obs, mode=mode)
        assert tuple(actions.shape) == (8, 50, 12) and bool(torch.isfinite(actions).all())
        self._selection_calls += 1
        write(Path(self.cfg.selection.output) / 'sampler' /
              f'{self._selection_phase}-{self._selection_calls:02d}.json',
              dict(batch_size=8, chunk=50, actions_shape=list(actions.shape),
                   before_rng_sha256=tensor_hash({k: before[k] for k in ['torch', 'cuda']}),
                   after_rng_sha256=tensor_hash({k: v for k, v in snapshot_rng().items()
                                                if k in ['torch', 'cuda']}),
                   actions_sha256=tensor_hash(actions.detach().cpu())))
        return actions, result


def configuration(admission, output, deadline):
    from fluffy_bassoon.meta_rl.rlinf_robocasa_flow_ppo import flow_contract
    from rlinf.config import validate_cfg
    cfg = OmegaConf.create(json.loads(Path(admission['config']).read_text()))
    assert sha(admission['config']) == admission['config_sha256']
    contract = flow_contract(cfg)
    assert contract['task'] == 'PickPlaceDrawerToCounter'
    assert contract['horizon'] == 600 and contract['chunk'] == 50
    assert cfg.env.eval.total_num_envs == 64
    original_model = OmegaConf.merge(cfg.actor.model, cfg.rollout.model)
    # The qualified source actor is BF16; its rollout overlay uses precision
    # null, so bind BF16 explicitly rather than inherit the eval factory default.
    original_model.precision = 'bf16'
    cfg.rollout.model = original_model
    cfg.runner.only_eval, cfg.runner.task_type = True, 'embodied_eval'
    cfg.runner.resume_dir, cfg.runner.ckpt_path = None, None
    cfg.runner.logger.log_path = str(output / 'logs')
    cfg.runner.logger.experiment_name = 'checkpoint-selection-qualification'
    cfg.runner.logger.logger_backends = []
    cfg.cluster.component_placement = {'env,rollout': '0'}
    cfg.env.eval.total_num_envs, cfg.env.eval.rollout_epoch = 8, 1
    cfg.env.eval.video_cfg.save_video = False
    cfg.native_pilot.output = str(output / 'bank')
    cfg.native_pilot.accounting_root = str(output.parent / 'accounting')
    cfg.native_pilot.attempt = output.name
    cfg.native_pilot.deadline_unix = deadline
    cfg.selection = dict(contract=contract, output=str(output))
    cfg = validate_cfg(cfg)
    assert cfg.runner.only_eval and cfg.env.eval.total_num_envs == 8
    assert cfg.rollout.model.precision == 'bf16'
    assert not cfg.rollout.enable_offload
    assert not cfg.rollout.get('enable_cuda_graph', False)
    assert not cfg.rollout.get('enable_torch_compile', False)
    return cfg


def run(prepared, output, deadline):
    from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    assert torch.cuda.device_count() == 1
    admission = json.loads(prepared.read_text())
    assert admission['completed'] and admission['task'] == 'PickPlaceDrawerToCounter'
    cfg = configuration(admission, output, deadline)
    write(output / 'resolved-config.json', OmegaConf.to_container(cfg, resolve=True))
    cluster = Cluster(cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path)
    placement = HybridComponentPlacement(cfg, cluster)
    assert placement.get_world_size('env') == placement.get_world_size('rollout') == 1
    rollout = SelectionRollout.create_group(cfg).launch(cluster, name=cfg.rollout.group_name,
        placement_strategy=placement.get_strategy('rollout'))
    env = SelectionEnvironment.create_group(cfg).launch(cluster, name=cfg.env.group_name,
        placement_strategy=placement.get_strategy('env'))
    runner = EmbodiedEvalRunner(cfg, rollout, env)
    runner.init_workers()
    rows = []
    for phase, step in [('zero-a', 0), ('final', 130), ('zero-b', 0)]:
        tick = time.monotonic()
        cp = next(r for r in admission['checkpoints'] if r['step'] == step)
        load = rollout.select_checkpoint(cp['staged_path'], cp['sha256'], step,
                                        phase, 11030000).wait()
        if phase == 'zero-a':
            parity = rollout.initial_likelihood_parity(admission['fixture'],
                                        admission['fixture_sha256']).wait()
            write(output / 'initial-parity.json', parity)
            # The parity forward can consume RNG. Reseed after it, without
            # changing weights, before the first real episode.
            rollout.select_checkpoint(cp['staged_path'], cp['sha256'], step,
                                        phase, 11030000).wait()
        env.select_bank(phase, step, 10030000).wait()
        evaluation_started = time.monotonic()
        metrics = runner.evaluate()
        evaluation_seconds = time.monotonic() - evaluation_started
        episodes = output / 'bank/environment/rank-00' / f'{phase}-i{step:03d}-round00-episodes.json'
        data = json.loads(episodes.read_text())
        assert data['completed'] and len(data['rows']) == 8
        assert [r['seed'] for r in data['rows']] == list(range(10030000, 10030008))
        samples = sorted((output / 'sampler').glob(phase + '-*.json'))
        assert len(samples) == 12
        rows.append(dict(phase=phase, step=step, seconds=time.monotonic()-tick,
            evaluation_seconds=evaluation_seconds,
            evaluation_controls_per_second=4800/evaluation_seconds,
            load=load, first_success=sum(r['success_once'] for r in data['rows']),
            success_at_end=sum(r['success_at_end'] for r in data['rows']),
            episodes=str(episodes), episodes_sha256=sha(episodes),
            metrics={k: np.asarray(v).tolist() for k, v in metrics.items()}))
        write(output / 'PROGRESS.json', dict(completed=False, rows=rows, SGD=0))
        print(json.dumps(rows[-1]), flush=True)
    sampler_rows = [[json.loads(p.read_text()) for p in
        sorted((output / 'sampler').glob(phase + '-*.json'))] for phase in ['zero-a', 'final', 'zero-b']]
    assert all([r['before_rng_sha256'] for r in values] ==
               [r['before_rng_sha256'] for r in sampler_rows[0]] for values in sampler_rows)
    assert all([r['after_rng_sha256'] for r in values] ==
               [r['after_rng_sha256'] for r in sampler_rows[0]] for values in sampler_rows)
    # Equality is recorded as a diagnostic, never imposed as a requirement on
    # GPU kernels. Numeric actions remain in the native full chunk records.
    repeated_success = []
    for phase, step in [('zero-a', 0), ('zero-b', 0)]:
        p = output / 'bank/environment/rank-00' / f'{phase}-i{step:03d}-round00-episodes.json'
        repeated_success.append([r['success_once'] for r in json.loads(p.read_text())['rows']])
    released = env.release_environments().wait()
    runner.metric_logger.finish()
    journals = list((output.parent / 'accounting').glob('rank-*/*.json'))
    calls = [r for p in journals for r in json.loads(p.read_text())['calls']]
    assert calls and all(r['resolved'] for r in calls)
    controls = sum(r['executed_or_conservative'] for r in calls)
    assert controls == 14400
    write(output / 'RESULT.json', dict(completed=True, rows=rows, SGD=0,
        policy_controls=14400, all_controls=controls, released=released,
        fixed_eight_lane_batches=True, common_sampler_rng_passed=True,
        repeated_initial_policy_success_equal=repeated_success[0]==repeated_success[1],
        repeated_initial_policy_action_hashes_equal=[r['actions_sha256'] for r in sampler_rows[0]]==
                                                  [r['actions_sha256'] for r in sampler_rows[2]],
        bitwise_deterministic_kernels_required=False,
        initial_parity=parity, source_admission_sha256=sha(prepared)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    run(args.prepared, args.output, args.deadline)


if __name__ == '__main__':
    main()
