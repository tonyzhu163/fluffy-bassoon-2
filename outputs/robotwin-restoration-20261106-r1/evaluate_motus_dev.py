"""Native B4/rank DEV of the recovered15-update Motus policy; zero SGD.

Uses the same worker sampling, measured controller, trace writer and metric
aggregation as training's DEV. Actor workers and training scenes are absent.
"""
import gc
import json
import os
from pathlib import Path
import time
from types import MethodType

from omegaconf import OmegaConf
import ray
import torch

from motus_teacher_entry import ROOT, MotusEnvironment, MotusRollout, worker_setup
from motus_configuration import validate
from recovery_signature import digest
from fluffy_bassoon.meta_rl import rlinf_native_ppo_runtime as runtime
from fluffy_bassoon.meta_rl.rlinf_native_ppo import json_metrics, restore_rng, sha, tensor_hash, write
from fluffy_bassoon.meta_rl.tensor_archive import read_tensor
from rlinf.config import validate_cfg
from rlinf.data.schema.embodied_types import EnvOutput
from rlinf.scheduler import Channel, Cluster
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.placement import HybridComponentPlacement


BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')
OLD = BASE/'motusppo04'


def terms():
    plan = json.loads((ROOT/'MOTUSDEV01-CONTRACT.json').read_bytes())
    assert plan['status'] in ('prepared', 'admitted')
    assert plan['closed_conservative_gpu_seconds'] + plan['reserved_gpu_seconds'] <= 180000
    path = BASE/'cpudev-inputs02/RESULT.json'
    assert sha(path) == plan['baseline_inputs_sha256']
    inputs = json.loads(path.read_bytes())
    assert inputs['status'] == 'passed' and len(inputs['rows']) == 4
    return plan, inputs


def baseline(rank):
    _, inputs = terms()
    row = next(r for r in inputs['rows'] if r['rank'] == rank)
    for key in ('sampling', 'initial_state', 'sidecar'):
        assert sha(row[key]) == row[key+'_sha256']
    saved = read_tensor(row['sampling'])
    initial = torch.load(row['initial_state'], map_location='cpu', weights_only=False)
    public = EnvOutput(obs=initial['initial_observation']).to_dict()['obs']
    assert tensor_hash(public) == saved['public_observation_sha256'] == row['rollout_public_sha256']
    return row, saved, public


def configuration(output):
    original = json.loads((OLD/'state/resolved-native-config.json').read_bytes())
    cfg = OmegaConf.create(original)
    cfg.native_pilot.case_manifest_path = str(ROOT/'MOTUS-TRAIN-CASES.json')
    cfg.native_pilot.dev_manifest_path = str(ROOT/'MOTUS-DEV-CASES.json')
    validate(cfg)  # Qualify the original numerical recipe before eval-only wiring.
    effective = OmegaConf.merge(cfg.actor.model, cfg.rollout.model)
    cfg.runner.only_eval = True
    cfg.runner.task_type = 'embodied_eval'
    cfg.runner.ckpt_path = str(Path(os.environ['TASK_LOCAL'])/'full_weights.pt')
    cfg.runner.logger.log_path = str(Path(os.environ['TASK_LOCAL'])/'logs')
    cfg.runner.logger.experiment_name = 'motus-dev01'
    cfg.runner.per_worker_log_path = str(output/'workers')
    cfg.rollout.model = effective
    cfg.native_pilot.output = str(output)
    cfg.native_pilot.deadline_unix = int(os.environ['TASK_DEADLINE'])
    cfg.native_pilot.accounting_root = str(output/'action-accounting')
    cfg.native_pilot.attempt = 'motus-dev01'
    cfg.native_pilot.maximum_actions = 4096
    cfg.native_pilot.passive_controller_source = os.environ['TASK_CONTROLLER']
    cfg.native_pilot.passive_controller_sha256 = os.environ['TASK_CONTROLLER_SHA']
    return cfg, original, OmegaConf.to_container(effective, resolve=True)


def validate_evaluation(cfg, original, effective):
    assert cfg.runner.only_eval and cfg.runner.task_type == 'embodied_eval'
    for key in ('actor', 'algorithm'):
        assert OmegaConf.to_container(cfg[key], resolve=True) == original[key], key
    assert OmegaConf.to_container(cfg.env.eval, resolve=True) == original['env']['eval']
    assert OmegaConf.to_container(cfg.rollout.model, resolve=True) == effective
    assert cfg.env.eval.total_num_envs == 16 and cfg.rollout.pipeline_stage_num == 1
    assert cfg.env.eval.auto_reset and cfg.env.eval.ignore_terminations
    assert cfg.env.eval.max_episode_steps == cfg.env.eval.max_steps_per_rollout_epoch == 192
    assert cfg.rollout.model.num_action_chunks == 32 and cfg.native_pilot.bounded_io
    assert cfg.native_pilot.maximum_actions == 4096


class EvaluationRollout(MotusRollout):
    def prepare_baseline(self):
        """Same saved B4 inputs/RNG must reproduce the original SFT first actions."""
        assert self.only_eval and not self.enable_train and torch.cuda.device_count() == 1
        row, saved, public = baseline(self._rank)
        if self.enable_offload:
            self.reload_model()
        self.hf_model.eval().requires_grad_(False)
        import rlinf.models.embodiment.openpi as factory
        base = factory.get_model(self.cfg.rollout.model).to(self.device).eval().requires_grad_(False)
        restore_rng(saved['rng_before'])
        with torch.no_grad():
            actions, _ = base.predict_action_batch(public, mode='eval')
        expected = saved['forward_sampling_fields']['action'].reshape(4, 32, 14).to(actions.device)
        error = float((actions.float()-expected.float()).abs().max())
        assert torch.isfinite(actions).all() and error <= 1e-5, error
        del base, actions, expected
        gc.collect()
        torch.cuda.empty_cache()
        self._evaluation_frozen = digest(dict(self.hf_model.named_parameters()))
        report = dict(rank=self._rank, passed=True, baseline_batch=4,
            initial_public_sha256=row['initial_public_sha256'],
            rollout_public_sha256=row['rollout_public_sha256'],
            first_sft_action_max_abs_error=error, static_saved_input_check=True,
            new_simulator_controls=0, frozen_updated_policy_sha256=self._evaluation_frozen)
        write(Path(self.cfg.native_pilot.output)/'baseline-calibration'/f'rank-{self._rank:02d}.json', report)
        self._baseline_rng = saved['rng_before']
        restore_rng(self._baseline_rng)
        return report

    def restore_baseline_rng(self):
        restore_rng(self._baseline_rng)
        return dict(rank=self._rank, rng_sha256=digest(self._baseline_rng))

    def finish_evaluation(self):
        self._flush_pilot_io()
        assert self._pilot_prediction_ordinal == 6
        assert not any(p.requires_grad or p.grad is not None for p in self.hf_model.parameters())
        after = digest(dict(self.hf_model.named_parameters()))
        assert after == self._evaluation_frozen
        report = dict(rank=self._rank, predictions=6, optimizer_updates=0,
                      frozen_parameters_unchanged=True, parameters_sha256=after)
        write(Path(self.cfg.native_pilot.output)/'frozen'/f'rank-{self._rank:02d}.json', report)
        return report


class EvaluationEnvironment(runtime.PilotEnvironment):
    def init_worker(self):
        from fluffy_bassoon.rl.robotwin_noise import _install_allocated_renderer, _install_reset_retry_limit
        worker_setup(self, 'env')
        root = Path(self.cfg.native_pilot.output)
        _install_allocated_renderer(root/'renderers'/f'rank-{self._rank:02d}')
        _install_reset_retry_limit()  # Qualified default; deferred setup remains off.
        from robotwin.envs.vector_env import SubEnv
        SubEnv._fb2_trim_after_scene_close = False
        result = super().init_worker()
        assert self.only_eval and not self.enable_train and not self.env_list
        assert len(self.eval_env_list) == 1
        physical = runtime.base_env(self.eval_env_list[0])
        MotusEnvironment._install_bindings(self, physical, root)
        row, _, _ = baseline(self._rank)
        seeds = json.loads((ROOT/'MOTUS-DEV-CASES.json').read_bytes())['seeds'][self._rank*4:(self._rank+1)*4]
        assert seeds == row['scene_seeds'] and len(seeds) == physical.num_envs == 4
        original = physical.reset
        self._initial_matched = False

        def reset(env, env_idx=None, env_seeds=None):
            indices = list(range(env.num_envs)) if env_idx is None else list(env_idx)
            selected = [seeds[int(index)] for index in indices]
            env.reset_state_ids[indices] = torch.tensor(selected, dtype=torch.long,
                                                       device=env.reset_state_ids.device)
            value = original(env_idx=env_idx, env_seeds=selected)
            if env_idx is None and not self._initial_matched:
                actual = tensor_hash(value[0])
                assert actual == row['initial_public_sha256'], 'Initial DEV pixels/state/prompt differ'
                write(root/'paired-initial'/f'rank-{self._rank:02d}.json',
                      dict(rank=self._rank, passed=True, scene_seeds=seeds,
                           actual_public_sha256=actual, baseline=row['initial_state']))
                self._initial_matched = True
            return value

        physical.reset = MethodType(reset, physical)
        return result

    def finish_evaluation(self):
        self._flush_pilot_io()
        assert self._initial_matched and self._pilot_trace_ordinal == 6
        assert all(row['resolved'] for row in self._pilot_journal.rows)
        return dict(rank=self._rank, initial_public_matched=True, traces=6,
                    charged_actions=self._pilot_journal.charged(), native_evaluation_offload=True)


def main():
    plan, _ = terms()
    assert plan['status'] == 'admitted' and os.uname().nodename.startswith('job-')
    root = Path(os.environ['TASK_OUT'])/'state'
    root.mkdir()
    cfg, original, effective = configuration(root)
    cfg = validate_cfg(cfg)
    validate_evaluation(cfg, original, effective)
    write(root/'resolved-native-config.json', OmegaConf.to_container(cfg, resolve=True))
    report = dict(status='starting', source_commit=os.environ['TASK_SOURCE'],
                  producing_policy_iteration=1, native_optimizer_steps_in_weights=15,
                  optimizer_updates_during_eval=0, baseline_policy_iteration=0)
    write(root/'RESULT.json', report)
    cluster = Cluster(cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path)
    placement = HybridComponentPlacement(cfg, cluster)
    rollout = EvaluationRollout.create_group(cfg).launch(cluster, name=cfg.rollout.group_name,
        placement_strategy=placement.get_strategy('rollout'))
    env = EvaluationEnvironment.create_group(cfg).launch(cluster, name=cfg.env.group_name,
        placement_strategy=placement.get_strategy('env'))
    started = time.monotonic()
    try:
        rollout.init_worker().wait()
        report['baseline_calibration'] = rollout.prepare_baseline().wait()
        write(root/'RESULT.json', report)
        env.init_worker().wait()
        for group in (env, rollout):
            group.set_global_step(1).wait()
        env.pilot_phase('dev-0001', policy_iteration=1).wait()
        rollout.pilot_evaluation_iteration(1).wait()
        rollout.restore_baseline_rng().wait()
        # Same channel calls/order and aggregation as EmbodiedRunner.evaluate.
        env_channel, rollout_channel = Channel.create('Env'), Channel.create('Rollout')
        env_handle = env.evaluate(input_channel=env_channel, rollout_channel=rollout_channel)
        rollout_handle = rollout.evaluate(input_channel=rollout_channel, output_channel=env_channel)
        results = env_handle.wait()
        rollout_handle.wait()
        metrics = compute_evaluate_metrics([r for r in results if r is not None])
        assert int(metrics['num_trajectories']) == 16
        report.update(metrics=json_metrics(metrics), environments=env.finish_evaluation().wait(),
                      frozen_rollouts=rollout.finish_evaluation().wait())
        assert sum(r['charged_actions'] for r in report['environments']) <= 4096
        report.update(status='complete', episodes=16, scientific_source_admission=False,
                      fixed_dev_panel=True, held_out_transfer_evaluation=False,
                      shared_native_train_eval_engineering=True)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['seconds'] = time.monotonic()-started
        write(root/'RESULT.json', report)
        env._close()
        rollout._close()
        ray.shutdown()


if __name__ == '__main__':
    main()
