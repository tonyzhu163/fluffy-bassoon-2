"""Run-scoped Motus H32 comparison; compose/validate on allocated Linux."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PAIRED = ROOT.parent / 'robotwin-sidney-matched-baselines-20261106-r1'


def overrides(algorithm, *, optimized_setup=False):
    assert algorithm in ('ppo', 'grpo')
    assert type(optimized_setup) is bool
    files = {name: ROOT / name for name in ('MOTUS-TRAIN-CASES.json', 'MOTUS-DEV-CASES.json')}
    result = [f'native_pilot.defer_scene_setup={str(optimized_setup).lower()}',
        f'native_pilot.recycle_scene_executor={str(optimized_setup).lower()}',
        f'algorithm.adv_type={"gae" if algorithm == "ppo" else "grpo"}',
        f'algorithm.loss_type={"actor_critic" if algorithm == "ppo" else "actor"}',
        f'actor.model.add_value_head={str(algorithm == "ppo").lower()}',
        'algorithm.group_size=8', 'env.train.group_size=8', 'actor.micro_batch_size=256',
        '++algorithm.recompute_logprobs=false', '+native_pilot.recompute_openpi_logprobs=true',
        'runner.val_check_interval=1', 'env.train.enable_offload=true',
        'env.eval.enable_offload=true', 'env.eval.total_num_envs=16',
        '++env.train.skip_intermediate_renders=true', '++env.eval.skip_intermediate_renders=true',
        f'runner.logger.log_path={os.environ["TASK_LOCAL"]}/logs',
        f'runner.logger.experiment_name=motus-matched-{algorithm}',
        f'+native_pilot.comparison_algorithm={algorithm}',
        '+native_pilot.stop_after_iteration=1', '+native_pilot.canonical_compression=gzip',
        '+native_pilot.sidecar_batch_drop=[forward_inputs]',
        f'+native_pilot.passive_controller_source={os.environ["TASK_CONTROLLER"]}',
        f'+native_pilot.passive_controller_sha256={os.environ["TASK_CONTROLLER_SHA"]}',
        f'+native_pilot.checkpoint_archive={os.environ["TASK_COLD"]}/checkpoints']
    for prefix, name in (('case', 'MOTUS-TRAIN-CASES.json'), ('dev', 'MOTUS-DEV-CASES.json')):
        path = files[name]
        result += [f'+native_pilot.{prefix}_manifest_path={path}',
                   f'+native_pilot.{prefix}_manifest_sha256={hashlib.sha256(path.read_bytes()).hexdigest()}']
    return result


def validate(cfg):
    from omegaconf import OmegaConf
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import validate_recipe
    from fluffy_bassoon.rl.robotwin_noise import MOTUS_REVISION
    method = cfg.native_pilot.comparison_algorithm
    assert method in ('ppo', 'grpo')
    setup = {key: cfg.native_pilot.get(key, False) for key in
             ('defer_scene_setup', 'recycle_scene_executor')}
    assert all(type(value) is bool for value in setup.values()), setup
    expected = {
        'algorithm.adv_type': 'gae' if method == 'ppo' else 'grpo',
        'algorithm.loss_type': 'actor_critic' if method == 'ppo' else 'actor',
        'actor.model.add_value_head': method == 'ppo',
        'algorithm.group_size': 8, 'env.train.group_size': 8,
        'algorithm.recompute_logprobs': False, 'native_pilot.recompute_openpi_logprobs': True,
        'runner.val_check_interval': 1, 'algorithm.filter_rewards': False,
        'actor.micro_batch_size': 256, 'actor.model.openpi.action_horizon': 32,
        'actor.model.openpi.action_chunk': 32, 'actor.model.num_action_chunks': 32,
        'actor.model.openpi_data.adapt_to_pi': True,
        'actor.model.openpi_data.extra_delta_transform': True,
        'actor.model.openpi.torch_compile': False,
        'native_pilot.world_size': 4, 'native_pilot.initializer_revision': MOTUS_REVISION,
        'native_pilot.maximum_actions': 500000, 'native_pilot.bounded_io': True,
        'native_pilot.sampling_record': 'full', 'native_pilot.canonical_compression': 'gzip',
        'env.eval.total_num_envs': 16,
    }
    for mode in ('train', 'eval'):
        expected.update({f'env.{mode}.{field}': 192 for field in (
            'max_episode_steps', 'max_steps_per_rollout_epoch', 'task_config.step_lim')})
        expected.update({f'env.{mode}.enable_offload': True,
            f'env.{mode}.skip_intermediate_renders': True,
            f'env.{mode}.task_config.task_name': 'place_phone_stand',
            f'env.{mode}.task_config.camera.head_camera_type': 'Large_D435',
            f'env.{mode}.task_config.camera.wrist_camera_type': 'Large_D435',
            f'env.{mode}.task_config.camera.collect_wrist_camera': True})
    for key, value in expected.items():
        actual = OmegaConf.select(cfg, key)
        # oc.env world_size is a string until hardware validation; accept its
        # integer value without changing the rest of the resolved contract.
        if key == 'native_pilot.world_size': actual = int(actual)
        assert actual == value, (key, actual, value)
    assert int(cfg.native_pilot.stop_after_iteration) in (1, 2)
    assert cfg.native_pilot.passive_controller_source and len(cfg.native_pilot.passive_controller_sha256) == 64
    case_bytes = Path(cfg.native_pilot.case_manifest_path).read_bytes()
    assert hashlib.sha256(case_bytes).hexdigest() == cfg.native_pilot.case_manifest_sha256
    schedule = json.loads(case_bytes)
    assert schedule['task'] == 'place_phone_stand' and schedule['iterations'] == 2
    assert schedule['rounds_per_iteration'] == 4 and schedule['environment_group_size'] == 8
    assert len(schedule['groups']) == len({row['scene_seed'] for row in schedule['groups']}) == 256
    bank = Path(cfg.env.train.seeds_path).read_bytes()
    assert hashlib.sha256(bank).hexdigest() == schedule['train_bank_sha256']
    train = set(json.loads(bank)['place_phone_stand']['success_seeds'])
    assert {r['scene_seed'] for r in schedule['groups']} <= train
    dev_path = Path(cfg.native_pilot.dev_manifest_path)
    assert hashlib.sha256(dev_path.read_bytes()).hexdigest() == cfg.native_pilot.dev_manifest_sha256
    dev = json.loads(dev_path.read_bytes())
    eval_bank = Path(cfg.env.eval.seeds_path).read_bytes()
    assert hashlib.sha256(eval_bank).hexdigest() == dev['eval_bank_sha256']
    assert dev['seeds'] == json.loads(eval_bank)['place_phone_stand']['success_seeds'][:16]
    assert len(set(dev['seeds'])) == 16 and not set(dev['seeds']) & train
    # Reuse the original guard for every unchanged optimizer/sampler/reduction
    # field after explicitly validating Motus and paired-layout differences.
    common = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    common.algorithm.adv_type = 'gae'; common.algorithm.loss_type = 'actor_critic'
    common.actor.model.add_value_head = True; common.actor.micro_batch_size = 32
    common.actor.model.num_action_chunks = 50; common.env.eval.total_num_envs = 128
    for mode in ('train', 'eval'):
        common.env[mode].task_config.task_name = 'adjust_bottle'
        for field in ('max_episode_steps', 'max_steps_per_rollout_epoch', 'task_config.step_lim'):
            OmegaConf.update(common, f'env.{mode}.{field}', 400)
    validate_recipe(common)
    return dict(protocol='motus_native_h32', algorithm=method, source_task='place_phone_stand',
        global_batch=2048, micro_batch=256, world=4, accumulation=2, update_epochs=5,
        rounds_per_iteration=4, decisions_per_episode=6, decisions_per_iteration=6144,
        adam_updates_per_iteration=15, intended_iterations=2,
        admitted_stop_after_iteration=int(cfg.native_pilot.stop_after_iteration),
        train_manifest_sha256=cfg.native_pilot.case_manifest_sha256, dev_manifest_sha256=cfg.native_pilot.dev_manifest_sha256,
        dev_episodes=16, dev_role='source-task development panel; never held-out transfer evaluation',
        value_protocol='native K-step mean rollout values, selected-step actor values; both independently audited',
        likelihood_protocol='OpenPi shape-aware reference at native post-shuffle hook, once before all epochs',
        simulator_setup=setup,
        original_initializer_reproduction=False, scientific_source_admission=False)


if __name__ == '__main__':
    assert sys.argv[2:] in ([], ['--optimized-setup'])
    optimized_setup = bool(sys.argv[2:])
    if sys.argv[1] in ('validate','validate_schema'):
        from hydra import compose, initialize_config_dir
        rows = []
        for method in ('ppo', 'grpo'):
            with initialize_config_dir(version_base='1.1', config_dir=str(ROOT.parents[1]/'configs/native_ppo')):
                cfg = compose(config_name='motus_phone_two', overrides=overrides(method, optimized_setup=optimized_setup))
            row = validate(cfg)
            if sys.argv[1] == 'validate_schema':
                # Run the complete upstream schema/scientific checks on Linux
                # with only hardware placement replaced by the declared four
                # ranks. Actual GPU/placement validation still runs in each job.
                assert sys.platform == 'linux' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
                from unittest.mock import patch
                import rlinf.config as native
                class Placement:
                    def get_world_size(self,role):
                        assert role in ('actor','rollout','env')
                        return 4
                with patch.object(native,'Cluster',lambda *args,**kwargs: None), \
                        patch.object(native,'HybridComponentPlacement',lambda *args,**kwargs: Placement()):
                    cfg = native.validate_cfg(cfg)
                assert cfg.runner.val_check_interval == 1 and not cfg.algorithm.recompute_logprobs
                assert cfg.native_pilot.recompute_openpi_logprobs
                row['full_native_schema_checked_with_declared_four_rank_placement'] = True
            rows.append(row)
        print(json.dumps(dict(status='complete', configs=rows, hardware_validation=False,
                              model_or_simulator_constructed=False)))
    else:
        print('\n'.join(overrides(sys.argv[1], optimized_setup=optimized_setup)))
