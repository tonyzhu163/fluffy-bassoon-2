"""Paired native/full-state expert checks on four predeclared old EVAL scenes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from motus_fresh_bank import check_initial_observation, sha, write

ROOT = Path(__file__).resolve().parent


def main(contract_path):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from fluffy_bassoon.baige_gpu import mapping, pci
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import cpu_tree, tensor_hash
    from fluffy_bassoon.rl.robotwin_noise import _install_allocated_renderer, _install_reset_retry_limit
    from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv
    from robotwin.envs.vector_env import SubEnv
    from envs.robot.planner import MplibPlanner, MplibWrapperPlanner
    from motus_planner_full_state import install, install_sync, full_qpos

    contract = json.loads(Path(contract_path).read_bytes())
    assert contract['status'] == 'admitted' and contract['GPU_count'] == 1
    assert contract['closed_conservative_gpu_seconds']+contract['reserved_gpu_seconds'] <= 360000
    assert os.uname().sysname == 'Linux' and os.uname().nodename.startswith('job-')
    source = os.environ['TASK_SOURCE']; code = ROOT.parents[1]
    assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip() == source
    for name, digest in contract['source_files'].items(): assert sha(code/name) == digest
    assert sha(contract['CPU_result']) == contract['CPU_sha256']
    assert sha(contract['original_configuration']) == contract['original_configuration_sha256']
    for key, expected in contract['pinned_sources'].items():
        assert subprocess.check_output(['git','-C',os.environ[key],'rev-parse','HEAD'],text=True).strip() == expected
        assert not subprocess.check_output(['git','-C',os.environ[key],'status','--porcelain','--untracked-files=no'])
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    devices = mapping(); assert len(devices) == 1
    os.environ['LYCO_BAIGE_ISOLATED_UUID'] = devices[0]['uuid']
    out = Path(os.environ['TASK_OUT']); out.mkdir()
    renderer = _install_allocated_renderer(out/'renderers')
    _install_reset_retry_limit(defer_scene=True)
    SubEnv._fb2_trim_after_scene_close = False
    config = OmegaConf.create(json.loads(Path(contract['original_configuration']).read_bytes())['env']['eval'])
    config.group_size = 1
    assert config.task_config.task_name == 'place_phone_stand' and config.task_config.planner_backend == 'mplib'
    assert config.max_episode_steps == 192
    profiles = contract['profiles']
    assert profiles in (['native','full_state'], ['full_state','full_state_sync'])
    original = MplibPlanner.plan_path
    report = dict(status='running', source_commit=source, records=[], planner_calls=[],
        model_calls=0, optimizer_updates=0, native_seed_inventory_unchanged=True,
        collision_checks_retained=True, cleanup_complete=False)
    active = {}; env = None

    def tracked(self, now_qpos, *args, **kwargs):
        value = np.asarray(now_qpos).copy()
        sub = active['sub']; robot = sub.task.robot
        entity = robot.left_entity if self is robot.left_planner else robot.right_entity
        measured = np.asarray(entity.get_qpos()).copy()
        internal = np.asarray(self.planner.robot.get_qpos()).copy()
        padded = self.planner.pad_move_group_qpos(value.copy())
        expected = full_qpos(self, measured)
        row = dict(profile=active['profile'], seed=active['seed'], input=value.tolist(),
            measured=measured.tolist(), internal=internal.tolist(), padded=padded.tolist(),
            all_joints=list(self.all_joints), active_joints=list(self.active_joints_name),
            padded_matches_measured_rounded_arm=bool(np.array_equal(padded, expected)))
        if active['profile'].startswith('full_state'):
            assert value.shape == expected.shape
            np.testing.assert_array_equal(value, expected)
            np.testing.assert_array_equal(padded, expected)
        report['planner_calls'].append(row)
        write(out/'RESULT.json', report)
        result = original(self, now_qpos, *args, **kwargs)
        row['status'] = str(result['status'])
        write(out/'RESULT.json', report)
        return result

    MplibPlanner.plan_path = tracked
    try:
        for profile in profiles:
            if profile == 'full_state': install()
            elif profile == 'full_state_sync': install_sync()
            env = RoboTwinEnv(config,1,seed_offset=0,total_num_processes=1,worker_info=None)
            inventory = env.success_seeds.clone()
            for seed in contract['seeds']:
                assert time.time()+100 < int(os.environ['TASK_DEADLINE'])
                sub = env.venv.envs[0]
                active.update(profile=profile,seed=seed,sub=sub)
                obs,_ = env.reset(env_seeds=[seed])
                assert int(sub.task.ep_num) == int(sub.env_seed) == seed
                assert pci(sub.task.scene.render_system.device.pci_string.removeprefix('pci:')) == renderer
                check_initial_observation(obs)
                initial = tensor_hash(cpu_tree({k:obs[k] for k in ('states','main_images','wrist_images')}))
                initial_state = obs['states'].tolist()
                sub.close(clear_cache=False)
                result = env.check_seeds([seed])[0]
                assert int(sub.task.ep_num) == seed
                assert torch.equal(inventory,env.success_seeds)
                report['records'].append(dict(profile=profile,seed=seed,initial_observation_sha256=initial,
                    initial_state=initial_state,physical_seed=int(sub.task.ep_num),**result))
                write(out/'RESULT.json',report)
                print('CONTROL',profile,seed,result,flush=True)
            env.venv.env_thread_pool.shutdown(wait=True); torch.cuda.synchronize(); env.offload(); env=None
        native = [r for r in report['records'] if r['profile']==profiles[0]]
        fixed = [r for r in report['records'] if r['profile']==profiles[1]]
        assert len(native)==len(fixed)==4
        paired = all(a['seed']==b['seed'] and a['initial_observation_sha256']==b['initial_observation_sha256'] for a,b in zip(native,fixed))
        passes = {p:sum(r['play_once_success'] for r in report['records'] if r['profile']==p) for p in profiles}
        full_calls = [r for r in report['planner_calls'] if r['profile']==profiles[1]]
        report.update(initial_observations_paired=paired,expert_successes=passes,
            full_state_calls=len(full_calls),
            status='qualified_control' if paired and passes[profiles[1]]>0 and full_calls else 'failed_control')
        # The positive control only admits further scene investigation, not a bank or learning claim.
        report['fresh_bank_published']=False
        report['learning_effectiveness_tested']=False
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error)); raise
    finally:
        if env is not None:
            env.venv.env_thread_pool.shutdown(wait=True); torch.cuda.synchronize(); env.offload()
        MplibPlanner.plan_path = original
        report['cleanup_complete']=True
        write(out/'RESULT.json',report)
    print('FINAL_CONTROL',report['status'],report.get('expert_successes'),sha(out/'RESULT.json'),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract',type=Path,default=ROOT/'MOTUS-PLANNER-CONTROL01-CONTRACT.json')
    main(parser.parse_args().contract)
