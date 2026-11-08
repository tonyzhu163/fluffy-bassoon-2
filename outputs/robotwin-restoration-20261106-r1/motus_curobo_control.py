"""Native cuRobo expert control, paired to four preserved MPLib scenes."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import time

from motus_fresh_bank import check_initial_observation, sha, write
from motus_curobo_paths import prepare

ROOT = Path(__file__).resolve().parent


def main(contract_path):
    import torch
    from omegaconf import OmegaConf
    from fluffy_bassoon.baige_gpu import mapping, pci
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import cpu_tree, tensor_hash
    from fluffy_bassoon.rl.robotwin_noise import _install_allocated_renderer, _install_reset_retry_limit

    p = json.loads(Path(contract_path).read_bytes())
    assert p['status'] == 'admitted' and p['GPU_count'] == 1
    assert p['closed_conservative_gpu_seconds']+p['reserved_gpu_seconds'] <= p['approved_cap_gpu_seconds'] == 360000
    assert os.uname().sysname == 'Linux' and os.uname().nodename.startswith('job-')
    source = os.environ['TASK_SOURCE']; code = ROOT.parents[1]
    assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip() == source
    for name,digest in p['source_files'].items(): assert sha(code/name) == digest
    for key in ('CPU_result','backend_readback','path_result','original_configuration','reference_control'):
        assert sha(p[key]) == p[key+'_sha256']
    for key,expected in p['pinned_sources'].items():
        assert subprocess.check_output(['git','-C',os.environ[key],'rev-parse','HEAD'],text=True).strip() == expected
        assert not subprocess.check_output(['git','-C',os.environ[key],'status','--porcelain','--untracked-files=no'])
    backend = json.loads(Path(p['backend_readback']).read_bytes())
    assert backend['status'] == 'passed_CPU_extension_ABI_load_device_import_still_required'
    assert backend['curobo_commit'] == p['curobo_commit']
    assert torch.__version__ == backend['torch_version']
    for row in backend['extension_modules'].values(): assert sha(row['file']) == row['sha256']
    reference = json.loads(Path(p['reference_control']).read_bytes())
    cases = {r['seed']:r for r in reference['records'] if r['profile']=='full_state'}
    assert sorted(cases) == sorted(p['seeds']) and len(cases) == 4
    assert all(r['setup_demo_success'] and not r['play_once_success'] for r in cases.values())
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    devices = mapping(); assert len(devices) == 1
    os.environ['LYCO_BAIGE_ISOLATED_UUID'] = devices[0]['uuid']
    out = Path(os.environ['TASK_OUT']); out.mkdir()
    report = dict(status='running',source_commit=source,records=[],planner_calls=[],path_repairs=[],
        reference_control_sha256=p['reference_control_sha256'],VLA_calls=0,optimizer_updates=0,
        native_seed_inventory_unchanged=True,collision_configuration='unmodified_native_table_only_curobo',
        fresh_bank_published=False,learning_effectiveness_tested=False,cleanup_complete=False)
    env = None; original_init = None; originals = {}; active = {}
    try:
        # MotionGen creates a default CUDA Pose at import time; imports therefore
        # occur only after actual allocation probing and logical-device selection.
        from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv
        from robotwin.envs.vector_env import SubEnv
        from envs.robot.planner import CuroboPlanner
        assert CuroboPlanner is not None
        report['full_native_API_import_passed'] = True
        write(out/'RESULT.json',report)
        renderer = _install_allocated_renderer(out/'renderers')
        _install_reset_retry_limit(defer_scene=True)
        SubEnv._fb2_trim_after_scene_close = False
        original_init = CuroboPlanner.__init__

        def runtime_init(self,*args,**kwargs):
            assert len(args) == 3 and set(kwargs) == {'yml_path'}
            native = Path(kwargs['yml_path'])
            assert native.name in ('curobo_left.yml','curobo_right.yml')
            row = prepare(native,os.environ['ROBOTWIN_PATH'],Path(os.environ['TASK_LOCAL'])/'native-config'/native.name)
            path_evidence = json.loads(Path(p['path_result']).read_bytes())
            expected = next(r for r in path_evidence['rows'] if r['native_yaml']==str(native))
            for key in ('native_yaml_sha256','actual_urdf_sha256','only_urdf_path_changed'):
                assert row[key] == expected[key]
            report['path_repairs'].append(row); write(out/'RESULT.json',report)
            original_init(self,*args,yml_path=row['runtime_yaml'])
            for motor in (self.motion_gen,self.motion_gen_batch):
                assert torch.device(motor.tensor_args.device) == torch.device('cuda:0')

        CuroboPlanner.__init__ = runtime_init
        for name in ('plan_path','plan_batch'):
            original = getattr(CuroboPlanner,name); originals[name] = original
            def tracked(self,*args,_original=original,_name=name,**kwargs):
                row = dict(seed=active.get('seed'),method=_name,started=time.time())
                report['planner_calls'].append(row); write(out/'RESULT.json',report)
                result = _original(self,*args,**kwargs)
                statuses = result['status']
                row.update(status=statuses.tolist() if hasattr(statuses,'tolist') else str(statuses),seconds=time.time()-row['started'])
                write(out/'RESULT.json',report)
                return result
            setattr(CuroboPlanner,name,tracked)
        config = OmegaConf.create(json.loads(Path(p['original_configuration']).read_bytes())['env']['eval'])
        assert config.task_config.planner_backend == 'mplib' and config.max_episode_steps == 192
        config.task_config.planner_backend = 'curobo'; config.group_size = 1
        assert config.task_config.task_name == 'place_phone_stand'
        env = RoboTwinEnv(config,1,seed_offset=0,total_num_processes=1,worker_info=None)
        inventory = env.success_seeds.clone()
        for seed in p['seeds']:
            assert time.time()+150 < int(os.environ['TASK_DEADLINE'])
            active['seed'] = seed
            sub = env.venv.envs[0]
            obs,_ = env.reset(env_seeds=[seed])
            assert int(sub.task.ep_num) == int(sub.env_seed) == seed
            assert not sub.task.robot.communication_flag
            assert isinstance(sub.task.robot.left_planner,CuroboPlanner) and isinstance(sub.task.robot.right_planner,CuroboPlanner)
            assert pci(sub.task.scene.render_system.device.pci_string.removeprefix('pci:')) == renderer
            check_initial_observation(obs)
            initial = tensor_hash(cpu_tree({k:obs[k] for k in ('states','main_images','wrist_images')}))
            assert initial == cases[seed]['initial_observation_sha256'], 'Physical pairing differs from preserved control'
            initial_state = obs['states'].tolist()
            sub.close(clear_cache=False); gc.collect(); torch.cuda.empty_cache()
            result = env.check_seeds([seed])[0]
            assert int(sub.task.ep_num) == seed and torch.equal(inventory,env.success_seeds)
            report['records'].append(dict(seed=seed,physical_seed=int(sub.task.ep_num),
                initial_observation_sha256=initial,initial_state=initial_state,paired_with_preserved_control=True,**result))
            write(out/'RESULT.json',report)
            print('NATIVE_CUROBO_EXPERT',seed,result,flush=True)
        passes = sum(r['play_once_success'] for r in report['records'])
        report.update(expert_successes=passes,initial_observations_paired=True,
            status='qualified_four_scene_expert_control' if passes==4 and report['planner_calls'] else 'failed_control')
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error)); raise
    finally:
        if env is not None:
            env.venv.env_thread_pool.shutdown(wait=True); torch.cuda.synchronize(); env.offload()
        if original_init is not None:
            CuroboPlanner.__init__ = original_init
            for name,original in originals.items(): setattr(CuroboPlanner,name,original)
        gc.collect(); torch.cuda.empty_cache()
        report['cleanup_complete'] = True
        write(out/'RESULT.json',report)
    print('FINAL_NATIVE_CUROBO',report['status'],report.get('expert_successes'),sha(out/'RESULT.json'),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract',type=Path,required=True)
    main(parser.parse_args().contract)
