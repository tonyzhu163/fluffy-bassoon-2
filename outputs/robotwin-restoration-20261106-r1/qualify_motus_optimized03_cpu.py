"""Check the exact next-run launch bodies on allocated Linux CPU.

Budget admission is tested with a temporary fixture only. The checked-in
contract stays prepared until qualification and live admission finish.
"""
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')


def main():
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import rlinf.config as native
    import motus_configuration as configuration
    import motus_optimized_entry as entry
    from fluffy_bassoon.rl.robotwin_batch import accounted_controller_source
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, write

    assert sys.platform == 'linux' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not torch.cuda.is_initialized()
    source = os.environ['TASK_SOURCE']
    code = ROOT.parents[1]
    assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip() == source
    assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain','--untracked-files=no'])
    out = BASE/'cpumotusoptimized03'
    out.mkdir()
    report = dict(status='checking',source_commit=source,gpu_seconds=0,
                  native_motus_or_simulator_constructed=False)
    write(out/'RESULT.json',report)
    try:
        plan = json.loads((ROOT/'MOTUSOPTIMIZED03-CONTRACT.json').read_bytes())
        assert plan['status'] == 'prepared' and plan['budget_extension_approved']
        for key,name in [('entry_sha256','motus_optimized_entry.py'),('launcher_sha256','train_motus_optimized03.sh')]:
            assert sha(ROOT/name) == plan[key]
        for name,expected in plan['source_files'].items():
            assert sha(code/name) == expected
        launcher = (ROOT/'train_motus_optimized03.sh').read_text()
        blocks = re.findall(r'<<\x27PY\x27\n(.*?)\nPY',launcher,re.S)
        assert len(blocks) == 4
        for block in blocks:
            ast.parse(block)
        subprocess.run(['bash','-n',str(ROOT/'train_motus_optimized03.sh')],check=True)
        barrier = blocks[0]
        original_contract = ROOT/'MOTUSOPTIMIZED03-CONTRACT.json'
        rejected = subprocess.run([sys.executable,'-c',barrier,str(original_contract)],capture_output=True,text=True)
        assert rejected.returncode != 0
        approved_fixture = dict(plan,status='admitted',budget_extension_approved=True,
                                approved_cap_gpu_seconds=plan['requested_cap_gpu_seconds'])
        fixture = out/'hypothetical-approved-contract.json'
        write(fixture,approved_fixture)
        subprocess.run([sys.executable,'-c',barrier,str(fixture)],check=True)
        # This fixture does not alter or admit the source contract.
        assert json.loads(original_contract.read_bytes()) == plan
        report['budget_barrier_rejects_unapproved_source'] = True

        local = Path('/tmp/rtmo3-cpu')
        os.environ.update(TASK_LOCAL=str(local),TASK_COLD=str(out/'stage-cold'),TASK_OUT=str(out),
            NATIVE_PPO_MODEL='/mowenjie/fluffy-bassoon-2/cache/robotwin-ppo-pilot-20261105-r1-restore02/pi05-motus-effc6e1a-r1',
            NATIVE_PPO_INITIALIZER_REVISION='effc6e1a99bd4d0edf9447e8ba313def67649cfe',
            NATIVE_PPO_OUTPUT=str(out/'stage-cold/state'),NATIVE_PPO_ACCOUNTING_ROOT=str(out/'action-accounting'),
            NATIVE_PPO_DEADLINE=str(int(time.time())+9000),NATIVE_PPO_ATTEMPT='ppo-optimized03')
        # Execute the launcher's unmodified model/source/storage admission body.
        subprocess.run([sys.executable,'-c',blocks[1],str(ROOT)],check=True)
        controller = accounted_controller_source(os.environ['ROBOTWIN_PATH'],local/'controller')
        os.environ.update(TASK_CONTROLLER=controller['controller_source'],
                          TASK_CONTROLLER_SHA=controller['controller_sha256'])
        with initialize_config_dir(version_base='1.1',config_dir=str(code/'configs/native_ppo')):
            cfg = compose(config_name='motus_phone_two',overrides=[
                *configuration.overrides('ppo',optimized_setup=True),'native_pilot.stop_after_iteration=2'])
        validated = configuration.validate(cfg)

        class Placement:
            def get_world_size(self,role):
                assert role in ('actor','env','rollout')
                return 8

        with patch.object(native,'Cluster',lambda *args,**kwargs:None), \
                patch.object(native,'HybridComponentPlacement',lambda *args,**kwargs:Placement()):
            cfg = native.validate_cfg(cfg)
        assert cfg.native_pilot.stop_after_iteration == 2
        assert cfg.native_pilot.defer_scene_setup and cfg.native_pilot.recycle_scene_executor
        assert validated['adam_updates_per_iteration'] == 15 and cfg.runner.max_steps == 2
        assert validated['world']==8 and validated['accumulation']==1
        assert validated['native_train_batch']==32 and validated['native_dev_batch']==2
        from case_schedule import CaseSchedule
        schedule=CaseSchedule(ROOT/'MOTUS-TRAIN-CASES.json',cfg.native_pilot.case_manifest_sha256)
        for ordinal in range(8):
            old=[row['scene_seed'] for rank in range(4) for row in schedule.rows(ordinal,rank,4)]
            new=[row['scene_seed'] for rank in range(8) for row in schedule.rows(ordinal,rank,8)]
            assert old==new and len(set(new))==32
            assert all(len(schedule.rows(ordinal,rank,8))==4 for rank in range(8))
        report['all256_scene_groups_preserved_in_order_across4_to8_ranks']=True
        report['world']=8;report['native_train_batch']=32;report['native_dev_batch']=2
        report['gradient_accumulation']=1
        write(out/'resolved-config.json',OmegaConf.to_container(cfg,resolve=True))
        with initialize_config_dir(version_base='1.1',config_dir=str(code/'configs/native_ppo')):
            grpo=compose(config_name='motus_phone_two',overrides=[*configuration.overrides('grpo',optimized_setup=True),'native_pilot.stop_after_iteration=2'])
        assert configuration.validate(grpo)['world']==8
        with patch.object(native,'Cluster',lambda *args,**kwargs:None), patch.object(native,'HybridComponentPlacement',lambda *args,**kwargs:Placement()):
            grpo=native.validate_cfg(grpo)
        assert grpo.actor.global_batch_size==cfg.actor.global_batch_size==2048
        assert grpo.algorithm.update_epoch==cfg.algorithm.update_epoch==5
        report['full_native_schema_checked_with_declared_eight_rank_placement'] = True
        report['ppo_and_grpo_configs_qualified']=True
        report['exact_launcher_admission_body_passed'] = True
        report['model_norm_and_runtime_source_hashes_verified'] = True

        import runpy, threading, signal
        import fluffy_bassoon.baige_gpu as gpu_mapping
        from unittest.mock import patch as fixture_patch
        fixture_root=local/'observer-archive-fixture';fixture_root.mkdir()
        fixture_live=local/'observer-live-fixture.json'
        release_archive=threading.Event();local_samples=[]
        def fixture_write(path,value):
            if Path(path)==fixture_root/'resources.json':
                assert release_archive.wait(10), 'Archive blocked local safety progress'
                assert len(local_samples)>=3
            else:
                local_samples.append(value['samples'])
                if value['samples']>=3 and not release_archive.is_set():
                    release_archive.set();os.kill(os.getpid(),signal.SIGTERM)
            write(path,value)
        old_argv=sys.argv[:];old_uname=os.uname()
        fake_uname=os.uname_result((old_uname.sysname,'job-cpu-fixture',old_uname.release,old_uname.version,old_uname.machine))
        os.environ['TASK_RESOURCE_LIVE']=str(fixture_live)
        sys.argv=[str(ROOT/'observe_motus_resources.py'),str(fixture_root),'8']
        try:
            with fixture_patch.object(gpu_mapping,'mapping',lambda:[dict(uuid=f'fixture-{i}') for i in range(8)]), \
                 fixture_patch.object(os,'uname',lambda:fake_uname), \
                 fixture_patch.object(subprocess,'check_output',lambda *a,**k:'\n'.join(f'fixture-{i},97880,80000,17880,0' for i in range(8))), \
                 fixture_patch('fluffy_bassoon.meta_rl.rlinf_native_ppo.write',fixture_write), \
                 fixture_patch.object(time,'sleep',lambda unused:None):
                runpy.run_path(str(ROOT/'observe_motus_resources.py'),run_name='__main__')
        finally:sys.argv=old_argv
        observed=json.loads(fixture_live.read_bytes())
        assert observed['samples']==3 and not observed['archive_writer_still_active_at_shutdown']
        assert json.loads((fixture_root/'resources.json').read_bytes())['samples']==3
        report['live_sensor_advances_while_archive_writer_blocked_fixture']=True
        report['fixture_nvml_is_not_gpu_measurement']=True
        os.environ['TASK_RESOURCE_LIVE']=str(local/'resources-live.json')
        healthy = dict(at_unix=time.time(),gpus={str(i):{} for i in range(8)},samples=1,
                       sampled_reserve_passed=True,host_peak_bytes=1,host_limit_bytes=10)
        worker = SimpleNamespace(pilot_role='actor',_rank=0,cfg=cfg)
        write(local/'resources-live.json',healthy)
        entry.require_reserve(worker)
        for change in (dict(sampled_reserve_passed=False),dict(at_unix=time.time()-60),
                       dict(host_peak_bytes=9),dict(gpus={'one':{}})):
            write(local/'resources-live.json',dict(healthy,**change))
            try:
                entry.require_reserve(worker)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Capacity guard admitted an invalid reserve fixture')
        assert json.loads(original_contract.read_bytes()) == plan
        assert not torch.cuda.is_initialized()
        report.update(status='passed',cuda_initialized=False,capacity_guard_fixture_cases=5,
            fixture_reports_are_not_gpu_measurements=True,hardware_validation=False,
            two_native_iterations_configured=True,budget_extension_approved=True,
            live_allocation_and_storage_recheck_required=True)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        write(out/'RESULT.json',report)


if __name__ == '__main__':
    main()
