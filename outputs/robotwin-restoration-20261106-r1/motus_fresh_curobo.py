"""Install the GPU-qualified native expert backend for a policy-free scan."""
import json
import os
from pathlib import Path

from motus_curobo_paths import prepare
from motus_fresh_bank import sha,write


def install(contract,report,output):
    import torch
    from envs.robot.planner import CuroboPlanner
    assert CuroboPlanner is not None and torch.cuda.device_count()==1
    proof=Path(contract['expert_control_readback'])
    assert sha(proof)==contract['expert_control_readback_sha256']
    qualified=json.loads(proof.read_bytes())
    assert qualified['status']=='passed_independent_four_scene_native_expert_control'
    assert qualified['cuRobo_expert_successes']==4 and qualified['initial_observations_paired']
    assert qualified['native_collision_configuration_unchanged']
    assert qualified['numerical_configuration_unchanged'] and qualified['cleanup_complete']
    path=Path(contract['path_result']);assert sha(path)==contract['path_result_sha256']
    evidence=json.loads(path.read_bytes());expected={x['native_yaml']:x for x in evidence['rows']}
    for key in ('referenced_assets','referenced_meshes'):
        for f,d in evidence[key].items(): assert sha(f)==d
    backend=Path(contract['backend_readback']);assert sha(backend)==contract['backend_readback_sha256']
    compiled=json.loads(backend.read_bytes());assert torch.__version__==compiled['torch_version']
    for row in compiled['extension_modules'].values():assert sha(row['file'])==row['sha256']
    original=CuroboPlanner.__init__
    report.update(expert_backend='native_curobo',expert_control_readback_sha256=sha(proof),
        native_collision_configuration='unmodified_native_table_only_curobo',backend_resources=[])

    def runtime_init(self,*args,**kwargs):
        assert len(args)==3 and set(kwargs)=={'yml_path'}
        native=Path(kwargs['yml_path']);assert str(native) in expected
        row=prepare(native,os.environ['ROBOTWIN_PATH'],Path(os.environ['TASK_LOCAL'])/'native-config'/native.name)
        for key in ('native_yaml_sha256','actual_urdf_sha256','actual_collision_yaml_sha256','only_asset_paths_changed'):
            assert row[key]==expected[str(native)][key]
        original(self,*args,yml_path=row['runtime_yaml'])
        for motor in (self.motion_gen,self.motion_gen_batch):
            device=torch.device(motor.tensor_args.device)
            assert device.type=='cuda' and device.index in (None,0)
        free,total=torch.cuda.mem_get_info()
        report['backend_resources'].append(dict(cuda_free_bytes=free,cuda_total_bytes=total,
            runtime_yaml_sha256=row['runtime_yaml_sha256'],native_yaml_sha256=row['native_yaml_sha256']))
        write(output/'RESULT.json',report)
        assert free>=max(8*2**30,int(.08*total)), 'Native expert violates the GPU reserve'

    CuroboPlanner.__init__=runtime_init
    return lambda:setattr(CuroboPlanner,'__init__',original)
