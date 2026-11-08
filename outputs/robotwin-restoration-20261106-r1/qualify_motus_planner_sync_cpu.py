"""Run-specific native SDK routing reference; no scene construction or GPU work."""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from mplib.sapien_utils import SapienPlanner
from envs.robot.planner import MplibWrapperPlanner, MplibPlanner
from motus_planner_full_state import full_qpos, plan_path_sync, plan_batch_sync

ROOT = Path(__file__).resolve().parent
BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')


def main():
    assert os.uname().sysname == 'Linux' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not torch.cuda.is_initialized()
    source = os.environ['TASK_SOURCE']; code = ROOT.parents[1]
    assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==source
    contract = json.loads((ROOT/'MOTUS-PLANNER-CONTROL02-CONTRACT.json').read_bytes())
    for name,digest in contract['source_files'].items():
        p=code/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
        if p.suffix=='.py':ast.parse(p.read_text())
    subprocess.run(['bash','-n',str(ROOT/'motus_planner_control02.sh')],check=True)
    p=BASE/'motus-planner-control01/RESULT.json'
    assert hashlib.sha256(p.read_bytes()).hexdigest()=='88113f90979a0110bdd649985d951a0fd6512a435c01cc2bd0ffcfb1bd69b336'
    prior=json.loads(p.read_bytes());events=[];calls=[]
    def update(*,update_attached_object=True):
        assert update_attached_object;events.append('world_update')
    proxy=SimpleNamespace(planning_world=SimpleNamespace(update_from_simulation=update))
    rows=[]
    for ref in [r for r in prior['planner_calls'] if r['profile']=='full_state']:
        owner=object.__new__(MplibWrapperPlanner);owner.all_joints=ref['all_joints'];owner.active_joints_name=ref['active_joints']
        owner.planner=SimpleNamespace(update_from_simulation=lambda:SapienPlanner.update_from_simulation(proxy))
        measured=np.asarray(ref['measured'],dtype=np.float32);before=measured.copy();expected=full_qpos(owner,measured)
        def solve(instance,*,now_qpos,**options):
            assert events[-1]=='world_update';events.append('native_solver')
            np.testing.assert_array_equal(now_qpos,expected);calls.append(now_qpos.copy())
            return dict(status='Success',position=np.zeros((2,6)),velocity=np.zeros((2,6)))
        target=SimpleNamespace(p=[0.,0.,.8],q=[1.,0.,0.,0.])
        with patch.object(MplibPlanner,'plan_path',solve):
            plan_path_sync(owner,measured,target)
            result=plan_batch_sync(owner,measured,[target,target])
            assert result['status'].tolist()==['Success','Success']
        np.testing.assert_array_equal(measured,before)
        rows.append(dict(seed=ref['seed'],DOFs=len(measured),active=ref['active_joints']))
    assert len(rows)==88 and len(calls)==264
    assert events==['world_update','native_solver']*264
    assert not torch.cuda.is_initialized()
    result=dict(status='passed_native_SapienPlanner_world_sync_and_actual38DOF_CPU_reference',source_commit=source,
        source_reference_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),actual_GPU_state_fixtures=88,single_and_batch_calls=264,
        each_world_sync_precedes_native_solver=True,measured_source_unmutated=True,
        native_sync_source_sha256=hashlib.sha256(inspect.getsource(SapienPlanner.update_from_simulation).encode()).hexdigest(),
        actual_simulators_constructed=False,actual_planner_solutions_computed=False,GPU_seconds=0,GPU_positive_controls_still_required=True)
    out=BASE/'cpumotusplannersync01';out.mkdir();p=out/'RESULT.json';p.write_text(json.dumps(result,indent=2)+'\n')
    print('PLANNER_SYNC_CPU_PASS',str(p),hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)


if __name__=='__main__':main()
