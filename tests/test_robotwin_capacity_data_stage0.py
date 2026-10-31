"""Allocated Linux CPU only; no VLA/model/simulator imports or downloads."""
import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest


spec=importlib.util.spec_from_file_location('capacity_stage0',Path(__file__).parents[1]/'scripts/diagnostics/robotwin_capacity_data_stage0.py')
stage=importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


def test_zip_episode_order_is_numeric_whole_and_not_lexical(tmp_path):
    path=tmp_path/'demos.zip'
    with zipfile.ZipFile(path,'w') as data:
        for i in [10,2,1,0,11,3,4,5,6,7,8,9]: data.writestr(f'data/episode{i}.hdf5',b'fixture')
    with zipfile.ZipFile(path) as data:
        assert [i for i,_ in stage.ordered_hdf5(data)]==list(range(12))


@pytest.mark.parametrize('extra',['../escape.txt','another/episode0.hdf5'])
def test_zip_rejects_escape_and_duplicate_episode_identity(tmp_path,extra):
    path=tmp_path/'demos.zip'
    with zipfile.ZipFile(path,'w') as data:
        for i in range(12): data.writestr(f'episode{i}.hdf5',b'fixture')
        data.writestr(extra,b'bad')
    with zipfile.ZipFile(path) as data:
        with pytest.raises(ValueError): stage.ordered_hdf5(data)


def test_hdf5_inventory_retains_drive_targets_metadata_and_unknown_alignment(tmp_path):
    import h5py
    path=tmp_path/'episode0.hdf5'
    with h5py.File(path,'w') as data:
        data.attrs.update(success=True,seed=77,instruction='actual source instruction')
        data.create_dataset('joint_action/vector',data=np.arange(56,dtype=np.float32).reshape(4,14)/100)
        data.create_dataset('timestamps',data=np.arange(4,dtype=np.float64)*.02)
        data.create_dataset('observation/head_camera/rgb',data=np.zeros((4,2,2,3),np.uint8))
    value=stage.hdf5_inventory(path,lambda:None)
    assert value['explicit_success'] is True and value['seed_metadata']==dict(seed=77)
    assert value['instruction_metadata']==dict(instruction='actual source instruction')
    assert value['datasets']['joint_action/vector']['shape']==[4,14]
    assert 'not inferred' in value['datasets']['joint_action/vector']['data_semantics']
    assert value['datasets']['timestamps']['metadata_values']==[0.,.02,.04,.06]


def test_unknown_success_is_preserved_instead_of_manufactured(tmp_path):
    import h5py
    path=tmp_path/'episode0.hdf5'
    with h5py.File(path,'w') as data: data.create_dataset('joint_action/vector',data=np.zeros((3,14),np.float32))
    assert stage.hdf5_inventory(path,lambda:None)['explicit_success'] is None


def test_model_import_firewall_rejects_without_importing_model():
    with pytest.raises(ImportError,match='rejects'):
        stage.NoModels().find_spec('torch')
    with pytest.raises(ImportError,match='rejects'):
        stage.NoModels().find_spec('rlinf.models')


def test_gripper_statistics_use_last_actual_masked_commands_only(tmp_path):
    root=tmp_path/'corpus';chunks=root/'episodes/episode-x/chunks';chunks.mkdir(parents=True)
    actions=np.zeros((3,14),np.float32);actions[0,[6,13]]=[999,999];actions[1,[6,13]]=[-.1,1.2];actions[2,[6,13]]=[.2,.8]
    np.savez(chunks/'decision.npz',issued_motor_joint_targets=actions)
    np.savez(chunks/'native.npz',valid_action_mask=np.array([0,1,1],bool))
    def ref(name):
        path=chunks/name
        return dict(path=str(path.relative_to(root)),sha256=stage.sha(path))
    (chunks/'000001.json').write_text(json.dumps(dict(status='complete',decision_arrays=ref('decision.npz'),
        native_result=dict(arrays=ref('native.npz'),executed_primitive_steps=2))))
    value=stage.a1_grippers([root],lambda:None)
    assert value['executed_actions']==2 and value['outside_0_1_counts']==[1,1]
    assert value['maximum'][0]<1 and value['maximum'][1]<2
