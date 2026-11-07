"""Protect named Adam ownership checks across native empty-shard offload."""
import copy
import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1]/'outputs/robotwin-sidney-matched-baselines-20261106-r1/optimizer_semantics.py'
_SPEC = importlib.util.spec_from_file_location('robotwin_optimizer_semantics',_PATH)
_RULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RULE)


def tensor(shape,values):
    return dict(kind='tensor',shape=shape,numel=len(values),dtype='torch.float32',
        finite=True,nonzero_elements=sum(v!=0 for v in values),values=values,sha256=str(values))


def fixture():
    owned = dict(parameter_shape=[2],parameter_numel=2,parameter_dtype='torch.float32',
        requires_grad=True,registered_model_parameter=True,local_nonempty_parameter=True,state_present=True,
        state=dict(step=tensor([], [1.]),exp_avg=tensor([2],[.2,.3]),exp_avg_sq=tensor([2],[.04,.09])))
    empty = dict(parameter_shape=[0],parameter_numel=0,parameter_dtype='torch.float32',
        requires_grad=True,registered_model_parameter=True,local_nonempty_parameter=False,state_present=False,state={})
    return dict(rank=0,optimizer_steps=1,parameters=dict(owned=owned,empty=empty),
        groups=[dict(parameters=['owned','empty'],hyperparameters=dict(lr=5e-6))])


def test_absent_materialized_empty_and_unused_zero_moments_have_same_semantics():
    absent = fixture()
    materialized = copy.deepcopy(absent);materialized['parameters']['empty']['state_present']=True
    moments = copy.deepcopy(materialized)
    moments['parameters']['empty']['state'] = dict(step=tensor([],[0.]),exp_avg=tensor([0],[]),exp_avg_sq=tensor([0],[]))
    assert len({_RULE.semantic_digest(v) for v in (absent,materialized,moments)}) == 1
    assert materialized['parameters']['empty']['state_present']  # Validation is read-only.


@pytest.mark.parametrize('change',['owned_empty','owned_absent','empty_owned','unregistered','frozen','nonzero_step','partial_state','duplicate_group'])
def test_empty_mapping_exception_never_hides_owned_or_nontrivial_state(change):
    value = fixture();empty=value['parameters']['empty'];owned=value['parameters']['owned']
    empty['state_present']=True
    if change=='owned_empty':owned['state']={}
    elif change=='owned_absent':owned['state_present']=False
    elif change=='empty_owned':empty.update(parameter_numel=1,local_nonempty_parameter=True)
    elif change=='unregistered':empty['registered_model_parameter']=False
    elif change=='frozen':empty['requires_grad']=False
    elif change=='nonzero_step':empty['state']=dict(step=tensor([],[1.]),exp_avg=tensor([0],[]),exp_avg_sq=tensor([0],[]))
    elif change=='partial_state':empty['state']=dict(step=tensor([],[0.]))
    elif change=='duplicate_group':value['groups'][0]['parameters'].append('empty')
    with pytest.raises(AssertionError):_RULE.semantic_digest(value)


@pytest.mark.parametrize('change',['moment_bytes','group_lr','counter'])
def test_owned_bytes_optimizer_options_and_counter_remain_exact(change):
    original=fixture();changed=copy.deepcopy(original)
    if change=='moment_bytes':changed['parameters']['owned']['state']['exp_avg']['sha256']='different-bytes'
    elif change=='group_lr':changed['groups'][0]['hyperparameters']['lr']=1e-4
    elif change=='counter':changed['optimizer_steps']+=1
    assert _RULE.semantic_digest(original) != _RULE.semantic_digest(changed)
