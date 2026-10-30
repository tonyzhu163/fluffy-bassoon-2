"""Distribution intervention algebra, original sampler parity and exact recovery."""
from copy import deepcopy
import json
import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.vla_distribution_control import DistributionController,qualify_original
from fluffy_bassoon.meta_rl.vla_distribution_plan import build_plan,METHODS,OPERATIONS
from fluffy_bassoon.meta_rl.vla_history import HistoryConfig,HistoryNoisePolicy,FrozenHistoryController
from fluffy_bassoon.meta_rl.vla_source40_evaluate import _episode_rng,_policy_signature
from fluffy_bassoon.meta_rl.vla_source40_two_goal_evaluate import RecordingController
import fluffy_bassoon.meta_rl.vla_distribution_diagnostic as module


def policy():
    torch.manual_seed(7)
    return HistoryNoisePolicy(HistoryConfig(4104,32,width=8,layers=1,heads=2,context_length=8))


def plan():
    return build_plan(worktree='/allocated/diagnostic',source_commit='a'*40,
        inputs=dict(matrix='/immutable/matrix',matrix_sha256='matrix',assembly='/immutable/assembly',assembly_sha256='assembly'))


def test_fixed9_seven_methods_query_only_have_exact_complete_costs():
    p=plan()
    assert [e['case']['task_id'] for e in p['cases']]==list(range(0,90,10))
    assert len(p['groups'])==63 and sum(g['maximum_actions'] for g in p['groups'])==25830
    assert p['counts']['action_cap']==27000 and p['counts']['gpu_seconds']==3600
    assert p['contract']['qualification_extra_resets']==0
    assert all([g['method'] for g in e['groups']]==list(METHODS) for e in p['cases'])
    assert all(g['support_budgets']==[0] and g['case']['initial_state']==0 and g['case']['scene_seed']==123 for g in p['groups'])


def test_original_reproduces_current_stochastic_actions_feedback_cache_and_rng():
    model=policy();before=_policy_signature(model)
    assert qualify_original(model,np.zeros(4104,np.float32))['bitwise_actions']
    a=FrozenHistoryController(model,mode='full');b=DistributionController(model)
    _episode_rng(a,123);_episode_rng(b,123)
    for step in range(6):
        obs=np.full(4104,step/10,np.float32)
        x,y=a.act(obs),b.act(obs)
        np.testing.assert_array_equal(x,y)
        assert torch.equal(a.generator.get_state(),b.generator.get_state())
        for c,z in ((a,x),(b,y)):c.observe(np.clip(z,-1,1),-1.,step in (2,5))
    assert _policy_signature(model)==before


@pytest.mark.parametrize('mean,std',[('zero','learned'),('original','unit')])
def test_overrides_retain_base_policy_and_one_exact_shared_random_draw(mean,std):
    model=policy();reference=DistributionController(model);changed=DistributionController(model,mean=mean,std=std)
    _episode_rng(reference,123);_episode_rng(changed,123)
    reference.act(np.zeros(4104,np.float32));z=changed.act(np.zeros(4104,np.float32))
    a,b=reference.last_distribution,changed.last_distribution
    np.testing.assert_array_equal(a['epsilon'],b['epsilon'])
    np.testing.assert_array_equal(a['base_mean'],b['base_mean'])
    assert torch.equal(reference.generator.get_state(),changed.generator.get_state())
    expected_mean=np.zeros(32) if mean=='zero' else b['base_mean']
    expected_std=np.ones(32) if std=='unit' else b['learned_std']
    np.testing.assert_array_equal(b['applied_mean'],expected_mean);np.testing.assert_array_equal(b['applied_std'],expected_std)
    assert b['random_draws']==1 and b['parameter_updates']==0 and changed.policy is model
    changed.observe(np.clip(z,-1,1),-1.,True)
    np.testing.assert_array_equal(changed.previous_noise,np.clip(z,-1,1))


def test_complete_distribution_snapshot_preserves_future_sample_and_before_rpc_receipts(tmp_path):
    model=policy();controller=DistributionController(model,mean='zero',record_output=tmp_path/'raw')
    _episode_rng(controller,123);x=controller.act(np.zeros(4104,np.float32))
    assert (tmp_path/'raw/draw-00001.pkl').is_file()
    controller.observe(np.clip(x,-1,1),-1.,True);saved=controller.snapshot()
    expected=controller.act(np.ones(4104,np.float32))
    resumed=DistributionController(model,mean='zero');resumed.restore(saved)
    np.testing.assert_array_equal(resumed.act(np.ones(4104,np.float32)),expected)
    assert resumed.draw_count==2
    wrong=DistributionController(model,std='unit')
    with pytest.raises(ValueError,match='intervention/recovery'):wrong.restore(saved)


def test_recording_wrapper_forwards_exact_distribution_to_trace():
    model=policy();base=DistributionController(model)
    wrapped=RecordingController(base,output='unused',method='ad_full',trial_index=0,motor_rng_state=lambda:None)
    wrapped.set_episode_seed(123);wrapped.act(np.zeros(4104,np.float32))
    assert wrapped.last_distribution is base.last_distribution
    assert wrapped.last_distribution['mean_operation']=='original'


def test_reservation_is_exact_read_only_and_not_created(tmp_path):
    p=plan();path=tmp_path/'ledger.json'
    r=dict(run_id='diagnostic',source_commit='a'*40,diagnostic_sha256=p['diagnostic_sha256'],maximum_total_actions=27000,
        worker_seconds=3600,purpose='target_evaluation',source_pretraining_data=False,fresh_ppo_training_data=False)
    ledger=dict(format=module.LEDGER_FORMAT,diagnostic_sha256=p['diagnostic_sha256'],maximum_total_actions=27000,
        maximum_gpu_seconds=3600,source_campaign_immutable=True,scope_amendment='Future explicitly reconciled unused time',reservations={'diagnostic':r})
    path.write_text(json.dumps(ledger));before=path.read_bytes()
    assert module.reservation(path,p,'diagnostic','a'*40)['record']==r and path.read_bytes()==before
    ledger['reservations']['diagnostic']['maximum_total_actions']=28000;path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError,match='existing exact27k'):module.reservation(path,p,'diagnostic','a'*40)
