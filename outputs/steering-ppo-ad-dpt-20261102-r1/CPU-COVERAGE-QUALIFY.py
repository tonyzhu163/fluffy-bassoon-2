"""Numerically qualify complete-episode sampling and exact recovery, no simulator."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import sys

import numpy as np
import torch
from fluffy_bassoon.meta_rl.vla_history import HistoryConfig,HistoryNoisePolicy,FrozenHistoryController,sample_batch
from fluffy_bassoon.meta_rl.vla_history_train import supervised_update,_rng_state,_restore_rng,validate

assert sys.platform=='linux' and not torch.cuda.is_available()
torch.set_num_threads(2)
report=dict(passed=True,kind='synthetic numerical reference, not a robot outcome',actual_simulator_episodes=0,arms=[])
for arm,dimension in [('velocity',32),('kv_all_layers',1152)]:
    rng=np.random.default_rng(924001);trials=[]
    for index in range(4):
        length=120;u=rng.normal(0,.5,(length,dimension)).astype(np.float32)
        trial=dict(observations=rng.normal(size=(length,2064)).astype(np.float32),executed_noise=u,
            target_noise=u.copy(),rewards=np.zeros(length,np.float32),dones=np.tile([False]*11+[True],10),
            episode_starts=np.tile([True]+[False]*11,6),teacher_steps=np.repeat(np.arange(1,11,dtype=np.int64),12))
        trial['rewards'][5::12]=1.;trials.append(trial)
    torch.manual_seed(0)
    config=HistoryConfig(2064,dimension,width=128,layers=2,heads=4,context_length=300)
    policy=HistoryNoisePolicy(config);policy.fit_observation_normalization(trials[:3])
    optimizer=torch.optim.AdamW(policy.parameters(),lr=3e-4,weight_decay=0.)
    sampler=np.random.default_rng(0)
    def batch(r):return sample_batch(trials[:3],policy,r,batch_size=8,max_context=None,max_episodes=9)
    first=batch(sampler);counts=first[2].sum(0)
    assert all(int(v)%12==0 and 12<=int(v)<=108 for v in counts)
    supervised_update(policy,optimizer,first,mode='full')
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'recovery.pt'
        torch.save(dict(config=config,policy=policy.state_dict(),optimizer=optimizer.state_dict(),rng=_rng_state(sampler)),path)
        second=batch(sampler);reference_metrics=supervised_update(policy,optimizer,second,mode='full')
        expected={k:v.clone() for k,v in policy.state_dict().items()};expected_optimizer=deepcopy(optimizer.state_dict())
        expected_sampler=deepcopy(sampler.bit_generator.state)
        saved=torch.load(path,map_location='cpu',weights_only=False)
        restored=HistoryNoisePolicy(saved['config']);restored.load_state_dict(saved['policy'])
        restored_optimizer=torch.optim.AdamW(restored.parameters(),lr=3e-4,weight_decay=0.)
        restored_optimizer.load_state_dict(saved['optimizer']);resumed_sampler=np.random.default_rng(998)
        _restore_rng(saved['rng'],resumed_sampler)
        resumed_batch=sample_batch(trials[:3],restored,resumed_sampler,batch_size=8,max_context=None,max_episodes=9)
        assert all(torch.equal(a,b) for a,b in zip(second,resumed_batch))
        resumed_metrics=supervised_update(restored,restored_optimizer,resumed_batch,mode='full')
        assert all(torch.equal(expected[k],v) for k,v in restored.state_dict().items())
        assert expected_sampler==resumed_sampler.bit_generator.state
        actual_optimizer=restored_optimizer.state_dict()
        for key,values in expected_optimizer['state'].items():
            for name,value in values.items():
                actual=actual_optimizer['state'][key][name]
                assert torch.equal(value,actual) if torch.is_tensor(value) else value==actual
        assert reference_metrics['nll']==resumed_metrics['nll']
    frozen={k:v.clone() for k,v in restored.state_dict().items()}
    one=validate(restored,trials[3:],mode='full',batch_size=8,batches=2,max_context=None,max_episodes=9)
    two=validate(restored,trials[3:],mode='full',batch_size=8,batches=2,max_context=None,max_episodes=9)
    assert one==two
    actor=FrozenHistoryController(restored,mode='full')
    for i in range(105):
        control=actor.act(trials[0]['observations'][i%120],deterministic=True)
        actor.observe(control,float(i%21==10),i%21==20)
    assert int(actor.state['length'][0])==105
    actor.reset_trial();assert actor.state is None and not actor.previous_noise.any()
    assert all(torch.equal(frozen[k],v) for k,v in restored.state_dict().items())
    coverage_rng=np.random.default_rng(734001)
    coverage=[]
    for _ in range(128):
        probe=sample_batch(trials[:3],restored,coverage_rng,batch_size=8,max_context=None,max_episodes=9)
        coverage.extend(int(v)//12 for v in probe[2].sum(0))
        if 9 in coverage:break
    assert 9 in coverage
    report['arms'].append(dict(arm=arm,features=2064,steering_dimension=dimension,context_capacity=300,
        sampled_complete_episode_counts=[int(v)//12 for v in counts],logical_batch_size=8,maximum_complete_source_episodes=9,observed_108_token_training_window=True,
        on_disk_optimizer_sampler_all_CPU_RNG_recovery_exact=True,validation_windows_and_NLL_repeat_exactly=True,
        five_1050_control_attempts_fit_without_eviction=True,next_case_empty_memory=True,
        policy_and_normalization_unchanged_during_validation_and_acting=True))
print(json.dumps(report,indent=2))
