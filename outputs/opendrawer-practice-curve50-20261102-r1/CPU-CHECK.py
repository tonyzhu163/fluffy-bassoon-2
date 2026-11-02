"""Qualify actual-checkpoint query-branch isolation on allocated Linux CPU."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile

import numpy as np
import torch
from fluffy_bassoon.meta_rl.residual_history import HistoryResidualActor
from fluffy_bassoon.meta_rl.vla_history import load_policy

assert sys.platform=='linux' and not torch.cuda.is_available()
torch.set_num_threads(2)
CODE=Path(__file__).resolve().parents[2]
ROOT=Path('/mowenjie/fluffy-bassoon-2')
report=dict(passed=True,checked_at=datetime.now(timezone.utc).isoformat(),checks=[],new_simulator_episodes=0)
for method in ('ad_full','dpt_full'):
    os.environ['TASK_CURVE_METHOD']=method
    os.environ['TASK_CURVE_OUTPUT']=str(ROOT/'outputs'/f'opendrawer-{method}-practice-qualification')
    m=runpy.run_path(str(Path(__file__).with_name('CURVE.py')))
    with tempfile.TemporaryDirectory() as temporary:
        journal_function=m['make_journal']
        original_out=journal_function.__globals__['OUT']
        journal_function.__globals__['OUT']=Path(temporary)
        os.environ['TASK_ATTEMPT']='pinned-runtime-api-check'
        try:
            for rank in range(8):
                journal=journal_function(rank)
                cursor=journal.begin(50,phase='CPU_API_CHECK_NO_SIMULATOR')
                journal.finish(cursor,0)
                assert journal.charged()==0
        finally:journal_function.__globals__['OUT']=original_out
    report.setdefault('startup_checks',[]).append(dict(method=method,pinned_journal_API_passed=True,
        all_eight_rank_cohorts_fit_recording_guard=True,constructor_checked_before_motor_loading=True))
    b=json.loads((m['SOURCE']/'binding.json').read_text());candidate=b['selected'][method]['candidate']
    policy,_=load_policy(candidate['path'],expected_sha256=candidate['sha256'],device='cpu')
    frozen={k:v.clone() for k,v in policy.state_dict().items()}
    for cohort in ('ID','OOD'):
        p=m['SOURCE']/'episodes'/cohort/'case-000'/method/'K1'
        sr=json.loads((p/'support/result.json').read_text());qr=json.loads((p/'query/result.json').read_text())
        support=torch.load(sr['records']['path'],map_location='cpu',weights_only=False)
        query=torch.load(qr['records']['path'],map_location='cpu',weights_only=False)
        assert support[-1]['reward']==sr['success']
        actor=HistoryResidualActor(policy,mode='full',reset_each_episode=False);m['replay'](actor,support)
        before=m['memory_signature'](actor);branch=m['fork_actor'](actor)
        assert m['memory_signature'](branch)==before
        assert actor.controller.state['memory'].data_ptr()!=branch.controller.state['memory'].data_ptr()
        assert not np.shares_memory(actor.controller.previous_noise,branch.controller.previous_noise)
        branch.begin_episode();m['replay'](branch,query)
        assert m['memory_signature'](actor)==before,'Query changed practice state'
        untouched=m['fork_actor'](actor)
        x=torch.as_tensor(query[0]['features'][None])
        actual=actor.act(x,deterministic=True)[0];expected=untouched.act(x,deterministic=True)[0]
        assert torch.equal(actual,expected),'Next practice depends on scored query'
        actor=HistoryResidualActor(policy,mode='full',reset_each_episode=False)
        # Capacity check only: repeating retained rows creates no scientific outcomes.
        for k in range(1,6):
            m['replay'](actor,support)
            assert int(actor.controller.state['length'][0])==k*len(support)
        branch=m['fork_actor'](actor);m['replay'](branch,query)
        assert int(branch.controller.state['length'][0])==6*len(support)<=128
        report['checks'].append(dict(method=method,cohort=cohort,query_cache_and_previous_control_disjoint=True,
            query_does_not_change_practice_state=True,next_practice_action_matches_untouched_branch_exactly=True,
            five_support_plus_query_cache_length=6*len(support),weights_frozen=True))
    assert all(torch.equal(v,frozen[k]) for k,v in policy.state_dict().items())
report['source']=m['BASE']['head'](CODE)
print(json.dumps(report,indent=2),flush=True)
