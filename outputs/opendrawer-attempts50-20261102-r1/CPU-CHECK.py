"""Actual checkpoint recovery, feedback continuity and pinned journal qualification."""
import json,os,runpy,sys,tempfile
from pathlib import Path
import torch
from fluffy_bassoon.meta_rl.residual_history import HistoryResidualActor
from fluffy_bassoon.meta_rl.vla_history import load_policy
assert sys.platform=='linux' and not torch.cuda.is_available()
torch.set_num_threads(2)
report={'passed':True,'checks':[],'new_simulator_episodes':0}
root=Path('/mowenjie/fluffy-bassoon-2')
for family in ('ad','dpt'):
    os.environ['TASK_CURVE_METHOD']=family+'_full';os.environ['TASK_CURVE_OUTPUT']='/tmp/unused-attempt-qualification'
    m=runpy.run_path(str(Path(__file__).with_name('ATTEMPTS.py')))
    old=json.loads((root/'outputs'/f'opendrawer-{family}-practice-curve50-20261102-r2/binding.json').read_text())
    policy,_=load_policy(old['selected']['candidate']['path'],expected_sha256=old['selected']['candidate']['sha256'],device='cpu')
    frozen={k:v.clone() for k,v in policy.state_dict().items()}
    with tempfile.TemporaryDirectory() as tmp:
        f=m['make_journal'];save=f.__globals__['OUT'];f.__globals__['OUT']=Path(tmp);os.environ['TASK_ATTEMPT']='cpu-api-check'
        try:
            for rank in range(8):
                journal=f(rank);t=journal.begin(50,phase='CPU_CHECK_NO_SIMULATOR');journal.finish(t,0);assert journal.charged()==0
        finally:f.__globals__['OUT']=save
    for c in ('ID','OOD'):
        candidates=[]
        for i in range(50):
            refs=[old['reuse'][f'{c}-{i:03d}']['support1']]
            for k in range(2,6):
                p=root/'outputs'/f'opendrawer-{family}-practice-curve50-20261102-r2/episodes'/c/f'case-{i:03d}'/f'K{k}'/'support/result.json'
                if not p.exists():break
                refs.append(m['ref'](p))
            candidates.append((len(refs),i,refs))
        n,i,refs=max(candidates)
        actor=HistoryResidualActor(policy,mode='full',reset_each_episode=False)
        initial=m['memory_signature'](actor)
        for k,ref in enumerate(refs,1):
            r=m['read_ref'](ref)
            assert r['motor_seeds']==m['motor_seeds'](c,i,k)
            rows=torch.load(r['records']['path'],map_location='cpu',weights_only=False)
            assert rows[-1]['reward']==r['success'] and rows[-1]['done']
            if k>=2:
                assert actor.controller.previous_reward==r['memory_entry']['previous_reward']
                assert actor.controller.previous_done==r['memory_entry']['previous_done']
            m['replay'](actor,rows)
            assert int(actor.controller.state['length'][0])==k*(m['horizon'](c)//50)
            assert actor.controller.previous_reward==r['success'] and actor.controller.previous_done
        fresh=HistoryResidualActor(policy,mode='full',reset_each_episode=False)
        assert m['memory_signature'](fresh)==initial
        report['checks'].append({'method':family,'cohort':c,'case':i,'actual_prefix_length':n,'feedback_and_cache_continuity':True,'fresh_case_empty_memory':True,'journal_api':True})
    assert all(torch.equal(v,frozen[k]) for k,v in policy.state_dict().items())
report['source']=m['BASE']['head'](Path(__file__).resolve().parents[2])
print(json.dumps(report,indent=2))
