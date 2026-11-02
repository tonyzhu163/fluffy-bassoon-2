"""Audit saved direct-head fitting, CPU recovery, data split and competence gates."""
import importlib.util
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch

ROOT=Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-interface-screen-20261102-r3')
CODE=Path('/mowenjie/fluffy-bassoon-2/worktrees/robocasa-interface-screen-20261102-r3/outputs/robocasa-interface-screen-20261102-r3')
TASKS=('OpenDrawer','PickPlaceDrawerToCounter','CoffeeSetupMug','TurnOffStove')


def module(name,file):
    spec=importlib.util.spec_from_file_location(name,CODE/file)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


carry=module('carry','CARRY.py')
interfaces=module('interfaces','INTERFACES.py')


def load(path):return torch.load(path,map_location='cpu',weights_only=False)


def mse(head,x,y,mask):
    indices=mask.nonzero().flatten()
    total,count=0.,0
    with torch.inference_mode():
        for batch in indices.split(128):
            error=(head(x[batch])-y[batch]).square()
            total+=float(error.sum());count+=error.numel()
    return total/count


def main():
    assert platform.system()=='Linux' and not torch.cuda.is_available()
    result={'completed':True,'checked_unix':time.time(),'PPO_updates':0,'tasks':[],
        'recovery_scope':'Selected/latest models and Adam load on CPU; CPU/Python/NumPy RNG roundtrip; CUDA global/sampler byte states retained and hashed. No CUDA resume or new GPU inference was performed.'}
    for rank,task in enumerate(TASKS):
        directory=ROOT/task
        fit=json.loads((directory/'DIRECT-FIT.json').read_text())
        assert fit['completed'] and fit['PPO_updates']==0
        bank,records,anchor_refs=carry.load_anchors(task)
        anchor_count=len(records)
        assert anchor_count==fit['anchor_records']
        for source in fit['source_refs']:
            assert carry.sha(source['path'])==source['sha256']
            payload=load(source['path']);assert payload['stage']=='pending_labels'
            records.extend(payload['pending_records'])
        x=torch.tensor(np.stack([r['features'] for r in records]),dtype=torch.float32)
        y=torch.tensor(np.stack([r['base_chunk'] for r in records]),dtype=torch.float32)
        manifest=json.loads((directory/'manifest.json').read_text())
        stats=json.loads(Path(manifest['initializer']['norm_path']).read_text());stats=stats.get('norm_stats',stats)
        y=(y-torch.tensor(stats['actions']['mean'][:12],dtype=torch.float32))/(torch.tensor(stats['actions']['std'][:12],dtype=torch.float32)+1e-6)
        groups=np.cumsum([r['episode_start'] for r in records])-1
        validation=torch.tensor(groups%5==0)
        assert all(bool(records[i]['done'])==bool(i+1==len(records) or records[i+1]['episode_start']) for i in range(len(records)))
        selected=load(directory/'direct-selected.pt');latest=load(directory/'direct-latest.pt')
        assert selected['best_step']==latest['best_step']==fit['selected_update']
        assert latest['step']==fit['updates']
        assert latest['contract']['sources']==fit['source_refs'] and latest['contract']['base_action_features_removed']
        head=interfaces.DirectHead();head.load_state_dict(selected['model'],strict=True);head.eval()
        assert all(torch.equal(value,latest['best_state'][key]) for key,value in selected['model'].items())
        inputs=head.inputs(x[~validation])
        torch.testing.assert_close(head.mean,inputs.mean(0),rtol=1e-4,atol=1e-5)
        torch.testing.assert_close(head.std,inputs.std(0,unbiased=False).clamp_min(1e-4),rtol=1e-4,atol=1e-5)
        val=mse(head,x,y,validation)
        target=mse(head,x[:anchor_count],y[:anchor_count],validation[:anchor_count])
        assert abs(val-fit['validation_mse'])<1e-4 and abs(target-fit['task_validation_mse'])<1e-4
        assert bool(val<=.25 and target<=.25)==fit['preliminary_mse_gate']
        resumed=interfaces.DirectHead();resumed.load_state_dict(latest['model'],strict=True)
        optimizer=torch.optim.AdamW(resumed.parameters(),lr=3e-4);optimizer.load_state_dict(latest['optimizer'])
        assert len(optimizer.state)==4
        for parameter,state in optimizer.state.items():
            assert int(state['step'])==latest['step']
            for key in ['exp_avg','exp_avg_sq']:
                assert state[key].shape==parameter.shape and torch.isfinite(state[key]).all()
        rng=latest['global_rng']
        py=random.Random();py.setstate(rng['python']);assert py.getstate()==rng['python']
        numpy=np.random.RandomState();numpy.set_state(rng['numpy']);assert np.array_equal(numpy.get_state()[1],rng['numpy'][1])
        cpu=torch.Generator();cpu.set_state(rng['torch']);assert torch.equal(cpu.get_state(),rng['torch'])
        assert len(rng['cuda'])==1 and rng['cuda'][0].dtype==torch.uint8 and rng['cuda'][0].numel()>0
        assert latest['sampler_rng'].dtype==torch.uint8 and latest['sampler_rng'].numel()>0
        best,best_step,stale=float('inf'),0,0
        for log in fit['log']:
            if log['validation_mse']<best-1e-5:best,best_step,stale=log['validation_mse'],log['step'],0
            else:stale+=1
        assert best==fit['validation_mse'] and best_step==latest['best_step']
        native_path=directory/'direct-qualification-native/RESULT.json'
        head_path=directory/'direct-qualification-head/RESULT.json'
        competence=None
        if fit['preliminary_mse_gate']:
            assert native_path.exists() and head_path.exists()
            a=json.loads(native_path.read_text());b=json.loads(head_path.read_text())
            assert len(a['rows'])==len(b['rows'])==8
            assert [r['case'] for r in a['rows']]==[r['case'] for r in b['rows']]
            native=sum(r['success'] for r in a['rows']);candidate=sum(r['success'] for r in b['rows'])
            competence={'native_successes':native,'direct_successes':candidate,'cases':8,'admitted':candidate>=max(1,native-1)}
        result['tasks'].append(dict(task=task,fit=carry.ref(directory/'DIRECT-FIT.json'),
            selected_checkpoint=carry.ref(directory/'direct-selected.pt'),latest_recovery_checkpoint=carry.ref(directory/'direct-latest.pt'),
            anchor_record_refs=anchor_refs,data_records=len(records),episodes=int(groups[-1]+1),
            train_records=int((~validation).sum()),validation_records=int(validation.sum()),
            CPU_validation_mse=val,CPU_task_validation_mse=target,preliminary_mse_gate=fit['preliminary_mse_gate'],
            base_action_features_removed=True,model_Adam_CPU_RNG_recovery_verified=True,
            CUDA_global_and_sampler_states_retained=True,derived_resume_best_mse=best,derived_resume_stale_checks=stale,
            competence=competence))
    path=ROOT/'DIRECT-AUDIT.json'
    with path.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
