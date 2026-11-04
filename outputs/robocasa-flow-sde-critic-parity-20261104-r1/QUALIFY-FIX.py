"""Numerically qualify the actual patched sampler methods on Linux CPU."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

import torch


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def functions(source,names,namespace):
    found={n.name:n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name in names}
    assert set(found)==set(names)
    for node in found.values():node.decorator_list=[]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*found.values()],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),'<native-methods>','exec'),namespace)
    return {name:namespace[name] for name in names}


def main():
    p=argparse.ArgumentParser();p.add_argument('--rlinf',type=Path,required=True)
    p.add_argument('--patch',type=Path,required=True);p.add_argument('--stub',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();torch.set_num_threads(2)
    relative=Path('rlinf/models/embodiment/openpi/tasks/rl.py');original=(args.rlinf/relative).read_text()
    sampler_path=args.rlinf/'rlinf/models/embodiment/openpi/sampling/rl_sampler.py'
    spec=importlib.util.spec_from_file_location('qualification_sampler',sampler_path)
    sampler=importlib.util.module_from_spec(spec);spec.loader.exec_module(sampler)
    with tempfile.TemporaryDirectory(prefix='suffix-critic-') as directory:
        target=Path(directory)/relative;target.parent.mkdir(parents=True);target.write_text(original)
        subprocess.run(['git','apply','--no-index',str(args.patch.resolve())],cwd=directory,check=True)
        patched=target.read_text()
    stub=next(n for n in ast.walk(ast.parse(args.stub.read_text())) if isinstance(n,ast.ClassDef) and n.name=='FakePolicy')
    observation=SimpleNamespace(state=torch.zeros(2,16),tokenized_prompt=torch.zeros(2,200,dtype=torch.long),
        tokenized_prompt_mask=torch.ones(2,200,dtype=torch.bool))
    stub_namespace=dict(torch=torch,SimpleNamespace=SimpleNamespace,observation=observation)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[stub],type_ignores=[])),'<synthetic-features>','exec'),stub_namespace)
    base=stub_namespace['FakePolicy'];chosen=0
    common=dict(torch=torch,rl_sampler=sampler,random=SimpleNamespace(randint=lambda lo,hi:chosen),
        preprocess_observation=lambda observation,**kw:observation)
    old_class=type('OriginalSampler',(base,),functions(original,['_predict_train','default_forward'],dict(common)))
    new_class=type('FixedSampler',(base,),functions(patched,['_predict_train','default_forward'],dict(common)))
    cases=[dict(label='selected',index=i) for i in range(10)]
    cases += [dict(label=label,index=i) for label in ['prefix','joint','nft','no_values','ignore_last'] for i in [0,8]]
    results=[]
    with torch.no_grad():
        for case in cases:
            chosen=case['index'];prefix=case['label']=='prefix'
            old,new=old_class(prefix),new_class(prefix)
            for model in [old,new]:
                model.rl_cfg.joint_logprob=case['label']=='joint'
                model.rl_cfg.is_nft=case['label']=='nft'
                model.rl_cfg.ignore_last=case['label']=='ignore_last'
            compute=case['label']!='no_values'
            generators=[torch.Generator().manual_seed(111),torch.Generator().manual_seed(111)]
            old_actions,a=old._predict_train(observation,noise=None,rng=generators[0],compute_values=compute)
            new_actions,b=new._predict_train(observation,noise=None,rng=generators[1],compute_values=compute)
            assert torch.equal(old_actions,new_actions)
            assert torch.equal(a['prev_logprobs'],b['prev_logprobs'])
            assert torch.equal(a['forward_inputs']['chains'],b['forward_inputs']['chains'])
            assert torch.equal(a['forward_inputs']['denoise_inds'],b['forward_inputs']['denoise_inds'])
            assert torch.equal(generators[0].get_state(),generators[1].get_state())
            replay=new.default_forward(b['forward_inputs'],compute_values=compute)
            if case['label'] in ['prefix','joint','nft','no_values']:
                assert torch.equal(a['prev_values'],b['prev_values'])
            if case['label']!='nft':
                assert torch.equal(b['prev_values'][:,0],replay['values'])
            old_gap=float((a['prev_values'][:,0]-replay['values']).abs().max()) if case['label']!='nft' else None
            fixed_gap=float((b['prev_values'][:,0]-replay['values']).abs().max()) if case['label']!='nft' else None
            if case['label']=='selected':assert old_gap>.04 and fixed_gap==0
            results.append(dict(case,old_rollout_replay_value_gap=old_gap,fixed_value_gap=fixed_gap,
                actions_equal=True,chains_equal=True,logprobs_equal=True,denoising_indices_equal=True,RNG_equal=True))
    result=dict(completed=True,device='cpu',cases=len(results),results=results,
        native_source_sha256=digest(args.rlinf/relative),patched_source_sha256=hashlib.sha256(patched.encode()).hexdigest(),
        patch_sha256=digest(args.patch),script_sha256=digest(__file__),sampler_sha256=digest(sampler_path),
        scope='Actual original/patched method ASTs and native sampler/value helpers with synthetic features; no VLA weights or simulator.',
        active_vendor_worktree_modified=False)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(completed=True,cases=len(results),selected_ppo_value_gap=0,actions_chains_logprobs_rng_unchanged=True)))


if __name__=='__main__':main()
