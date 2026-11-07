"""Read actual rejected layouts and reproduce native offload's defaultdict key."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import motus_teacher_entry
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha,write
from optimizer_evidence import evidence
from optimizer_semantics import semantic_digest,without_devices

BASE=Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')
OLD=Path('/mowenjie/fluffy-bassoon-2/sources/robotwin-restoration-20261108-camera01/outputs/robotwin-sidney-matched-baselines-20261106-r1/optimizer_semantics.py')


def main():
    assert os.uname().sysname=='Linux' and os.environ['CUDA_VISIBLE_DEVICES']==''
    assert not torch.cuda.is_initialized()
    assert sha(OLD)=='5c70ed49c2f8fefb85328d5a743116fd54583e89a314fb1c7b963dd186ab99bb'
    spec=importlib.util.spec_from_file_location('original_strict_guard',OLD)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    state=BASE/'motus-source-ppo-dev02/state'
    result=json.loads((state/'RESULT.json').read_bytes())
    assert len(result['panels'])==4 and all(r['step']==0 for r in result['panels'])
    rows=[]
    for after_path in sorted((state/'readout-actor').glob('policy-0-after-rank-*.json')):
        after=json.loads(after_path.read_bytes());rank=after['rank']
        before_path=state/'readout-actor'/f'policy-0-before-rank-{rank:02d}.json'
        before=json.loads(before_path.read_bytes())
        assert before['optimizer_steps']==after['optimizer_steps']==0
        rejected=None
        try:old.semantic_digest(after)
        except AssertionError as error:rejected=str(error)
        assert rejected is not None
        assert old.semantic_digest(before)==semantic_digest(before)==semantic_digest(after)
        names=[];owned=0
        for name,entry in after['parameters'].items():
            prior=before['parameters'][name]
            if entry['local_nonempty_parameter']:
                owned+=1
                assert without_devices(entry)==without_devices(prior),name
            elif entry['state_present'] and entry['state']=={}:
                assert entry['registered_model_parameter'] and entry['requires_grad']
                assert entry['parameter_numel']==0 and entry['parameter_shape']==[0]
                names.append(name)
        assert names
        rows.append(dict(rank=rank,original_guard_failure=rejected,owned_parameters_exact=owned,
            empty_unowned_names=names,before_sha256=sha(before_path),after_sha256=sha(after_path)))
    assert rows and len({r['rank'] for r in rows})==len(rows)
    import rlinf.hybrid_engines.fsdp.strategy.fsdp as native
    model=torch.nn.Module();model.register_parameter('owned',torch.nn.Parameter(torch.tensor([1.,2.])))
    model.register_parameter('empty',torch.nn.Parameter(torch.empty(0)))
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-6)
    optimizer.state[model.owned]=dict(step=torch.tensor(1.),exp_avg=torch.tensor([.2,.3]),exp_avg_sq=torch.tensor([.04,.09]))
    actor=SimpleNamespace(model=model,optimizer=optimizer,_rank=0,optimizer_steps=1,is_optimizer_offloaded=False)
    before=evidence(actor);assert model.empty not in optimizer.state
    with patch.object(native,'clear_memory',lambda:None):
        native.FSDPStrategy.offload_optimizer(None,optimizer)
        assert model.empty in optimizer.state and optimizer.state[model.empty]=={}
        native.FSDPStrategy.onload_optimizer(None,optimizer,torch.device('cpu'))
    after=evidence(actor)
    assert semantic_digest(before)==semantic_digest(after)
    rejected=False
    try:old.semantic_digest(after)
    except AssertionError:rejected=True
    assert rejected and not torch.cuda.is_initialized()
    output=BASE/'motus-readout-empty-state-cpu01';output.mkdir()
    write(output/'RESULT.json',dict(status='passed_actual_rejected_layout_and_native_offload_reference',
        source_commit=os.environ['TASK_SOURCE'],actual_rejected_layouts=rows,
        observed_context='Read-only SFT checkpoint0 restore, weight synchronization,64 ODE evaluation episodes; no SGD.',
        all_observed_owned_Adam_and_group_metadata_exact=True,
        native_offload_and_onload_actual_functions_used=True,fixture_device='cpu',
        native_memory_cleanup_mocked=True,
        native_source_sha256=sha(Path(native.__file__)),old_guard_sha256=sha(OLD),
        native_defaultdict_materializes_empty_unowned_entry=True,new_rule_preserves_previous_semantic_digests=True,
        VLA_model_or_simulator_constructed=False,GPU_seconds=0,learning_gain_claim=False))
    print('EMPTY_STATE_AUDIT_PASSED',sha(output/'RESULT.json'),len(rows),flush=True)


if __name__=='__main__':main()
