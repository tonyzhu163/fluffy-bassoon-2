"""Allocated-step admission; all receipts are evidence, not new authorization."""
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

import torch

assert sys.platform=='linux' and socket.gethostname().startswith('job-')
plan=json.loads(Path(sys.argv[1]).read_text());phase=sys.argv[2]
out=Path(os.environ['NATIVE_PPO_OUTPUT'])
assert plan['status']=='admitted' and phase in ('optimizer_probe','gradient_probe','qualification','ppo','grpo')
assert plan['human_cumulative_gpu_hours']==50
assert plan['additional_engineering_gpu_hours']==(23 if phase in ('optimizer_probe','gradient_probe','qualification') else 46)
reserved=plan['completed_additional_conservative_gpu_seconds']+sum(plan['gpu_reservations_seconds'].values())
assert reserved<=plan['additional_engineering_gpu_hours']*3600
assert reserved+plan['legacy_gpu_seconds']<=50*3600
assert plan['gpu_reservations_seconds'][phase]>=4*(int(os.environ['TASK_LIMIT'])+60)
if phase=='gradient_probe':
    diagnostic=Path(plan['optimizer_metadata_diagnostic_output'])
    assert json.loads((diagnostic/'EXIT.json').read_text())['exit_code']==0
    comparison=json.loads((diagnostic/'SEMANTIC-COMPARISON.json').read_text())
    assert comparison['status']=='zero_update_optimizer_representation_explained'
    assert len(comparison['ranks'])==4
    for row in comparison['ranks']:
        assert row['groups_order_hyperparameters_exact'] and row['owned_parameter_states_exact_ignoring_only_device']
        for ref in row['files']:
            assert hashlib.sha256((diagnostic/ref['path']).read_bytes()).hexdigest()==ref['sha256']
if phase=='qualification':
    probe=Path(plan['gradient_output'])
    assert json.loads((probe/'EXIT.json').read_text())['exit_code']==0
    for suffix in ('train','restore'):
        assert json.loads((probe/suffix/'RESULT.json').read_text())['status']=='complete'
    for rank in range(4):
        initial=json.loads((probe/'train/fixture-initializer'/f'actor-{rank:02d}.json').read_text())
        assert initial['exact_identity']
        trained=json.loads((probe/'train'/f'probe-rank-{rank:02d}.json').read_text())
        assert trained['status']=='complete' and trained['optimizer_steps']==1
        restored=json.loads((probe/'restore/recovery-signatures'/f'actor-{rank:02d}.json').read_text())
        assert restored['semantic_identity'] and restored['actual']['optimizer_steps']==1
        assert restored['actual']['trainable_parameters_sha256']!=initial['expected']['trainable_parameters_sha256']
        assert json.loads((probe/'train/actor-invariants'/f'rank-{rank:02d}-step-0-after.json').read_text())['unchanged']
        assert json.loads((probe/'restore/recovery-frozen'/f'actor-{rank:02d}.json').read_text())['unchanged']
        for micro in range(8):
            audit=json.loads((probe/'train/gradient-audit'/f'rank-{rank:02d}-micro-{micro:02d}.json').read_text())
            assert audit['all_finite'] and audit['frozen_nonzero_elements']==audit['frozen_optimizer_members']==0
            if micro==7:assert audit['synchronized_boundary'] and audit['frozen_non_none_views']==0

pilot=Path(plan['engineering_pilot'])
boundary=pilot/'sidney-native-two/checkpoints/global_step_1'
receipt=json.loads((boundary/'pilot-boundary.json').read_text())
assert receipt['completed'] and receipt['global_step']==1 and len(receipt['workers'])==12
for rank in range(4):
    state=torch.load(boundary/'pilot'/f'actor-{rank:02d}.pt',map_location='cpu',weights_only=False)
    assert state['optimizer_boundary'] and state['optimizer_steps']==20
    frozen=json.loads((pilot/'actor-invariants'/f'rank-{rank:02d}-step-0-after.json').read_text())
    assert frozen['unchanged']
    replay=json.loads((pilot/'first-replay'/f'rank-{rank:02d}.json').read_text())
    assert replay['passed'] and replay['max_abs_error']['logprobs']<=1e-3
    del state

cache=Path(os.environ['NATIVE_PPO_MODEL'])
for name,expected in plan['initializer_sha256'].items():
    h=hashlib.sha256()
    with (cache/name).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024**2),b''):h.update(block)
    assert h.hexdigest()==expected,name

if phase not in ('qualification','gradient_probe','optimizer_probe'):
    completed=json.loads((pilot/'RESULT.json').read_text())
    assert completed['status']=='complete' and completed['rollout_iterations']==2
    assert json.loads((pilot/'EXIT.json').read_text())['exit_code']==0
    qualification=Path(plan['qualification_output'])
    assert json.loads((qualification/'EXIT.json').read_text())['exit_code']==0
    for suffix in ('train','restore'):
        result=json.loads((qualification/suffix/'RESULT.json').read_text())
        assert result['status']=='complete' and result['qualification_only']
        assert result['optimizer_updates']==1 and result['recovery_only']==(suffix=='restore')
    numerical=json.loads((qualification/'learner-numerics.json').read_text())
    assert numerical['status']=='complete'
    for rank in range(4):
        recovered=json.loads((qualification/'restore/recovery-signatures'/f'actor-{rank:02d}.json').read_text())
        assert recovered['semantic_identity'] and recovered['actual']['optimizer_steps']==1
        assert json.loads((qualification/'restore/recovery-frozen'/f'actor-{rank:02d}.json').read_text())['unchanged']
        grads=json.loads((qualification/'train/paired-gradients'/f'rank-{rank:02d}.json').read_text())
        assert grads['finite'] and grads['frozen_parameters_without_nonzero_gradients'] and not grads['critic_present']
        assert grads['frozen_parameters_excluded_from_optimizer']
        for micro in range(8):
            evidence=json.loads((qualification/'train/gradient-audit'/f'rank-{rank:02d}-micro-{micro:02d}.json').read_text())
            assert evidence['all_finite'] and evidence['frozen_nonzero_elements']==0
            assert evidence['frozen_optimizer_members']==0
            if micro==7:assert evidence['synchronized_boundary'] and evidence['frozen_non_none_views']==0
        group=json.loads((qualification/'train/group-resets'/f'rank-{rank:02d}-reset-0000.json').read_text())
        assert group['group_public_reset_identity'] and group['group_exposed_articulation_identity']
        assert group['raw_public_reference_verified']
        assert len(group['groups'])==8 and all(g['replica_count']==8 for g in group['groups'])

remaining=[]
for row in plan['storage_reservations']:
    root=Path(row['path'])
    used=sum(p.stat().st_blocks*512 for p in root.rglob('*') if p.is_file()) if root.exists() else 0
    cap=int(row['maximum_gib'])*2**30
    assert used<=cap,f'Storage envelope exceeded: {root}'
    if row.get('closed'):
        assert (root/'EXIT.json').is_file()
        increment=0
    else:increment=cap-used
    remaining.append(dict(path=str(root),allocated_bytes=used,remaining_bytes=increment))
volume=os.statvfs(out);free=volume.f_bavail*volume.f_frsize
required=plan['other_owners_and_floor_gib']*2**30+sum(r['remaining_bytes'] for r in remaining)
assert free>=required,f'Insufficient forward storage: {free/2**30:.3f} < {required/2**30:.3f}GiB'
probe=out/'storage-write-probe'
with probe.open('xb') as handle:handle.write(b'allocated-step-write-probe\n');handle.flush();os.fsync(handle.fileno())
probe.unlink()
report=dict(status='complete',phase=phase,gpu_seconds_reserved=reserved,
    cumulative_gpu_seconds_reserved=reserved+plan['legacy_gpu_seconds'],
    free_bytes=free,required_free_bytes=required,storage=remaining,write_fsync=True,
    first_engineering_ppo_checkpoint_adam_steps=20,initializer='original Sidney broad SFT')
(out/'admission.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
