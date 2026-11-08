"""Explicit fixed-DEV endpoint support; no confirmation-bank reuse or training."""
import json
from pathlib import Path

from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha

ROOT=Path(__file__).resolve().parent


def selected_steps(plan):
    step=plan.get('selected_training_iteration',2)
    extended=plan.get('extended_source_curve_readout',False)
    assert type(extended) is bool and type(step) is int
    if extended:
        assert plan['algorithm']=='grpo' and plan['bank']=='dev'
        assert step in (6,10)
        assert plan['source_SFT_reference_state']==str(Path(plan['checkpoints']['0']['path']).parents[1]/'state')
    else:
        assert step in (1,2)
    assert set(plan['checkpoints'])=={'0',str(step)}
    return (0,step)


def allowed_steps(cfg):
    if cfg.native_pilot.get('extended_source_curve_readout',False):
        step=int(cfg.native_pilot.readout_selected_iteration)
        assert step in (6,10) and cfg.native_pilot.readout_bank=='dev'
        return (0,step)
    return (0,1,2)


def original_sft_restore(cfg,schedule,state,directory):
    """Permit the original C0 schedule only for explicitly read-only curve DEV."""
    if not cfg.native_pilot.get('extended_source_curve_readout',False):return False
    boundary=Path(directory).parent.name
    if boundary!='global_step_0':return False
    prefix=ROOT/'MOTUS-TRAIN-CASES.json'
    assert cfg.native_pilot.readout_bank=='dev'
    assert schedule.data['prefix_manifest_sha256']==sha(prefix)==state['manifest_sha256']
    assert state['next_reset_ordinal']==0
    assert schedule.data['groups'][:256]==json.loads(prefix.read_bytes())['groups']
    return True


def result_steps(result):
    steps=result['checkpoints']
    if result.get('extended_source_curve_readout',False):
        assert result['algorithm']=='grpo' and result['bank']=='dev'
        assert result['fixed_DEV_only'] and result['new_confirmation'] is False
        assert steps in ([0,6],[0,10])
        assert set(result['input_checkpoints'])==set(map(str,steps))
    else:assert steps in ([0,1],[0,2])
    return steps


def verify_one_restore(step,row,source=None):
    """Use actual recorded GPU signatures and, when supplied, the full cold input."""
    from profile_checkpoint import verify
    assert row['step']==step
    if source is not None:
        path=Path(source['path'])
        assert sha(path/'MANIFEST.json')==source['manifest_sha256']
        manifest=json.loads((path/'MANIFEST.json').read_bytes())
        assert manifest['global_step']==step
        verify(path,manifest)
    workers=row['workers']
    assert len(workers)==24
    assert {(r['role'],r['rank']) for r in workers}=={(role,rank) for role in ('actor','rollout','env') for rank in range(8)}
    for worker in workers:
        assert worker['restored']
        if source is not None:
            assert worker['snapshot_sha256']==manifest['files'][f"pilot/{worker['role']}-{worker['rank']:02d}.pt"]['sha256']
    before=sorted(row['actor_before'],key=lambda r:r['rank'])
    after=sorted(row['actor_after'],key=lambda r:r['rank'])
    assert [r['rank'] for r in before]==[r['rank'] for r in after]==list(range(8))
    for a,b in zip(before,after,strict=True):
        assert a['optimizer_steps']==b['optimizer_steps']==15*step
        clean=lambda item:{k:v for k,v in item.items() if k!='optimizer_sha256'}
        assert clean(a)==clean(b)
        if source is not None:
            expected=json.loads((path/'pilot'/f"motus-actor-signature-{a['rank']:02d}.json").read_bytes())
            assert clean(a)==clean(dict(expected,rank=a['rank']))
    return dict(step=step,workers=24,actors=8,cold_source_verified=source is not None)


def verify_readout_restores(result):
    """Audit recorded actor invariants; cold-check source inputs for new endpoints."""
    steps=result_steps(result);extended=result.get('extended_source_curve_readout',False)
    restores=result['restores'];assert [r['step'] for r in restores]==steps
    rows=[verify_one_restore(step,row,result['input_checkpoints'][str(step)] if extended else None)
        for step,row in zip(steps,restores,strict=True)]
    return dict(all24_worker_restores_each_policy_verified=True,
        all8_actor_before_after_owned_Adam_scheduler_scaler_parameters_RNG_exact=True,
        new_endpoint_source_checkpoints_cold_verified=bool(extended),boundaries=rows)
