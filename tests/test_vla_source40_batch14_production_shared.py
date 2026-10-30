"""Real shared-PFS recovery integration, with no policy/VLA construction or SGD.

Absent shared paths skip in an ordinary local checkout. The dedicated allocated
CPU launcher sets REQUIRE=1, so unavailable or changed PFS evidence fails there.
"""
import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl import vla_source40_batch14_production as prod
from fluffy_bassoon.meta_rl.vla_rl2 import RL2Config
from fluffy_bassoon.meta_rl.vla_source40_batch14_warmstart import legacy_fingerprint
from fluffy_bassoon.meta_rl.vla_source40_ppo import physical_charge


ROOT=Path('/mowenjie/fluffy-bassoon-2/outputs/libero40-batch14-warm-qualification-20261031-r1')
RECOVERY_SHA='c689a2ddb190510ba09452ad9c755446bf605351d22b41017c75187834e90ee0'
PRODUCER='376415e6915399436cb3fea214c50bd3e41f32d5'


def test_actual_qualified_full_recovery_migrates_losslessly_without_model_or_optimizer_construction(tmp_path):
    recovery=ROOT/'experiment/learner/latest.pt'
    if not recovery.is_file():
        if os.environ.get('FLUFFY_REQUIRE_PPO_SHARED_CHECKPOINT')=='1':
            pytest.fail('Dedicated allocated integration requires the preserved qualified shared-PFS recovery')
        pytest.skip('Shared Baige PFS absent; actual checkpoint integration not exercised here')
    assert prod.sha(recovery)==RECOVERY_SHA
    original=torch.load(recovery,map_location='cpu',weights_only=False)
    before=legacy_fingerprint(original)
    assert original['contract']['source']['source_commit']==PRODUCER
    assert original['progress']['iteration']==1
    assert original['runtime']['task_cycle']['cursor']==14
    assert original['runtime']['task_cycle']['remaining']==0
    assert {float(row['step']) for row in original['optimizer']['state'].values() if 'step' in row}=={478.}
    history=original['contract']['source']['warm_start']
    proposal=prod.make_plan(history['path'],history['sha256'],'full',native_root=ROOT)
    plan_path=tmp_path/'production-plan.json';prod.write(plan_path,proposal)
    plan_sha=prod.sha(plan_path)
    assert prod.load_plan(plan_path,plan_sha)==proposal
    assert proposal['prior_source_resource_actions']==205236
    assert proposal['maximum_native_stream_actions']==774036
    assert proposal['maximum_new_actions']==764764
    assert proposal['native_parent']['source_contract_sha256']==prod.sha(ROOT/'experiment/source-contract.json')
    # Only explicit phase metadata changes; constructing RL2Config is a dataclass,
    # not constructing a policy. No policy forward or optimizer object is used.
    source=copy.deepcopy(original['contract']['source'])
    source.update(source_commit=os.environ.get('FLUFFY_PPO_CPU_SOURCE','a'*40),
        updates_target=100,maximum_total_actions=774036)
    source=prod.source_contract(source,plan_path,plan_sha,actions=774036,updates=100,mode='full',seed=2)
    config=RL2Config(**original['contract']['config'])
    learner=tmp_path/'new-immutable-experiment/learner';learner.parent.mkdir()
    prod.prepare(learner,plan_path,plan_sha,source,config,copy.deepcopy(original['runtime']['task_cycle']),
        updates=100,actions=774036,seed=2,mode='full')
    restored=torch.load(learner/'latest.pt',map_location='cpu',weights_only=False)
    proof=prod.read(learner/'production-migration.json')
    after=legacy_fingerprint(restored)
    checks=dict(actual_pfs_recovery=True,exact_parent_checksum=prod.sha(recovery)==RECOVERY_SHA,
        every_non_contract_leaf=before==after==proof['non_contract_sha256_before']==proof['non_contract_sha256_after'],
        cursor14=restored['runtime']['task_cycle']==original['runtime']['task_cycle'],
        adam478={float(row['step']) for row in restored['optimizer']['state'].values() if 'step' in row}=={478.},
        pending_state=legacy_fingerprint({'progress':restored['progress']})==legacy_fingerprint({'progress':original['progress']}),
        typed_motor_rng=(restored['runtime']['motor_rng']['format']=='jax_motor_key_v1'
            and restored['runtime']['motor_rng']['implementation']=='threefry2x32'
            and np.asarray(restored['runtime']['motor_rng']['key_data']).dtype==np.uint32
            and np.asarray(restored['runtime']['motor_rng']['key_data']).shape==(2,)),
        optimizer_steps_zero=proof['restore_optimizer_steps']==0,
        copied_physical_prefix=physical_charge(learner.parent/'execution-journal')['total_action_charge']==9272,
        original_bytes_unchanged=prod.sha(recovery)==RECOVERY_SHA,
        explicit_proposed_phase=restored['contract']['updates_target']==100
            and restored['contract']['maximum_total_actions']==774036)
    assert all(checks.values()),checks
    # Exact continuation of this new identity must not rewrite any saved state.
    latest_sha=prod.sha(learner/'latest.pt')
    prod.prepare(learner,plan_path,plan_sha,source,config,{},updates=100,actions=774036,seed=2,mode='full')
    assert prod.sha(learner/'latest.pt')==latest_sha
    receipt=dict(format='source40_actual_pfs_ppo_production_migration_test_v1',passed=True,checks=checks,
        parent_root=str(ROOT),parent_source_commit=PRODUCER,parent_recovery_sha256=RECOVERY_SHA,
        plan_path=str(plan_path),plan_sha256=plan_sha,maximum_native_stream_actions=774036,
        maximum_new_actions=764764,prior_source_resource_actions=205236,final_source_validation_actions=30000,
        before_non_contract_sha256=before,after_non_contract_sha256=after,
        migrated_checkpoint_path=str(learner/'latest.pt'),migrated_checkpoint_sha256=latest_sha,
        migration_proof_path=str(learner/'production-migration.json'),migration_proof_sha256=prod.sha(learner/'production-migration.json'),
        copied_journal_files={k:len(v) for k,v in proof['copied_journals'].items()},
        test_scope='Actual shared checkpoint deserialization/metadata rewrite/copy; no model construction, forward, simulator or SGD',
        production_run_created=False)
    output=os.environ.get('FLUFFY_PPO_CPU_RECEIPT_DIR')
    if output:prod.write(Path(output)/'shared-checkpoint-integration.json',receipt)
