"""Rebind an audited whole-round handoff; CPU metadata only, inside its owner job.

Run with CUDA_VISIBLE_DEVICES='' and an external 90-second timeout. This never
constructs a learner/simulator/budget, signals a process, or changes old files.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import zipfile

RUNTIME = '57016ba08eed79043fd67bf8b085bd6d42e6d766'
CONTROL = 'c62be99615e797fbec108e1f1b5e232e4a591494'
PREDECESSOR = 'e825821a7c2231952bcf1042162f3470bd8ecdea'
LINEAGE = 'db0184089fa88a5f96c153f9b80029bb91b77def'
PARENT_SHA = '1f80c08310dfa4543ec7a748ca6e2bf9aad589cd55978a402210abb0d83bdbc0'
INPUTS = Path('/mowenjie/fluffy-bassoon-2/inputs')


def require(condition, message):
    if not condition:
        raise ValueError('Owner manifest rebind: ' + message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def bound(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def read_bound(ref):
    require(Path(ref['path']).is_absolute() and sha(ref['path']) == ref['sha256'], 'bound artifact hash differs')
    return json.loads(Path(ref['path']).read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def clean_pin(root, pin):
    require(root.is_absolute() and root.is_dir(), 'missing absolute Git worktree')
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() == pin,
            'Git worktree pin differs')
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip(),
            'Git worktree is dirty')


def write_exclusive(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return bound(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('expected-hostname', 'method', 'owner-receipt-sha256', 'artifact-commit'):
        parser.add_argument('--' + name, required=True)
    for name in ('owner-receipt', 'runtime-root', 'control-root', 'new-manifest', 'new-receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    require(sys.platform == 'linux' and socket.gethostname().startswith('job-')
            and socket.gethostname() == args.expected_hostname, 'not the exact owned Linux job hostname')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA must be hidden before importing torch')
    require(args.method in ('ppo', 'ppo_reset'), 'only the two trained PPO methods are eligible')
    artifact_root = Path(__file__).resolve().parents[2]
    clean_pin(artifact_root, args.artifact_commit)
    clean_pin(args.runtime_root, RUNTIME)
    clean_pin(args.control_root, CONTROL)
    for path in (args.owner_receipt, args.new_manifest, args.new_receipt):
        require(path.is_absolute(), 'all input/output paths must be absolute')
    destinations = [args.new_manifest.resolve(), args.new_receipt.resolve()]
    require(len(set(destinations)) == 2, 'manifest and receipt destinations must differ')
    for path in destinations:
        require(path.is_relative_to(INPUTS.resolve()) and not path.exists(), 'destination must be fresh under remote inputs')

    control_path = args.control_root / 'outputs/robotwin-trained-handoff-control-20261030-r1/control.py'
    spec = importlib.util.spec_from_file_location('qualified_owner_rebind_control', control_path)
    control = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = control
    spec.loader.exec_module(control)  # Import only; never invoke the signal/CLI entry points.
    owner_ref = dict(path=str(args.owner_receipt.resolve()), sha256=args.owner_receipt_sha256)
    audit = read_bound(owner_ref)
    require(audit.get('schema') == 'robotwin_owner_handoff_control_v1'
            and audit.get('status') == 'paused_child_complete_handoff_verified'
            and audit.get('hostname') == args.expected_hostname and audit.get('method') == args.method
            and audit.get('source_commit') == PREDECESSOR and audit.get('controller_commit') == CONTROL
            and audit.get('parent_left_stopped') is True and audit.get('source_children_inactive') is True
            and audit.get('permission_to_start_new_writer') is False,
            'owner boundary receipt is not a current completed c62 handoff')
    require(audit.get('receipt_path') == owner_ref['path']
            and audit['controller'] == bound(audit['controller']['path'])
            and audit['controller']['sha256'] == sha(control_path), 'owner receipt/controller binding differs')
    old_output = Path(audit['output']).resolve()
    require(args.owner_receipt.resolve().parent == old_output / 'control'
            and all(not path.is_relative_to(old_output) for path in destinations), 'receipt/output ownership differs')
    process = control.proc(audit['supervisor_pid'])
    require(process is not None and len(process['argv']) >= 2, 'owned supervisor is absent')
    argv = process['argv']
    owner = control.Owner(args.expected_hostname, audit['supervisor_pid'], audit['supervisor_start_ticks'],
        Path(argv[1]), control.SUPERVISOR_SHA, Path(argv[0]), audit['run_id'], args.method,
        PREDECESSOR, Path(control.option(argv, '--code-root')), old_output)

    def stopped_owner():
        current = control.check_parent(owner)
        require(current['state'] in ('T', 't')
                and all(row['state'] == 'Z' for row in control.direct_children(owner)),
                'supervisor is not stopped or its source writer is still active')
        return current

    stopped_owner()
    handoff = audit['handoff']
    require(handoff.get('round_boundary') is True and handoff.get('pending_ppo_trials') == 0
            and handoff.get('checkpoint_crc_ok') is True and handoff.get('all_executed_actions_retained') is True
            and handoff.get('raw_json_index_hashes_verified') is True
            and handoff.get('referenced_artifacts_present') is True, 'owner full boundary proof is incomplete')
    old_ref = handoff['manifest']; old = read_bound(old_ref)
    checkpoint_ref = old['checkpoint']
    require(checkpoint_ref == handoff['checkpoint'] and sha(checkpoint_ref['path']) == checkpoint_ref['sha256'],
            'owner checkpoint binding differs')
    require(old.get('execution_source_commit') == PREDECESSOR and old.get('lineage_source_commit') == LINEAGE
            and old.get('method') == args.method and old.get('kind') == 'whole_trial_boundary'
            and old.get('trial') is None and old.get('prefix') == []
            and old['expected'].get('pending_rollout_digests') == [], 'old manifest is not the eligible whole-round handoff')
    parent_path = Path(control.option(argv, '--parent-manifest')).resolve()
    parent_ref = bound(parent_path)
    require(parent_ref['sha256'] == PARENT_SHA, 'original immutable parent differs')
    parent = read_bound(parent_ref); worker = parent['workers'][args.method]
    ledger_root = Path(handoff['ledger_root'])
    before, before_sha = control.ledger_snapshot(ledger_root)
    require(before_sha == handoff['ledger_snapshot_sha256']
            and before['journal_sha256'] == handoff['ledger_journal_sha256'], 'canonical ledger advanced after owner audit')
    with zipfile.ZipFile(checkpoint_ref['path']) as archive:
        require(archive.testzip() is None, 'unchanged checkpoint CRC differs')
    sys.path.insert(0, str(args.runtime_root / 'src'))
    import torch
    import fluffy_bassoon.meta_rl.robotwin_trial_resume as resume_module
    from fluffy_bassoon.meta_rl.robotwin_trained_parallel import (
        validate_lineage_parent, validate_trained_parallel_boundary)
    require(Path(resume_module.__file__).resolve() == args.runtime_root.resolve()
            / 'src/fluffy_bassoon/meta_rl/robotwin_trial_resume.py', 'loader import did not use qualified runtime')
    checkpoint = torch.load(checkpoint_ref['path'], map_location='cpu', weights_only=False)
    contract = checkpoint['contract']; state = checkpoint['states'][args.method]; learner = state['learner']
    trials = state['completed_source_trials']; iterations = state['completed_source_iterations']
    require(0 < iterations < 100 and trials == 6 * iterations
            and state['schedule']['trials'] == trials and learner['updates'] == iterations
            and learner['optimizer_steps'] == 5 * iterations and not state.get('pending_rollouts')
            and not state.get('selected_trial_recovery') and learner['primitive_steps'] == state['source_steps']
            and sum(state['task_steps'].values()) == state['source_steps'], 'trained whole-round counters differ')
    require(handoff['completed_source_trials'] == trials and handoff['completed_source_iterations'] == iterations
            and handoff['source_steps'] == state['source_steps'], 'owner and checkpoint counters differ')
    control.validate_ledger_header(before, worker, args.method, contract)
    validate_lineage_parent(parent_path, PARENT_SHA, checkpoint=checkpoint,
        campaign_root=ledger_root, native_episodes=worker['native_episodes'])
    summary = read_bound(handoff['summary']); frozen = read_bound(handoff['frozen_motor'])
    require(summary['status'] == 'wall_budget_stop' and canonical(summary['contract']) == canonical(contract)
            and frozen.get('unchanged') and frozen.get('before') == frozen.get('after') and frozen.get('before'),
            'bound summary/frozen motor differs')
    load = resume_module.load_trial_continuation
    original = load(old_ref['path'], old_ref['sha256'], checkpoint_path=checkpoint_ref['path'], checkpoint=checkpoint,
        execution_commit=PREDECESSOR, lineage_commit=LINEAGE)
    trained_boundary = validate_trained_parallel_boundary(original, checkpoint,
        manifest_path=old_ref['path'], campaign_root=ledger_root)
    new = deepcopy(old); new['execution_source_commit'] = RUNTIME
    require({k: v for k, v in old.items() if k != 'execution_source_commit'}
            == {k: v for k, v in new.items() if k != 'execution_source_commit'}, 'another manifest field changed')
    new_ref = write_exclusive(args.new_manifest, new)
    admitted = load(new_ref['path'], new_ref['sha256'], checkpoint_path=checkpoint_ref['path'], checkpoint=checkpoint,
        execution_commit=RUNTIME, lineage_commit=LINEAGE)
    require(original.method == admitted.method == args.method and original.trial is None and admitted.trial is None
            and original.receipt['checkpoint'] == admitted.receipt['checkpoint'] == checkpoint_ref
            and original.receipt['ledger_journal_sha256'] == admitted.receipt['ledger_journal_sha256'],
            'qualified old/new loader results differ')
    old_roots = control.corpus_chain(old_ref); new_roots = control.corpus_chain(new_ref)
    require(old_roots == new_roots, 'ordered corpus ancestry changed')
    stopped_owner()
    _, after_sha = control.ledger_snapshot(ledger_root)
    require(after_sha == before_sha and sha(checkpoint_ref['path']) == checkpoint_ref['sha256']
            and sha(owner_ref['path']) == owner_ref['sha256'], 'old artifacts or canonical ledger changed during rebinding')
    receipt = dict(schema='robotwin_executor_handoff_rebinding_v1',
        created_at_utc=datetime.now(timezone.utc).isoformat(), hostname=args.expected_hostname, method=args.method,
        artifact=bound(__file__), artifact_commit=args.artifact_commit, qualified_runtime_commit=RUNTIME,
        predecessor_manifest=old_ref, admitted_manifest=new_ref,
        predecessor_execution_source_commit=PREDECESSOR, admitted_execution_source_commit=RUNTIME,
        lineage_source_commit=LINEAGE, original_parent=parent_ref, owner_boundary_audit=owner_ref,
        unchanged_checkpoint=checkpoint_ref, only_existing_field_changed='execution_source_commit',
        all_other_existing_fields_and_artifact_refs_equal=True, original_loader_validation_passed=True,
        admitted_loader_validation_passed=True, qualified_trained_boundary=trained_boundary,
        corpus_chain_reader_commit=CONTROL,
        old_corpus_roots=old_roots, new_corpus_roots=new_roots, source_actions_added=0,
        scientific_contract_changed=False, canonical_ledger_modified=False,
        canonical_ledger_snapshot_sha256=after_sha, permission_to_start_new_writer=False)
    receipt_ref = write_exclusive(args.new_receipt, receipt)
    print(json.dumps(dict(status='validated_rebinding_only', manifest=new_ref, receipt=receipt_ref), indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
