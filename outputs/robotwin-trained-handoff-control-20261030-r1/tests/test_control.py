"""Actual Linux fake-process regressions. Run only in an allocated CPU job."""
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import pytest

MODULE=Path(__file__).resolve().parents[1]/'control.py'
spec=importlib.util.spec_from_file_location('robotwin_owner_control',MODULE)
control=importlib.util.module_from_spec(spec);sys.modules[spec.name]=control;spec.loader.exec_module(control)
pytestmark=pytest.mark.skipif(sys.platform!='linux' or not hasattr(os,'pidfd_open')
    or not hasattr(signal,'pidfd_send_signal'),reason='Actual allocated Linux pidfd regressions')

DRIVER=r'''
import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--output',type=Path);p.add_argument('--fake-delay',type=float)
a,_=p.parse_known_args();a.output.mkdir(parents=True)
(a.output/'started.json').write_text(json.dumps({'pid':os.getpid()}))
time.sleep(a.fake_delay)
(a.output/'summary.json').write_text(json.dumps({'status':'wall_budget_stop','closed':True}))
(a.output/'finished').write_text('complete')
'''
SUPERVISOR=r'''
import argparse,json,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser()
for name in ('source-commit','run-id','method','code-root','output'):p.add_argument('--'+name)
p.add_argument('--fake-delay',type=float);a=p.parse_args()
root=Path(a.output);root.mkdir()
for n in range(1,50):
 d=root/'slices'/f'{n:04d}';d.mkdir(parents=True)
 command=[sys.executable,str(Path(a.code_root)/'scripts/diagnostics/robotwin_noise_compare.py'),
 '--output',str(d/'experiment'),'--source-commit',a.source_commit,
 '--lineage-source-commit','db0184089fa88a5f96c153f9b80029bb91b77def',
 '--methods',a.method,'--max-source-trials-per-process','6','--recycle-at-round-boundary',
 '--fake-delay',str(a.fake_delay)]
 (d/'launch.json').write_text(json.dumps({'command':command}))
 with (d/'console.log').open('w') as f:
  child=subprocess.Popen(command,stdout=f,stderr=f,start_new_session=True)
  (d/'child-pid').write_text(str(child.pid))
  code=child.wait()
 (d/'exit.json').write_text(json.dumps({'output':str(d/'experiment'),'exit_code':code}))
'''


def wait_until(predicate,seconds=5):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        result=predicate()
        if result:return result
        time.sleep(.01)
    raise AssertionError('Owned fake process did not reach its expected state')


@pytest.fixture
def owned(tmp_path):
    running=[]
    def start(delay=.8):
        root=tmp_path/f'fake-{len(running)}';root.mkdir();code=root/'code'
        driver=code/'scripts/diagnostics/robotwin_noise_compare.py';driver.parent.mkdir(parents=True);driver.write_text(DRIVER)
        script=root/'supervise.py';script.write_text(SUPERVISOR);output=root/'owned-output'
        command=[sys.executable,str(script),'--source-commit','e825821a7c2231952bcf1042162f3470bd8ecdea',
            '--run-id','owned-output','--method','ppo','--code-root',str(code),'--output',str(output),
            '--fake-delay',str(delay)]
        parent=subprocess.Popen(command,start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        wait_until(lambda:(output/'slices/0001/experiment/started.json').exists())
        snapshot=control.proc(parent.pid)
        owner=control.Owner(socket.gethostname(),parent.pid,snapshot['start_ticks'],script,control.sha(script),
            Path(sys.executable),'owned-output','ppo','e825821a7c2231952bcf1042162f3470bd8ecdea',code,output,False)
        running.append((parent,owner));return owner
    yield start
    for parent,owner in running:
        if parent.poll() is None:
            fd=os.pidfd_open(parent.pid,0)
            try:signal.pidfd_send_signal(fd,signal.SIGKILL,None,0)
            finally:os.close(fd)
            parent.wait(timeout=5)
        # Fake finite children have their own sessions. Scope cleanup to this
        # unique tmp driver path and recheck start identity through pidfd.
        for pid in control.pids():
            value=control.proc(pid)
            if value is None or len(value['argv'])<2 or Path(value['argv'][1])!=owner.code_root/'scripts/diagnostics/robotwin_noise_compare.py':continue
            try:
                fd=os.pidfd_open(pid,0);again=control.proc(pid)
                if again is not None and again['start_ticks']==value['start_ticks']:
                    signal.pidfd_send_signal(fd,signal.SIGTERM,None,0)
                os.close(fd)
            except ProcessLookupError:pass


def fake_audit(experiment,owner,children):
    value=json.loads((experiment/'summary.json').read_text())
    assert value=={'status':'wall_budget_stop','closed':True}
    assert (experiment/'finished').exists()
    return dict(experiment=str(experiment),fake_only=True)


def test_real_pause_allows_finite_child_to_finish_and_prevents_next(owned):
    owner=owned()
    result=control.pause_and_drain(owner,wait_seconds=4,poll_seconds=.01,
        audit=lambda scope,children:control.select_latest(scope,children,auditor=fake_audit))
    assert result['status']=='paused_child_complete_handoff_verified'
    assert control.proc(owner.pid)['state']=='T'
    assert (owner.output/'slices/0001/experiment/finished').exists()
    time.sleep(.15)
    assert not (owner.output/'slices/0002').exists()
    assert [row['signal'] for row in result['signals']]==['SIGSTOP']
    assert result['source_children_inactive'] and not result['permission_to_start_new_writer']
    assert result['handoff']['experiment'].endswith('0001/experiment')


@pytest.mark.parametrize('field',['hostname','start_ticks','run_id','method','source_commit','code_root','output'])
def test_wrong_scope_is_rejected_without_signalling_another_process(owned,field):
    owner=owned(delay=2.)
    value=getattr(owner,field)
    wrong=(value+1 if isinstance(value,int) else value/'wrong' if isinstance(value,Path) else value+'-wrong')
    with pytest.raises(control.ControlError):control.pause_and_drain(replace(owner,**{field:wrong}),wait_seconds=.1)
    assert control.proc(owner.pid)['state'] not in ('T','t')
    child=control.direct_children(owner)[0]
    assert child['state'] not in ('T','t')


def test_wrong_pid_rejects_foreign_process_and_preserves_owned_supervisor(owned):
    owner=owned(delay=2.)
    foreign=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)'],start_new_session=True)
    try:
        snapshot=control.proc(foreign.pid)
        with pytest.raises(control.ControlError):
            control.pause_and_drain(replace(owner,pid=foreign.pid,start_ticks=snapshot['start_ticks']),wait_seconds=.1)
        assert control.proc(foreign.pid)['state'] not in ('T','t')
        assert control.proc(owner.pid)['state'] not in ('T','t')
    finally:foreign.terminate();foreign.wait(timeout=5)


def test_timeout_rollback_revalidates_and_resumes_only_our_supervisor(owned):
    owner=owned(delay=1.)
    with pytest.raises(control.ChildWaitTimeout) as failure:
        control.pause_and_drain(owner,wait_seconds=.05,poll_seconds=.01,rollback_on_timeout=True)
    receipt=failure.value.receipt
    assert receipt['status']=='timeout_rolled_back' and receipt['rollback_verified']
    assert [row['signal'] for row in receipt['signals']]==['SIGSTOP','SIGCONT']
    wait_until(lambda:control.proc(owner.pid)['state'] not in ('T','t'))
    assert control.direct_children(owner)[0]['state'] not in ('T','t')
    wait_until(lambda:(owner.output/'slices/0002/experiment/started.json').exists())


def test_default_timeout_leaves_parent_stopped_without_signalling_source(owned):
    owner=owned(delay=1.)
    with pytest.raises(control.ChildWaitTimeout) as failure:
        control.pause_and_drain(owner,wait_seconds=.05,poll_seconds=.01)
    assert failure.value.receipt['parent_left_stopped']
    assert [row['signal'] for row in failure.value.receipt['signals']]==['SIGSTOP']
    assert control.proc(owner.pid)['state']=='T'
    assert control.direct_children(owner)[0]['state'] not in ('T','t')


def test_newest_missing_proof_never_falls_back_to_stale_boundary(owned):
    owner=owned()
    receipt=control.pause_and_drain(owner,wait_seconds=4,poll_seconds=.01)
    first=owner.output/'slices/0001'
    (first/'exit.json').write_text(json.dumps({'output':str(first/'experiment'),'exit_code':0}))
    latest=owner.output/'slices/0002';latest.mkdir();(latest/'console.log').write_text('started but no durable proof')
    with pytest.raises(control.ControlError,match='do not select an older cursor'):
        control.select_latest(owner,receipt['owned_children'],auditor=fake_audit)


def test_pause_before_popen_can_select_latest_closed_but_not_first_stale(owned):
    owner=owned()
    receipt=control.pause_and_drain(owner,wait_seconds=4,poll_seconds=.01)
    second=owner.output/'slices/0002';(second/'experiment').mkdir(parents=True)
    (second/'experiment/summary.json').write_text(json.dumps({'status':'wall_budget_stop','closed':True}))
    (second/'experiment/finished').write_text('complete')
    (second/'exit.json').write_text(json.dumps({'output':str(second/'experiment'),'exit_code':0}))
    third=owner.output/'slices/0003';third.mkdir();(third/'launch.json').write_text('{}')
    result=control.select_latest(owner,[],auditor=fake_audit)
    assert result['experiment'].endswith('0002/experiment')
    assert result['inert_unstarted_slices']==[str(third)]


def test_stale_start_identity_cannot_send_rollback_continue(owned):
    owner=owned()
    control.pause_and_drain(owner,wait_seconds=4,poll_seconds=.01)
    fd=os.pidfd_open(owner.pid,0);sent=[]
    try:
        with pytest.raises(control.ControlError,match='ownership mismatch'):
            control.checked_signal(replace(owner,start_ticks=owner.start_ticks+1),fd,signal.SIGCONT,sent)
    finally:os.close(fd)
    assert sent==[] and control.proc(owner.pid)['state']=='T'


def test_wait_polling_does_not_repeat_git_subprocesses(owned,monkeypatch):
    owner=replace(owned(delay=.6),check_git=True);calls=[]
    def git(command,**kwargs):
        calls.append(command)
        return owner.source_commit+'\n' if command[-2:]==['rev-parse','HEAD'] else ''
    monkeypatch.setattr(control.subprocess,'check_output',git)
    result=control.pause_and_drain(owner,wait_seconds=4,poll_seconds=.01)
    assert result['status']=='paused_child_complete_handoff_verified'
    assert len(calls)<=12 # Full pre-signal/final checks, independent of polling count.
    assert result['wait_observation_count']>5
    assert control.proc(owner.pid)['state']=='T'


def test_external_stop_before_signal_is_rejected_without_claiming_pause(owned):
    owner=owned(delay=2.);fd=os.pidfd_open(owner.pid,0);sent=[]
    try:
        # Test-owned fake only: reproduce an external stop after initial admission.
        signal.pidfd_send_signal(fd,signal.SIGSTOP,None,0)
        wait_until(lambda:control.proc(owner.pid)['state']=='T')
        with pytest.raises(control.ControlError,match='already stopped'):
            control.checked_signal(owner,fd,signal.SIGSTOP,sent)
    finally:os.close(fd)
    assert sent==[] and control.proc(owner.pid)['state']=='T'
    assert control.direct_children(owner)[0]['state'] not in ('T','t')


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)+'\n');return control.bound(path)


def ancestry(tmp_path):
    old=tmp_path/'old/corpus';new=tmp_path/'new/corpus'
    old_contract=write_json(old/'contract.json',{'source_commit':control.LINEAGE})
    old_ref=write_json(tmp_path/'old/handoff.json',{'corpus_root':str(old),'corpus_contract':old_contract})
    new_contract=write_json(new/'contract.json',{'source_commit':control.LINEAGE,
        'execution_source_commit':control.EXECUTOR,'resume_trial_manifest_sha256':old_ref['sha256']})
    new_ref=write_json(tmp_path/'new/handoff.json',{'corpus_root':str(new),'corpus_contract':new_contract})
    migration=dict(lineage_source_commit=control.LINEAGE,execution_source_commit=control.EXECUTOR,
        manifest_path=old_ref['path'],manifest_sha256=old_ref['sha256'])
    write_json(new.parent/'execution-migration.json',migration)
    return old,new,new_ref,migration


def test_corpus_ancestry_follows_current_bound_contract(tmp_path):
    old,new,ref,_=ancestry(tmp_path)
    assert control.corpus_chain(ref)==[str(old),str(new)]


@pytest.mark.parametrize('field',['manifest_sha256','execution_source_commit','lineage_source_commit'])
def test_unbound_ancestry_tampering_is_rejected(tmp_path,field):
    _,new,ref,migration=ancestry(tmp_path)
    migration[field]='0'*len(migration[field]);write_json(new.parent/'execution-migration.json',migration)
    with pytest.raises(control.ControlError,match='not bound'):control.corpus_chain(ref)


def test_migration_cannot_redirect_to_an_unbound_prior_manifest(tmp_path):
    old,new,ref,migration=ancestry(tmp_path)
    alternative=write_json(tmp_path/'other-handoff.json',{'corpus_root':str(old),'corpus_contract':control.bound(old/'contract.json'),'tamper':True})
    migration.update(manifest_path=alternative['path'],manifest_sha256=alternative['sha256'])
    write_json(new.parent/'execution-migration.json',migration)
    with pytest.raises(control.ControlError,match='not bound'):control.corpus_chain(ref)


def test_resumed_corpus_cannot_lose_migration(tmp_path):
    _,new,ref,_=ancestry(tmp_path);(new.parent/'execution-migration.json').unlink()
    with pytest.raises(control.ControlError,match='lost its execution migration'):control.corpus_chain(ref)


def test_changed_bound_corpus_contract_is_rejected(tmp_path):
    _,new,ref,_=ancestry(tmp_path)
    write_json(new/'contract.json',{'source_commit':control.LINEAGE,'tamper':True})
    with pytest.raises(control.ControlError,match='bound handoff artifact changed'):control.corpus_chain(ref)


def array_ref(root,relative,arrays,*,frame=False):
    import hashlib
    import numpy as np
    path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as stream:np.savez_compressed(stream,**arrays)
    ref=dict(path=str(relative),bytes=path.stat().st_size,sha256=control.sha(path),
        arrays={name:dict(dtype=str(value.dtype),shape=list(value.shape),
            sha256=hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest())
            for name,value in arrays.items()})
    if frame:
        ref.update(content_sha256=control.sha(path),language=['fake test instruction'])
        write_json(path.with_suffix('.json'),ref)
    return ref


def raw_corpus(root,*,phase='source',seed=11,trial=7,empty=False):
    """Two real typed decisions/three public frames, no model or simulator."""
    import numpy as np
    root=Path(root);episode='episode-test';directory=root/'episodes'/episode
    context=dict(phase=phase,method='ppo',trial_id=trial)
    frames=[]
    for n in range(3 if not empty else 1):
        frames.append(array_ref(root,Path('frames')/f'frame-{n}.npz',dict(
            main_images=np.full((1,2,2,3),n,dtype=np.uint8),
            wrist_images=np.full((1,2,2,3),n,dtype=np.uint8),
            states=np.full((1,2),n,dtype=np.float32),features=np.full((1,3),n,dtype=np.float32)),frame=True))
    write_json(directory/'reset.json',dict(episode_id=episode,context=context,
        reset_info=dict(task='handover_mic',requested_seed=seed,actual_seed=seed),initial_observation=frames[0]))
    refs=[]
    for n in range(1,1 if empty else 3):
        relative=Path('episodes')/episode/'chunks'/f'{n:06d}'
        decision=array_ref(root,relative.with_suffix('.npz'),dict(
            sampled_noise=np.zeros((1,2),dtype=np.float32),issued_motor_joint_targets=np.zeros((1,2),dtype=np.float32),
            **{'metadata/input':np.zeros((1,2),dtype=np.float32)}))
        native=array_ref(root,relative.with_name(relative.name+'-native').with_suffix('.npz'),dict(
            raw_rewards=np.zeros(1,dtype=np.float32),raw_terminations=np.zeros(1,dtype=np.bool_),
            raw_truncations=np.array([n==2],dtype=np.bool_),valid_action_mask=np.ones(1,dtype=np.bool_)))
        record=dict(status='complete',episode_id=episode,chunk_index=n,previous_chunk_index=n-1 if n>1 else None,
            context=context,observation=frames[n-1],next_observation=frames[n],decision_arrays=decision,
            decision_metadata={'input':{'numeric_array':'metadata/input'}},issued_primitive_steps=1,
            native_result=dict(executed_primitive_steps=1,arrays=native,info={}),
            learning_result=dict(terminated=False,truncated=n==2,budget_truncated=False,episode_primitive_steps=n))
        ref=write_json(root/relative.with_suffix('.json'),record)
        refs.append(dict(path=str(Path('chunks')/Path(ref['path']).name),sha256=ref['sha256']))
    manifest=dict(episode_id=episode,reset='reset.json',chunks=refs,pending_decision=None,
        completed_chunks=len(refs),executed_primitive_steps=len(refs))
    write_json(directory/'manifest.json',manifest)
    return root,directory


def resign_chunk(directory,n,mutate):
    path=directory/'chunks'/f'{n:06d}.json';record=json.loads(path.read_text());mutate(record);write_json(path,record)
    manifest=json.loads((directory/'manifest.json').read_text());manifest['chunks'][n-1]['sha256']=control.sha(path)
    write_json(directory/'manifest.json',manifest)


def test_every_raw_decision_and_reference_is_checked_with_honest_hash_scope(tmp_path):
    root,_=raw_corpus(tmp_path/'raw')
    counts,receipt=control.raw_action_counts([str(root)])
    assert counts=={'source/ppo':2} and receipt['completed_chunks']==2
    assert receipt['unique_nonzero_source_episodes']==1 and receipt['source_episode_keys_unique']
    assert receipt['raw_json_index_hashes_verified'] and receipt['referenced_artifacts_present']
    assert receipt['array_headers_verified'] and receipt['native_valid_action_masks_verified']
    assert receipt['full_blob_hash_audit'] is False


def test_tampered_earlier_chunk_is_rejected_even_when_last_chunk_is_intact(tmp_path):
    root,directory=raw_corpus(tmp_path/'raw');path=directory/'chunks/000001.json'
    path.write_text(path.read_text()+' ')
    with pytest.raises(control.ControlError,match='index/order/hash'):control.raw_action_counts([str(root)])


@pytest.mark.parametrize('kind',['earlier_chunk','decision_npz','native_npz','initial_frame','middle_frame','frame_sidecar'])
def test_missing_earlier_decision_native_or_frame_artifact_is_rejected(tmp_path,kind):
    root,directory=raw_corpus(tmp_path/'raw')
    paths={'earlier_chunk':directory/'chunks/000001.json','decision_npz':directory/'chunks/000001.npz',
        'native_npz':directory/'chunks/000001-native.npz','initial_frame':root/'frames/frame-0.npz',
        'middle_frame':root/'frames/frame-1.npz','frame_sidecar':root/'frames/frame-1.json'}
    paths[kind].unlink()
    with pytest.raises(control.ControlError,match='missing'):control.raw_action_counts([str(root)])


def test_declared_array_shape_must_match_real_npz_header(tmp_path):
    root,directory=raw_corpus(tmp_path/'raw')
    resign_chunk(directory,1,lambda row:row['decision_arrays']['arrays']['sampled_noise'].update(shape=[9,9]))
    with pytest.raises(control.ControlError,match='shape/dtype/size'):control.raw_action_counts([str(root)])


def test_every_chunk_status_and_cumulative_action_sum_is_checked(tmp_path):
    root,directory=raw_corpus(tmp_path/'raw')
    resign_chunk(directory,1,lambda row:row.update(status='native_complete'))
    with pytest.raises(control.ControlError,match='closed status'):control.raw_action_counts([str(root)])


def test_native_action_mask_must_bind_every_executed_duration(tmp_path):
    root,directory=raw_corpus(tmp_path/'raw')
    def tamper(row):
        row['native_result']['executed_primitive_steps']=0
        row['learning_result']['episode_primitive_steps']=0
    resign_chunk(directory,1,tamper)
    with pytest.raises(control.ControlError,match='valid-action mask'):control.raw_action_counts([str(root)])


def test_duplicate_nonzero_source_episode_across_roots_is_rejected(tmp_path):
    first,_=raw_corpus(tmp_path/'prefix');second,_=raw_corpus(tmp_path/'suffix')
    with pytest.raises(control.ControlError,match='executed twice'):
        control.raw_action_counts([str(first),str(second)])


def test_partial_trial_prefix_and_suffix_have_distinct_source_cases(tmp_path):
    first,_=raw_corpus(tmp_path/'prefix',seed=11,trial=239);second,_=raw_corpus(tmp_path/'suffix',seed=12,trial=239)
    counts,receipt=control.raw_action_counts([str(first),str(second)])
    assert counts=={'source/ppo':4} and receipt['unique_nonzero_source_episodes']==2


def test_repeated_evaluation_case_and_empty_preflight_are_allowed(tmp_path):
    first,_=raw_corpus(tmp_path/'val1',phase='source_validation');second,_=raw_corpus(tmp_path/'val2',phase='source_validation')
    empty,_=raw_corpus(tmp_path/'empty',empty=True)
    counts,receipt=control.raw_action_counts([str(first),str(second),str(empty)])
    assert counts=={'source_validation/ppo':4} and receipt['unique_nonzero_source_episodes']==0


def ledger_fixture():
    contract={'source_commit':control.LINEAGE,'methods':['ppo']}
    worker=dict(method='ppo',campaign_id='original-campaign',maximum_actions=1226400)
    snapshot=dict(header=dict(campaign_id=worker['campaign_id'],total_actions=worker['maximum_actions'],
        source_caps={'ppo':1100000},phase_caps={},contract=contract))
    return snapshot,worker,contract


def test_live_ledger_header_matches_original_parent_worker():
    snapshot,worker,contract=ledger_fixture()
    control.validate_ledger_header(snapshot,worker,'ppo',contract)


@pytest.mark.parametrize('field,value',[('campaign_id','other-campaign'),('total_actions',9999999),
    ('source_caps',{'ppo':2200000}),('phase_caps',{'source':9999999}),('contract',{'source_commit':'changed'})])
def test_live_ledger_header_tampering_is_rejected_independently_of_tail_hash(field,value):
    snapshot,worker,contract=ledger_fixture();snapshot['header'][field]=value
    with pytest.raises(control.ControlError,match='header identity/caps/contract'):
        control.validate_ledger_header(snapshot,worker,'ppo',contract)


def external_contract_ancestry(tmp_path):
    old,new,ref,migration=ancestry(tmp_path)
    backup=tmp_path/'exports/recovery/ppo/corpus-contract.json'
    backup.parent.mkdir(parents=True);backup.write_bytes((old/'contract.json').read_bytes())
    previous=Path(migration['manifest_path']);manifest=json.loads(previous.read_text())
    manifest['corpus_contract']=control.bound(backup);old_ref=write_json(previous,manifest)
    new_contract=json.loads((new/'contract.json').read_text())
    new_contract['resume_trial_manifest_sha256']=old_ref['sha256']
    contract_ref=write_json(new/'contract.json',new_contract)
    current=json.loads(Path(ref['path']).read_text());current['corpus_contract']=contract_ref
    ref=write_json(ref['path'],current)
    migration['manifest_sha256']=old_ref['sha256'];write_json(new.parent/'execution-migration.json',migration)
    return old,new,ref


def test_external_hash_identical_backup_contract_is_valid_ancestry(tmp_path):
    old,new,ref=external_contract_ancestry(tmp_path)
    assert control.corpus_chain(ref)==[str(old),str(new)]


def test_external_bound_contract_cannot_disagree_with_canonical_root(tmp_path):
    old,_,ref=external_contract_ancestry(tmp_path)
    write_json(old/'contract.json',{'source_commit':control.LINEAGE,'changed':True})
    with pytest.raises(control.ControlError,match='canonical corpus contract content'):control.corpus_chain(ref)
