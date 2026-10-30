"""Pre-armed one-shot CUA proof artifact bridge. Never submits or runs a gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import signal
import subprocess
import sys
import time

RUN = 'robotwin-sac65-cpu-compat-20261031-r2'
REMOTE_INPUT = '/mowenjie/fluffy-bassoon-2/inputs/' + RUN
REMOTE_OUTPUT = '/mowenjie/fluffy-bassoon-2/outputs/' + RUN


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_new(path, value):
    """Atomic no-clobber publication of an owned regular file."""
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path)  # Atomic create; a stale final file is never overwritten.
    temporary.unlink()


def regular_bytes(path, root, maximum):
    require(path.parent == root and root.resolve() == root and not root.is_symlink()
            and path.is_file() and not path.is_symlink(), 'Uncontained or redirected proof artifact')
    require(0 < path.stat().st_size <= maximum, 'Empty or oversized proof artifact')
    raw = path.read_bytes()
    require(0 < len(raw) <= maximum, 'Empty or oversized proof artifact')
    return raw


def prepare_proof(inbox, ready, *, nonce, code_commit, armed_at, now):
    marker = json.loads(regular_bytes(inbox/'proof-ready.json', inbox, 16384))
    require(marker.get('schema') == 'robotwin_cpu_actual_ui_proof_ready_v1'
        and marker.get('run_id') == RUN and marker.get('code_commit') == code_commit
        and marker.get('nonce') == nonce and marker.get('dispatcher_pid') == ready['pid']
        and type(marker.get('captured_at_unix')) in (int,float)
        and armed_at <= marker['captured_at_unix'] <= now < ready['expires_at_unix'],
        'Stale, expired or mismatched CUA proof marker')
    image = regular_bytes(inbox/'actual-running.png', inbox, 10 << 20)
    require(image.startswith(b'\x89PNG\r\n\x1a\n'), 'CUA screenshot must be PNG')
    ax_raw = regular_bytes(inbox/'actual-running.txt', inbox, 256 << 10)
    ax = ax_raw.decode('utf-8')
    name = 'mwj_1031_robotwin_sac65_cpu_r2_' + code_commit[:7]
    require(re.search(r'\d+ text ' + re.escape(name) + r'\n[^\n]*\d+ text 运行中(?:\n|$)', ax),
            'Actual task is not currently Running in CUA AX proof')
    ids = set(re.findall(r'\d+ text ID:\s+(job-[a-z0-9]+)(?:\n|$)', ax))
    require(len(ids) == 1, 'Actual CUA job identity is missing or ambiguous')
    allocation = ids.pop()
    hosts = set(re.findall(r'\d+ text (' + re.escape(allocation) + r'-master-\d+)(?:\n|$)', ax))
    require(hosts == {allocation+'-master-0'}, 'Actual single allocation hostname is missing')
    # Parse the actual Worker resource table, rather than supplied request fields.
    rows = re.findall(r'\d+ text Worker\n.*?\d+ text (\d+)\n.*?'
        r'\d+ text 不使用加速芯片\n.*?\d+ text (\d+) 核\n.*?\d+ text (\d+) GiB(?:\n|$)', ax, re.S)
    require(rows == [('1','8','64')], 'Actual CUA Worker resources must be exactly1x0GPU/8CPU/64GiB')
    resource = dict(schema='robotwin_actual_cpu_allocation_v1',
        status='issued_actual_running_allocation', allocation_id=allocation,
        hostname=allocation+'-master-0', allocated_gpus=0, allocated_cpus=8, memory_gib=64,
        observed_running_allocation=True, source='Actual Running CUA task detail and Worker resource table',
        proof=dict(path=REMOTE_INPUT+'/actual-running.png',sha256=digest(image)),
        ax_proof=dict(path=REMOTE_INPUT+'/actual-running.txt',sha256=digest(ax_raw)),
        capture=dict(nonce=nonce, dispatcher_pid=ready['pid'], at_unix=marker['captured_at_unix']))
    return resource, {'actual-running.png':image, 'actual-running.txt':ax_raw}


def validate_intent(intent, resource, code_commit, now):
    require(intent.get('run_id') == RUN and intent.get('code_commit') == code_commit
        and intent.get('hostname') == resource['hostname'] and intent.get('output') == REMOTE_OUTPUT
        and intent.get('max_wait_seconds') == 120 and intent.get('work_deadline_seconds_from_start') == 1400
        and type(intent.get('bootstrap_expires_at_unix')) in (float,int)
        and 0 < intent['bootstrap_expires_at_unix']-now <= 121,
        'Late or mismatched actual bootstrap intent')


def bounded_command(command, *, timeout, log):
    """Trusted ssh/scp only; retain output and reap the owned session on failures."""
    guard=Path(__file__).with_name('transfer_guard.py')
    receipt=log.with_name(log.name+'.'+str(time.time_ns())+'.receipt.json')
    with log.open('ab') as stream:
        process = subprocess.Popen([sys.executable,'-I','-B','-S',str(guard)],
            stdin=subprocess.PIPE,stdout=stream,stderr=subprocess.STDOUT,text=True,
            start_new_session=True, close_fds=True, restore_signals=True)
        try:
            process.stdin.write(json.dumps(dict(command=command,parent_pid=os.getpid(),
                deadline=time.monotonic()+timeout,receipt=str(receipt)))+'\n');process.stdin.flush()
            code = process.wait(timeout=timeout+3)
            require(code == 0, 'Artifact transfer command failed: '+str(code))
        finally:
            old={n:signal.signal(n,signal.SIG_IGN) for n in (signal.SIGTERM,signal.SIGINT)}
            process.stdin.close()  # Parent loss/EOF also reaches the independent guard.
            if process.poll() is None:
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.terminate();process.wait(timeout=3)
            for n,handler in old.items():signal.signal(n,handler)


REMOTE_PUBLISH = r'''
import hashlib,json,os,sys,time
from pathlib import Path
stage,base,out,pin,expected_intent = map(str,sys.argv[1:])
stage,base,out=Path(stage),Path(base),Path(out)
assert base.resolve()==base and stage.parent==base and not base.is_symlink()
assert stage.resolve()==stage and stage.is_dir() and not stage.is_symlink()
for name in ('actual-running.png','actual-running.txt','actual-resource.json','actual-bootstrap.json'):
 assert (stage/name).is_file() and not (stage/name).is_symlink()
assert not (out/'terminal.json').exists()
raw=(out/'bootstrap-intent.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()==expected_intent
intent=json.loads(raw)
assert time.time()<intent['bootstrap_expires_at_unix']
ticket=json.loads((stage/'actual-bootstrap.json').read_bytes())
resource_raw=(stage/'actual-resource.json').read_bytes(); resource=json.loads(resource_raw)
assert ticket['code_commit']==pin and ticket['hostname']==intent['hostname']==resource['hostname']
assert ticket['run_id']==intent['run_id'] and ticket['output']==str(out)
assert ticket['resource_receipt']['sha256']==hashlib.sha256(resource_raw).hexdigest()
for field in ('proof','ax_proof'):
 ref=resource[field]; name=Path(ref['path']).name
 assert Path(ref['path'])==base/name
 path=stage/name
 assert path.is_file() and not path.is_symlink()
 assert hashlib.sha256(path.read_bytes()).hexdigest()==ref['sha256']
for name in ('actual-running.png','actual-running.txt','actual-resource.json','actual-bootstrap.json'):
 assert not (base/name).exists() and not (base/name).is_symlink()
assert not (out/'terminal.json').exists() and time.time()<intent['bootstrap_expires_at_unix']
for name in ('actual-running.png','actual-running.txt','actual-resource.json','actual-bootstrap.json'):
 os.link(stage/name,base/name)
 (stage/name).unlink()
print(json.dumps(dict(published=True,ticket=str(base/'actual-bootstrap.json'),at_unix=time.time())))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inbox',type=Path,required=True)
    p.add_argument('--code-commit',required=True)
    p.add_argument('--ssh-target',required=True);p.add_argument('--ssh-port',type=int,required=True)
    a=p.parse_args()
    require(re.fullmatch('[0-9a-f]{40}',a.code_commit) and 0<a.ssh_port<65536,'Wrong exact pin/port')
    require(re.fullmatch(r'[A-Za-z0-9_.@-]+',a.ssh_target),'Wrong configured SSH target')
    inbox=a.inbox.absolute()
    require(inbox.resolve()==inbox and not inbox.exists(),'Fresh canonical local inbox required')
    inbox.mkdir(parents=True,exist_ok=False)
    armed=time.time(); expiry=armed+300; nonce=secrets.token_hex(16)
    ready=dict(schema='robotwin_cpu_proof_dispatcher_ready_v1',pid=os.getpid(),nonce=nonce,
        run_id=RUN,code_commit=a.code_commit,armed_at_unix=armed,expires_at_unix=expiry,
        task_name='mwj_1031_robotwin_sac65_cpu_r2_'+a.code_commit[:7],
        output=REMOTE_OUTPUT,remote_input=REMOTE_INPUT)
    def interrupted(signum,frame):raise RuntimeError('Proof dispatcher signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    status='failed'; failure=None
    try:
        write_new(inbox/'dispatcher-ready.json',ready)
        while not (inbox/'proof-ready.json').exists():
            require(time.time()<expiry,'CUA proof capture did not arrive before dispatcher expiry')
            time.sleep(.1)
        resource,files=prepare_proof(inbox,ready,nonce=nonce,code_commit=a.code_commit,
                                   armed_at=armed,now=time.time())
        options=['-o','BatchMode=yes','-o','ConnectTimeout=5','-o','ServerAliveInterval=2',
                 '-o','ServerAliveCountMax=1']
        ssh=['ssh',*options,'-p',str(a.ssh_port),a.ssh_target]
        def transfer(command,maximum,log):
            remaining=expiry-time.time()-6  # Cleanup is inside the one dispatcher lifetime.
            require(remaining>0,'No transfer time remains before dispatcher expiry')
            bounded_command(command,timeout=min(maximum,remaining),log=log)
        # Resource and proof are actual CUA output before any remote publication.
        intent_path=REMOTE_OUTPUT+'/bootstrap-intent.json'
        read_program="from pathlib import Path;import sys; p=Path(sys.argv[1]);sys.stdout.buffer.write(p.read_bytes() if p.is_file() and p.stat().st_size<=65536 else b'')"
        while True:
            require(time.time()<expiry,'Actual gate intent did not arrive in dispatcher lifetime')
            log=inbox/('intent-read-'+str(time.time_ns())+'.log')
            command=' '.join(shlex.quote(x) for x in ['timeout','--kill-after=1s','10',
                'python3','-I','-B','-S','-c',read_program,intent_path])
            transfer([*ssh,command],12,log)
            raw=log.read_bytes()
            if raw.strip():
                intent=json.loads(raw);break
            time.sleep(.25)
        validate_intent(intent,resource,a.code_commit,time.time())
        resource_raw=(json.dumps(resource,indent=2)+'\n').encode()
        files['actual-resource.json']=resource_raw
        ticket=dict(schema='robotwin_cpu_gate_actual_bootstrap_v1',run_id=RUN,
            code_commit=a.code_commit,hostname=resource['hostname'],output=REMOTE_OUTPUT,
            resource_receipt=dict(path=REMOTE_INPUT+'/actual-resource.json',sha256=digest(resource_raw)))
        files['actual-bootstrap.json']=(json.dumps(ticket,indent=2)+'\n').encode()
        stage=inbox/'upload';stage.mkdir()
        for name,data in files.items():(stage/name).write_bytes(data)
        remote_stage=REMOTE_INPUT+'/upload-'+nonce
        transfer([*ssh,'mkdir -- '+shlex.quote(remote_stage)],10,inbox/'transfer.log')
        transfer(['scp',*options,'-P',str(a.ssh_port),*[str(stage/name) for name in files],
            a.ssh_target+':'+remote_stage+'/'],15,inbox/'transfer.log')
        command=' '.join(shlex.quote(x) for x in ['timeout','--kill-after=1s','10',
            'python3','-I','-B','-S','-c',REMOTE_PUBLISH,remote_stage,REMOTE_INPUT,
            REMOTE_OUTPUT,a.code_commit,digest(raw)])
        transfer([*ssh,command],12,inbox/'publish.log')
        status='published_actual_ticket_once'
    except BaseException as exc:
        failure=type(exc).__name__+': '+str(exc);raise
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        write_new(inbox/'dispatcher-terminal.json',dict(status=status,failure=failure,
            pid=os.getpid(),nonce=nonce,finished_at_unix=time.time(),submitted_jobs=0))


if __name__=='__main__':main()
