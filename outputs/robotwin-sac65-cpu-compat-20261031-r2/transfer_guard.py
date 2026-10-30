"""One transfer's owned SID, bounded independently of dispatcher lifetime.

Supported ssh/scp descendants may change process group but must remain in the
captured session. A live holder reserves its identity through final cleanup.
"""
import ctypes
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time


def session_worker():
    signal.signal(signal.SIGTERM,lambda *_:None)
    signal.signal(signal.SIGINT,lambda *_:None)
    command,status=json.loads(sys.argv[2]),Path(sys.argv[3])
    try:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,close_fds=True,restore_signals=True)
        code=child.wait()
    except BaseException:
        code=127
    try:
        temporary=status.with_name(status.name+'.partial')
        with temporary.open('x') as stream:
            json.dump(dict(command_returncode=code),stream);stream.flush();os.fsync(stream.fileno())
        os.link(temporary,status);temporary.unlink()
    except BaseException as exc:
        print('Command status publication failed: '+str(exc),file=sys.stderr,flush=True)
    finally:
        # A failed open/fsync/link must never release the reserved identity.
        while True:time.sleep(.02)


def become_subreaper():
    if sys.platform!='linux':return False  # macOS launchd reaps orphaned children.
    libc=ctypes.CDLL(None,use_errno=True);enabled=ctypes.c_int()
    if (libc.prctl(36,1,0,0,0)!=0 or libc.prctl(37,ctypes.byref(enabled),0,0,0)!=0
            or enabled.value!=1):raise RuntimeError('Cannot establish transfer subreaper')
    return True


def process_ids():
    if sys.platform=='linux':
        result=[]
        for path in Path('/proc').iterdir():
            if not path.name.isdecimal():continue
            try:result.append((int(path.name),path.stat().st_uid))
            except FileNotFoundError:pass
        return result
    if sys.platform=='darwin':
        raw=subprocess.check_output(['/bin/ps','-axo','pid=,uid='],text=True,timeout=.5)
        return [tuple(map(int,line.split())) for line in raw.splitlines() if line.strip()]
    raise RuntimeError('Unsupported transfer ownership platform')


def session_members(holder):
    if holder.poll() is not None or os.getsid(holder.pid)!=holder.pid:
        raise RuntimeError('Owned transfer SID holder was lost before cleanup')
    members=[]
    for pid,uid in process_ids():
        if uid!=os.getuid():continue
        try:
            if os.getsid(pid)==holder.pid:members.append(pid)
        except ProcessLookupError:pass
    return sorted(members)


def clean_session(holder,subreaper):
    deadline=time.monotonic()+5;signals=[];reaped=[];known={holder.pid}
    errors=[];unresolved=False
    def send(pid,number):
        if holder.poll() is not None:raise RuntimeError('Transfer holder died during cleanup')
        try:
            if os.getsid(pid)==holder.pid:
                os.kill(pid,number);signals.append(dict(pid=pid,signal=int(number)))
        except ProcessLookupError:pass
    def reap():
        if not subreaper:return
        while True:
            try:pid,status=os.waitpid(-1,os.WNOHANG)
            except ChildProcessError:return
            if not pid:return
            if pid==holder.pid:raise RuntimeError('Transfer holder unexpectedly exited')
            reaped.append(dict(pid=pid,status=status))
    def members():
        nonlocal unresolved
        for _ in range(2):
            if time.monotonic()>=deadline-.5:break
            try:
                current=session_members(holder);known.update(current);unresolved=False
                return current
            except (OSError,ValueError,subprocess.SubprocessError) as exc:
                errors.append(type(exc).__name__+': '+str(exc))
                time.sleep(.02)
        unresolved=True
        current=[]
        for pid in sorted(known):
            try:
                if os.getsid(pid)==holder.pid:current.append(pid)
            except ProcessLookupError:pass
        return current
    def original_group(number):
        # This fallback runs while the owned holder is live. It catches the
        # original ssh/scp group even if Darwin ps fails before a PID snapshot.
        if holder.poll() is not None or os.getsid(holder.pid)!=holder.pid:
            raise RuntimeError('Cannot signal a released transfer identity')
        os.killpg(holder.pid,number)
        signals.append(dict(pgid=holder.pid,signal=int(number),fallback=True))
    for pid in members():send(pid,signal.SIGTERM)
    if unresolved:original_group(signal.SIGTERM)
    graceful=min(deadline-2,time.monotonic()+1)
    while time.monotonic()<graceful:
        reap()
        if members()==[holder.pid]:break
        time.sleep(.02)
    # The holder remains live while changed-PGID members are killed/reaped.
    while time.monotonic()<deadline-1:
        current=[pid for pid in members() if pid!=holder.pid]
        if not current:break
        for pid in current:send(pid,signal.SIGKILL)
        reap();time.sleep(.02)
    reap();remaining=[pid for pid in members() if pid!=holder.pid]
    if unresolved:
        # Kill known changed groups first, then the captured original group
        # including its holder. No signal is sent after releasing that SID.
        for pid in remaining:send(pid,signal.SIGKILL)
        original_group(signal.SIGKILL)
    else:send(holder.pid,signal.SIGKILL)
    holder.wait(timeout=max(.1,deadline-time.monotonic()))
    # Never signal after releasing the reserved SID identity.
    after=[]
    try:final_processes=process_ids()
    except (OSError,ValueError,subprocess.SubprocessError) as exc:
        errors.append(type(exc).__name__+': '+str(exc));unresolved=True;final_processes=[]
    for pid,uid in final_processes:
        if uid!=os.getuid():continue
        try:
            if os.getsid(pid)==holder.pid:after.append(pid)
        except ProcessLookupError:pass
    return dict(clean=not unresolved and not remaining and not after,unresolved_SID=unresolved,
        enumeration_errors=errors,remaining_before_holder_release=remaining,
        remaining_after=after,signals=signals,reaped_adopted=reaped,holder_wait_status=holder.returncode)


def main():
    parent=os.getppid();ready,_,_=select.select([sys.stdin],[],[],5)
    if not ready:return
    line=sys.stdin.readline(65537)
    if not line:return
    config=json.loads(line)
    if (config['parent_pid']!=parent or not 0<config['deadline']-time.monotonic()<=16
            or len(line)>65536):raise ValueError('Wrong owned transfer owner/deadline')
    if sys.platform not in ('linux','darwin'):raise RuntimeError('Unsupported transfer ownership platform')
    subreaper=become_subreaper();stopping=False
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    process=None;reason='command_exit';code=1;command_code=None;cleanup=None;failure=None
    status=Path(config['receipt']+'.command.json')
    try:
        process=subprocess.Popen([sys.executable,'-I','-B','-S',str(Path(__file__).resolve()),
            '--session-worker',json.dumps(config['command']),str(status)],stdin=subprocess.DEVNULL,
            start_new_session=True,close_fds=True,restore_signals=True)
        while True:
            if process.poll() is not None:raise RuntimeError('Transfer holder exited before release')
            readable,_,_=select.select([sys.stdin],[],[],0)
            parent_lost=os.getppid()!=parent or (readable and not os.read(sys.stdin.fileno(),1))
            if stopping or parent_lost or time.monotonic()>=config['deadline']:
                reason='signal' if stopping else ('parent_lost' if parent_lost else 'timeout');break
            if status.exists():
                command_code=json.loads(status.read_text())['command_returncode'];code=command_code;break
            time.sleep(.02)
    except BaseException as exc:
        code=1;failure=type(exc).__name__+': '+str(exc)
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        if process is not None:
            try:
                cleanup=clean_session(process,subreaper)
                if not cleanup['clean']:code=1
            except BaseException as exc:
                code=1;failure=(failure or '')+' cleanup: '+str(exc)
                if process.poll() is None:process.kill();process.wait(timeout=1)
        receipt=dict(reason=reason,guardian_pid=os.getpid(),parent_pid=parent,child_pid=process.pid if process else None,
            holder_returncode=process.returncode if process else None,command_returncode=command_code,exit_code=code,
            cleanup=cleanup,failure=failure,subreaper=subreaper,finished_monotonic=time.monotonic())
        final=Path(config['receipt']);temporary=final.with_name(final.name+'.partial')
        with temporary.open('x') as stream:
            json.dump(receipt,stream);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.link(temporary,final);temporary.unlink()
    sys.exit(code)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--session-worker':session_worker()
    else:main()
