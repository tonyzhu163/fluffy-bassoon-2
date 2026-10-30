"""One local trusted transfer, bounded independently of dispatcher lifetime."""
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time


def session_worker():
    """Reserve the owned SID through TERM grace, even if the transfer exits."""
    stopping=False
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    child=subprocess.Popen(json.loads(sys.argv[2]),stdin=subprocess.DEVNULL,
        close_fds=True,restore_signals=True)
    while child.poll() is None or stopping:
        time.sleep(.02)
    sys.exit(child.returncode)


def main():
    parent=os.getppid()
    ready,_,_=select.select([sys.stdin],[],[],5)
    if not ready:return
    line=sys.stdin.readline(65537)
    if not line:return
    config=json.loads(line)
    if (config['parent_pid']!=parent or not 0<config['deadline']-time.monotonic()<=16
            or len(line)>65536):raise ValueError('Wrong owned transfer owner/deadline')
    stopping=False
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    process=None;reason='command_exit';code=1
    try:
        process=subprocess.Popen([sys.executable,'-I','-B','-S',str(Path(__file__).resolve()),
            '--session-worker',json.dumps(config['command'])],stdin=subprocess.DEVNULL,
            start_new_session=True,close_fds=True,restore_signals=True)
        while process.poll() is None:
            readable,_,_=select.select([sys.stdin],[],[],0)
            parent_lost=os.getppid()!=parent or (readable and not os.read(sys.stdin.fileno(),1))
            if stopping or parent_lost or time.monotonic()>=config['deadline']:
                reason='signal' if stopping else ('parent_lost' if parent_lost else 'timeout')
                break
            time.sleep(.02)
        if process.poll() is not None:
            code=process.returncode
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        if process is not None and process.poll() is None:
            # The live owned leader reserves this process-group identity.
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=1)
        receipt=dict(reason=reason,guardian_pid=os.getpid(),parent_pid=parent,
            child_pid=process.pid if process is not None else None,
            child_returncode=process.returncode if process is not None else None,
            code=code,finished_monotonic=time.monotonic())
        with Path(config['receipt']).open('x') as stream:
            json.dump(receipt,stream);stream.write('\n')
    sys.exit(code)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--session-worker':session_worker()
    else:main()
