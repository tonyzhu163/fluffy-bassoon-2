"""CPU-only dispatch around exact0c16 qualified owned-session primitives.

No allocation, GPU method, model, native runtime, source control or ledger
operation. The independent guardian shares the caller's absolute deadline.
"""
import argparse
import ctypes
import hashlib
import importlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

HELPER_PIN = '0c16c2b794f8de5b78603206917e72195083c265'
HELPERS = {
    'robotwin_trained_pair_runtime.py':'116b5251b43454a4c61fe0bd4faeb3d1cbf6d7a2dd08356241599cbd2f32827c',
    'robotwin_trained_pair_continuation.py':'173774c46c177548750c78e1e09aa5c8caf583d265be8ecb5517757077274c9b',
    'robotwin_trained_pair.py':'d8ee934c580cb1d1ab3991f446618af2bd9e06a6ece97cf0fe83aa828b882dc0'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_helpers(directory):
    if {p.name for p in directory.iterdir()} != set(HELPERS):
        raise ValueError('Owned primitive directory must contain only the three pinned source files')
    for name, expected in HELPERS.items():
        path = directory / name
        if path.is_symlink() or sha(path) != expected:
            raise ValueError('Qualified ownership primitive changed: ' + name)


def write(path, value):
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def become_subreaper():
    # Linux stable UAPI PR_SET_CHILD_SUBREAPER/PR_GET_CHILD_SUBREAPER. This
    # guardian is a new process; adopted children can only be its own subtree.
    libc = ctypes.CDLL(None, use_errno=True)
    enabled = ctypes.c_int()
    if (libc.prctl(36, 1, 0, 0, 0) != 0
            or libc.prctl(37, ctypes.byref(enabled), 0, 0, 0) != 0 or enabled.value != 1):
        raise RuntimeError('Cannot establish owned-descendant subreaping')


def reap_adopted(deadline):
    reaped = []
    while time.monotonic() < deadline - 1:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return dict(clean=True, reaped=reaped)
        if pid:
            reaped.append(dict(pid=pid, wait_status=status))
        else:
            time.sleep(.02)
    return dict(clean=False, reaped=reaped, error='Owned adopted descendants not reaped by deadline')


def guardian_main(helper_dir):
    verify_helpers(helper_dir)
    sys.path.insert(0, str(helper_dir))
    base = importlib.import_module('robotwin_trained_pair_runtime')
    continuation = importlib.import_module('robotwin_trained_pair_continuation')
    parent = os.getppid()
    line = base._control_line(time.monotonic() + 10, parent)
    if line is None:
        return
    config = json.loads(line)
    if (config['parent_pid'] != parent or config['cleanup_reserve'] != 30
            or not 0 < config['deadline'] - time.monotonic() <= 1700):
        raise ValueError('Wrong CPU guardian owner/deadline contract')
    become_subreaper()
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT})
    guardian = continuation.ContinuationGuardian(config)
    signals = []
    original_signal = base._signal
    def record_signal(identity, number, *, leader_only=False):
        row = dict(pid=identity['pid'], sid=identity['sid'], start_ticks=identity['start_ticks'],
                   signal=int(number), leader_only=leader_only)
        signals.append(row)
        try:
            original_signal(identity, number, leader_only=leader_only)
            row['status'] = 'sent_or_already_exited'
        except BaseException as exc:
            row['status'] = type(exc).__name__; raise
    base._signal = record_signal
    def stop(signum, frame):
        raise RuntimeError('Owned CPU guardian termination: ' + str(signum))
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    def cleanup():
        # Never let repeated termination interrupt bounded cleanup. No child
        # is launched while these handlers are ignored.
        old = {n:signal.signal(n, signal.SIG_IGN) for n in (signal.SIGTERM, signal.SIGINT)}
        try:
            result = guardian.cleanup(graceful=False)
            adopted = reap_adopted(guardian.deadline)
            result.update(adopted=adopted, signal_attempts=list(signals))
            result['clean'] = result['clean'] and adopted['clean']
            write(guardian.output / ('owned-cleanup-%03d.json' % guardian.cleanup_number), result)
            write(guardian.output / 'owned-cleanup-latest.json', result)
            if not result['clean']:
                guardian.failed = True
                raise RuntimeError('Owned CPU descendants remain unreaped')
            return result
        finally:
            for number, handler in old.items():
                signal.signal(number, handler)
    print(json.dumps(dict(ready=True, guardian_pid=os.getpid(), helper_pin=HELPER_PIN)), flush=True)
    try:
        while True:
            line = base._control_line(guardian.deadline, parent)
            if line is None:
                break
            request = json.loads(line)
            try:
                if guardian.failed:
                    raise RuntimeError('Failed CPU guardian cannot launch more work')
                timeout = request['timeout_seconds']
                if not 0 < timeout <= 900:
                    raise ValueError('CPU phase timeout outside cap')
                started = time.monotonic()
                entry = guardian.launch(request['name'], request['command'], request['cwd'],
                                        request['env'], request['log_path'])
                until = min(guardian.work_deadline, started + timeout)
                while entry['process'].poll() is None:
                    guardian.parent_alive()
                    if time.monotonic() >= until:
                        raise TimeoutError('Owned CPU phase deadline reached')
                    time.sleep(.02)
                code = entry['process'].returncode
                cleaned = cleanup()
                print(json.dumps(dict(ok=True, result=dict(returncode=code, cleanup=cleaned,
                     elapsed_seconds=time.monotonic()-started))), flush=True)
            except BaseException as exc:
                guardian.failed = True
                try:
                    cleaned = cleanup()
                except BaseException as error:
                    cleaned = dict(clean=False, error=str(error))
                failure = dict(ok=False, error=type(exc).__name__ + ': ' + str(exc), cleanup=cleaned)
                write(guardian.output / 'owned-failure.json', failure)
                print(json.dumps(failure), flush=True)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN); signal.signal(signal.SIGINT, signal.SIG_IGN)
        cleanup()


class CPUPhaseRuntime:
    def __init__(self, helper_dir, output, *, deadline):
        self.helper_dir, self.output = Path(helper_dir).resolve(), Path(output).resolve()
        verify_helpers(self.helper_dir)
        if sys.platform != 'linux' or not 31 < deadline - time.monotonic() <= 1700:
            raise ValueError('Need allocated Linux and remaining shared deadline/cleanup reserve')
        self.output.mkdir(parents=True, exist_ok=False)
        self.deadline, self.closed = deadline, False
        self.stderr = (self.output / 'guardian.stderr.log').open('x')
        self.guardian = subprocess.Popen([sys.executable, '-I', '-B', '-S', str(Path(__file__).resolve()),
            '--guardian', '--helper-dir', str(self.helper_dir)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.stderr, text=True, start_new_session=True,
            close_fds=True, restore_signals=True)
        config = dict(parent_pid=os.getpid(), output=str(self.output), deadline=deadline, cleanup_reserve=30)
        self.guardian.stdin.write(json.dumps(config)+'\n'); self.guardian.stdin.flush()
        write(self.output / 'owned-runtime.json', dict(config, guardian_pid=self.guardian.pid,
              helper_pin=HELPER_PIN, helper_sha256=HELPERS, adapter_sha256=sha(__file__), subreaper=True))
        ready, _, _ = select.select([self.guardian.stdout], [], [], min(10, self.remaining()))
        if not ready or not json.loads(self.guardian.stdout.readline()).get('ready'):
            self.close(); raise RuntimeError('CPU guardian did not establish ownership')

    def remaining(self):
        return max(0., self.deadline-time.monotonic())

    def run(self, name, command, cwd, environment, log_path, timeout):
        if self.closed:
            raise RuntimeError('CPU runtime closed')
        request = dict(name=name, command=list(command), cwd=str(cwd), env=dict(environment),
                       log_path=str(log_path), timeout_seconds=timeout)
        self.guardian.stdin.write(json.dumps(request)+'\n'); self.guardian.stdin.flush()
        ready, _, _ = select.select([self.guardian.stdout], [], [], self.remaining())
        if not ready:
            raise TimeoutError('CPU guardian shared deadline expired')
        line = self.guardian.stdout.readline()
        if not line:
            raise RuntimeError('CPU guardian exited without phase completion')
        reply = json.loads(line)
        if not reply['ok']:
            raise RuntimeError(reply['error'])
        return reply['result']

    def close(self):
        if self.closed and self.guardian.poll() is not None:
            if self.guardian.returncode != 0:
                raise RuntimeError('Owned CPU guardian failed cleanup')
            return
        self.closed = True
        if not self.guardian.stdin.closed:
            self.guardian.stdin.close()
        try:
            self.guardian.wait(timeout=max(1., self.remaining()))
        finally:
            if self.guardian.poll() is not None:
                self.stderr.close(); self.guardian.stdout.close()
        if self.guardian.returncode != 0:
            raise RuntimeError('Owned CPU guardian failed cleanup')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--guardian', action='store_true', required=True)
    p.add_argument('--helper-dir', type=Path, required=True)
    args = p.parse_args()
    guardian_main(args.helper_dir)
