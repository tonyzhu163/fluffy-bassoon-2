#!/usr/bin/env python3
"""Bounded process ownership for a caller-supplied trained two-arm launch.

No model, scientific arguments, campaign or allocation is created here. A
separate stdlib guardian owns all command sessions and watches the caller's
pipe/PPID, so caller SIGKILL or native abort cannot bypass session cleanup.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import hashlib
import math
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import tempfile
import time


GIB = 1 << 30
MAX_SECONDS = 2520
_INPUT = bytearray()


def _control_line(deadline, parent):
    # TextIO.readline may read the next request ahead of select(), hiding it
    # in a userspace buffer. Keep one explicit buffer for the control pipe.
    while b"\n" not in _INPUT:
        if os.getppid() != parent: return None
        remaining = deadline-time.monotonic()
        if remaining <= 0: raise TimeoutError("Guardian control deadline reached")
        ready,_,_ = select.select([sys.stdin.fileno()],[],[],min(.2,remaining))
        if not ready: continue
        block = os.read(sys.stdin.fileno(),65536)
        if not block: return None
        _INPUT.extend(block)
        if len(_INPUT)>1<<20: raise ValueError("Oversized runtime request")
    line,_,rest = _INPUT.partition(b"\n")
    _INPUT[:] = rest
    return line.decode()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".runtime-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, allow_nan=False, indent=2)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _stat(pid):
    try:
        path = Path("/proc") / str(pid)
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        return dict(pid=int(pid), state=fields[0], pgid=int(fields[2]),
                    sid=int(fields[3]), start_ticks=int(fields[19]), uid=path.stat().st_uid)
    except (FileNotFoundError, ProcessLookupError):
        return None


def session_members(identity):
    """Include different PGIDs within our captured SID; reject leader reuse."""
    leader = _stat(identity["pid"])
    if leader is not None and leader["start_ticks"] != identity["start_ticks"]:
        raise RuntimeError("Owned session leader PID was reused; refusing signals")
    members = []
    for path in Path("/proc").glob("[0-9]*"):
        value = _stat(int(path.name))
        if value is not None and value["sid"] == identity["sid"] and value["state"] not in ("Z", "X"):
            if value["uid"] != identity["uid"]:
                raise RuntimeError("Owned SID contains a different UID; refusing signals")
            members.append(value)
    return members


def _signal(identity, number, *, leader_only=False):
    for member in session_members(identity):
        if leader_only and member["pid"] != identity["pid"]:
            continue
        # Recheck each PID immediately before signalling; never kill an
        # unrelated process just because its recycled PID was in an old scan.
        current = _stat(member["pid"])
        if current is None:
            continue
        if any(current[key] != member[key] for key in ("sid", "start_ticks", "uid")):
            raise RuntimeError("Owned PID identity changed before signal")
        try: os.kill(member["pid"], number)
        except ProcessLookupError: pass


def validate_probe(path, *, not_before):
    path = Path(path).resolve()
    raw = path.read_bytes(); probe = json.loads(raw)
    devices = probe.get("devices", [])
    if (probe.get("status") != "passed" or probe.get("purpose") != "training"
            or probe.get("allocation_type") != "baige_container" or len(devices) != 1
            or probe.get("hostname") != socket.gethostname()
            or probe.get("idle_utilization_required") is not True
            or probe.get("admission_policy") != "strict_idle_training"
            or probe.get("compute_processes_before") != []):
        raise RuntimeError("Require the fresh strict-idle one-GPU training probe")
    created = datetime.fromisoformat(probe["created_at_utc"]).timestamp()
    device = devices[0]; uuid = device.get("uuid")
    allocation = device.get("allocation_check", {})
    inventory = device.get("inventory", {})
    memory = float(inventory.get("memory_used_mib",math.inf))
    utilization = float(inventory.get("utilization_percent",math.inf))
    if (not math.isfinite(created) or created < not_before or created > time.time() + 5
            or not isinstance(uuid, str) or not uuid.startswith("GPU-")
            or allocation.get("touched_gpu_uuids") != [uuid]
            or allocation.get("own_pid_gpu_uuids") != [uuid]
            or probe.get("gpu_identity_mapping", {}).get("assigned_nvml_uuids") != [uuid]
            or not math.isfinite(memory) or not 0<=memory<=256
            or not math.isfinite(utilization) or not 0<=utilization<=5
            or device.get("cuda_total_bytes", 0) <= 0):
        raise RuntimeError("Probe is stale, busy or does not attest the allocated UUID")
    return probe, dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), uuid=uuid)


class _Guardian:
    def __init__(self, configuration):
        self.parent = configuration["parent_pid"]
        self.output = Path(configuration["output"])
        self.deadline = configuration["deadline"]
        self.work_deadline = self.deadline - configuration["cleanup_reserve"]
        self.sessions = []
        self.stage = "new"
        self.failed = False
        self.cleanup_number = 0

    def parent_alive(self):
        if os.getppid() != self.parent:
            raise RuntimeError("Runtime caller died; abort owned commands")
        if _INPUT: raise RuntimeError("Unexpected pipelined runtime request")
        ready, _, _ = select.select([sys.stdin.fileno()], [], [], 0)
        if ready:
            value = os.read(sys.stdin.fileno(), 1)
            if not value:
                raise RuntimeError("Runtime caller control pipe closed")
            raise RuntimeError("Unexpected concurrent runtime request")

    def launch(self, name, command, cwd, environment, log):
        self.parent_alive()
        if not command or not all(isinstance(value, str) and value for value in command):
            raise ValueError("Require an explicit argv, without a shell")
        stream = Path(log).open("x")
        try:
            process = subprocess.Popen(command, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        finally:
            stream.close()
        identity = _stat(process.pid)
        if identity is None or identity["sid"] != process.pid or identity["pgid"] != process.pid:
            raise RuntimeError("Failed to capture the newly owned session identity")
        entry = dict(name=name, process=process, identity=identity, command=command, log=str(log))
        self.sessions.append(entry)
        return entry

    def cleanup(self, *, graceful=True):
        def survivors():
            return [dict(name=e["name"], **member) for e in self.sessions
                    for member in session_members(e["identity"])]
        started = time.monotonic()
        before = survivors()
        for entry in self.sessions:
            _signal(entry["identity"], signal.SIGINT if graceful else signal.SIGTERM, leader_only=graceful)
            if graceful and entry["process"].poll() is not None:
                _signal(entry["identity"],signal.SIGTERM)
        phases = ((signal.SIGTERM, 60 if graceful else 2,15), (signal.SIGKILL,10,5), (None,5,0))
        for number, seconds, later_reserve in phases:
            until = min(self.deadline-1-later_reserve,time.monotonic()+seconds)
            while survivors() and time.monotonic() < until:
                for entry in self.sessions: entry["process"].poll()
                time.sleep(.05)
            if not survivors(): break
            if number is not None:
                for entry in self.sessions: _signal(entry["identity"], number)
        remaining = survivors()
        codes = {e["name"]: e["process"].poll() for e in self.sessions}
        receipt = dict(clean=not remaining and all(code is not None for code in codes.values()),
            sessions=[dict(name=e["name"], identity=e["identity"], returncode=codes[e["name"]]) for e in self.sessions],
            before=before, survivors=remaining, elapsed_seconds=time.monotonic()-started,
            deadline=self.deadline)
        self.cleanup_number += 1
        _write(self.output / f"cleanup-{self.cleanup_number:03d}.json", receipt)
        if not receipt["clean"]:
            self.failed = True
            raise RuntimeError("Owned commands did not reap; CUDA/further work forbidden")
        self.sessions = []
        return receipt

    def bounded(self, request):
        purpose = request["purpose"]
        if purpose == "cpu_preflight":
            if self.stage not in ("new", "pair_complete"):
                raise RuntimeError("CPU preflight/audit is out of sequence")
        elif purpose != "gpu_probe" or self.stage != "cpu_passed":
            raise RuntimeError("One GPU probe must follow the successful CPU preflight")
        timeout = float(request["timeout_seconds"])
        if not math.isfinite(timeout) or not 0 < timeout <= 90:
            raise ValueError("External preflight/probe/audit must be bounded to at most90 seconds")
        environment = dict(request["env"])
        if purpose == "cpu_preflight": environment["CUDA_VISIBLE_DEVICES"] = ""
        else:
            command = request["command"]
            if "--output" not in command: raise ValueError("GPU probe requires its fresh output file")
            self.probe_path = Path(command[command.index("--output")+1]).resolve()
            if self.probe_path.exists(): raise FileExistsError("Never reuse a GPU probe")
        started = time.monotonic()
        entry = self.launch(purpose, request["command"], request["cwd"], environment, request["log_path"])
        deadline = min(self.deadline-5 if self.stage=="pair_complete" else self.work_deadline,
                       started + timeout)
        while entry["process"].poll() is None:
            self.parent_alive()
            if time.monotonic() >= deadline: raise TimeoutError(purpose + " exceeded its external deadline")
            time.sleep(.05)
        code = entry["process"].returncode
        cleanup = self.cleanup(graceful=False)
        if code != 0: raise RuntimeError(f"{purpose} failed with exit{code}; no retry")
        receipt = dict(purpose=purpose, returncode=code, pid=entry["identity"]["pid"],
            elapsed_seconds=time.monotonic()-started, cleanup=cleanup, command=request["command"])
        if purpose == "gpu_probe":
            self.probe, self.probe_binding = validate_probe(self.probe_path, not_before=self.cpu_finished_utc)
            self.stage = "probe_passed"
            receipt["probe"] = self.probe_binding
        elif self.stage == "new":
            self.cpu_finished_utc = time.time(); self.stage = "cpu_passed"
        else: self.stage = "audit_complete"
        _write(self.output / f"{purpose}-{entry['identity']['pid']}.json", receipt)
        return receipt

    def sample_gpu(self, uuid):
        # nvidia-smi itself is an owned, bounded helper; a stuck query cannot
        # block this guardian in communicate()/wait() after an unkillable task.
        log = self.output / f".gpu-query-{time.time_ns()}.txt"
        entry = self.launch("telemetry", ["nvidia-smi", "--id="+uuid,
            "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            str(self.output), os.environ.copy(), log)
        until = min(self.deadline-1, time.monotonic()+3)
        while entry["process"].poll() is None:
            self.parent_alive()
            if time.monotonic() >= until: raise TimeoutError("GPU telemetry query stalled")
            time.sleep(.02)
        if entry["process"].returncode != 0: raise RuntimeError("Allocated UUID telemetry failed")
        if session_members(entry["identity"]): raise RuntimeError("Telemetry left an owned descendant")
        self.sessions.remove(entry)
        used, total, utilization = map(float, log.read_text().strip().split(","))
        log.unlink()
        if (not all(math.isfinite(v) for v in (used,total,utilization)) or not 0<=used<=total
                or total<=0 or not 0<=utilization<=100):
            raise RuntimeError("Invalid allocated UUID telemetry")
        return dict(uuid=uuid, used_bytes=int(used*(1<<20)), total_bytes=int(total*(1<<20)),
            utilization_percent=utilization, utc=datetime.now(timezone.utc).isoformat())

    def pair(self, request):
        if self.stage != "probe_passed": raise RuntimeError("Pair requires a completed fresh probe")
        if Path(request["probe_path"]).resolve() != self.probe_path:
            raise RuntimeError("Pair probe differs from the owned completed probe")
        _, binding = validate_probe(self.probe_path, not_before=self.cpu_finished_utc)
        if binding != self.probe_binding: raise RuntimeError("GPU probe changed")
        commands = request["commands"]
        if not isinstance(commands, dict) or len(commands) != 2:
            raise ValueError("Require exactly two independently supplied trained commands")
        deadline = min(float(request["deadline"]), self.work_deadline)
        if not math.isfinite(deadline) or deadline <= time.monotonic(): raise TimeoutError("No paired runtime remains")
        environment = dict(request["env"], LYCO_BAIGE_GPU_MAP=str(self.probe_path),
                           LYCO_BAIGE_ISOLATED_UUID=binding["uuid"])
        entries = {}
        for name, command in commands.items():
            if not name or Path(name).name != name: raise ValueError("Invalid arm name")
            local = dict(environment, FB2_ROBOTWIN_RENDER_RECORDS=str(Path(request["output"])/name/"renderer-bindings"))
            entries[name] = self.launch(name, command, request["worktree"], local,
                                       Path(request["output"])/(name+"-console.log"))
        started = time.monotonic(); samples = []; reason = None
        with (self.output / "gpu-memory.jsonl").open("x", buffering=1) as stream:
            while True:
                self.parent_alive()
                codes = {name:e["process"].poll() for name,e in entries.items()}
                if any(code not in (None,0) for code in codes.values()): reason="arm_failed"; break
                if all(code is not None for code in codes.values()): break
                if time.monotonic() >= deadline: reason="pair_deadline"; break
                row = self.sample_gpu(binding["uuid"])
                row.update(elapsed_seconds=time.monotonic()-started,
                    parents={name:e["identity"]["pid"] for name,e in entries.items() if codes[name] is None},
                    owned_pids=[m["pid"] for e in entries.values() for m in session_members(e["identity"])])
                stream.write(json.dumps(row)+"\n"); samples.append(row)
                if row["used_bytes"] > min(85*GIB,.9*row["total_bytes"]): reason="memory_headroom"; break
                time.sleep(.5)
        cleanup = self.cleanup()
        final = self.sample_gpu(binding["uuid"])
        until = min(self.deadline-1, time.monotonic()+10)
        while final["used_bytes"] > 256*(1<<20) and time.monotonic() < until:
            time.sleep(.2); final=self.sample_gpu(binding["uuid"])
        if final["used_bytes"] > 256*(1<<20): reason=reason or "GPU_contexts_not_released"
        result = dict(return_codes={name:e["process"].poll() for name,e in entries.items()},
            stop_reason=reason, cleanup=cleanup, final_gpu=final, probe=binding,
            elapsed_seconds=time.monotonic()-started,
            peak_used_bytes=max((s["used_bytes"] for s in samples),default=0),
            peak_utilization_percent=max((s["utilization_percent"] for s in samples),default=0),
            samples=len(samples), automatic_retries=0, total_deadline=self.deadline,
            collection_deadline=deadline)
        _write(self.output / "pair-result.json", result)
        if reason or any(code != 0 for code in result["return_codes"].values()):
            raise RuntimeError("Paired execution failed; no automatic retry: " + str(reason))
        self.stage = "pair_complete"
        return result


def _guardian_main():
    bootstrap = _control_line(time.monotonic()+10,os.getppid())
    if bootstrap is None: return
    configuration = json.loads(bootstrap)
    guardian = _Guardian(configuration)
    def stop(signum, frame): raise RuntimeError("Guardian received termination")
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        while True:
            line = _control_line(guardian.deadline,guardian.parent)
            if not line: break
            try:
                if guardian.failed: raise RuntimeError("Failed runtime cannot execute more commands")
                request = json.loads(line)
                result = guardian.bounded(request) if request["kind"] == "bounded" else guardian.pair(request)
                print(json.dumps(dict(ok=True,result=result)),flush=True)
            except BaseException as error:
                guardian.failed = True
                try: cleanup = guardian.cleanup()
                except BaseException as cleanup_error: cleanup=dict(clean=False,error=str(cleanup_error))
                failure=dict(ok=False,error=f"{type(error).__name__}: {error}",cleanup=cleanup)
                _write(guardian.output / "runtime-failure.json",failure)
                print(json.dumps(failure),flush=True)
    finally:
        # Do not let repeated INT/TERM interrupt the final bounded escalation.
        signal.signal(signal.SIGTERM,signal.SIG_IGN); signal.signal(signal.SIGINT,signal.SIG_IGN)
        try: guardian.cleanup()
        except BaseException as error: _write(guardian.output/"cleanup-failure.json",dict(clean=False,error=str(error)))


class TrainedPairRuntime:
    """One CPU preflight, one fresh probe, one pair, optional bounded CPU audit."""
    def __init__(self, worktree, output, *, maximum_seconds=MAX_SECONDS, cleanup_reserve_seconds=180):
        if (sys.platform != "linux" or not 0 < maximum_seconds <= MAX_SECONDS
                or not 10 <= cleanup_reserve_seconds < maximum_seconds
                or not all(math.isfinite(v) for v in (maximum_seconds,cleanup_reserve_seconds))):
            raise ValueError("Require Linux and a total runtime<=2520s with bounded cleanup reserve")
        self.worktree=Path(worktree).resolve(); self.output=Path(output).resolve()
        self.output.mkdir(parents=True,exist_ok=True)
        (self.output/"runtime-ownership").mkdir(exist_ok=False)
        self.started=time.monotonic(); self.deadline=self.started+maximum_seconds
        self.collection_deadline=self.deadline-cleanup_reserve_seconds
        self.failed=False; self.closed=False
        self._errors=(self.output/"guardian-stderr.log").open("x")
        self.guardian=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--guardian"],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self._errors,text=True,
            start_new_session=True,close_fds=True)
        self.guardian.stdin.write(json.dumps(dict(parent_pid=os.getpid(),output=str(self.output),
            deadline=self.deadline,cleanup_reserve=cleanup_reserve_seconds))+"\n")
        self.guardian.stdin.flush()
        _write(self.output/"runtime-owner.json",dict(parent_pid=os.getpid(),guardian_pid=self.guardian.pid,
            deadline=self.deadline,collection_deadline=self.collection_deadline,
            hard_allocation_seconds=2700,helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            environment_saved=False))

    def remaining_seconds(self): return max(0.,self.deadline-time.monotonic())

    def _request(self, request):
        if self.closed or self.failed: raise RuntimeError("Runtime closed/failed; fresh identity required")
        try:
            self.guardian.stdin.write(json.dumps(request)+"\n"); self.guardian.stdin.flush()
            ready,_,_=select.select([self.guardian.stdout.fileno()],[],[],self.remaining_seconds())
            if not ready: raise TimeoutError("Guardian exceeded the total runtime deadline")
            line=self.guardian.stdout.readline()
            if not line: raise RuntimeError("Guardian exited without a completion receipt")
            reply=json.loads(line)
            if not reply["ok"]: raise RuntimeError(reply["error"])
            return reply["result"]
        except BaseException:
            self.failed=True
            self.close()
            raise

    def run_bounded(self, command, cwd, env, log_path, timeout_seconds, *, purpose="cpu_preflight"):
        return self._request(dict(kind="bounded",command=list(command),cwd=str(Path(cwd).resolve()),
            env=dict(env),log_path=str(Path(log_path).resolve()),timeout_seconds=timeout_seconds,purpose=purpose))

    def run_pair(self, commands, probe_path, *, env=None, deadline=None):
        return self._request(dict(kind="pair",commands=commands,probe_path=str(Path(probe_path).resolve()),
            env=dict(os.environ if env is None else env),worktree=str(self.worktree),output=str(self.output),
            deadline=min(self.collection_deadline,self.collection_deadline if deadline is None else deadline)))

    def close(self):
        if self.closed: return
        self.closed=True
        self.guardian.stdin.close()
        try:
            self.guardian.wait(timeout=max(1.,self.remaining_seconds()))
        except subprocess.TimeoutExpired as error:
            # Do not kill the independent guardian while it owns an unkillable
            # session. Fail closed; the hard scheduler/allocation envelope is
            # still required to contain D-state or kernel/driver failures.
            _write(self.output/"guardian-unreaped.json",dict(clean=False,guardian_pid=self.guardian.pid,
                deadline=self.deadline,requires_allocation_cleanup=True))
            raise RuntimeError("Guardian has not reaped owned resources; no further CUDA") from error
        finally:
            self._errors.close(); self.guardian.stdout.close()
        if self.guardian.returncode != 0:
            raise RuntimeError("Guardian did not exit cleanly; inspect runtime receipts")

    def __enter__(self): return self
    def __exit__(self,*args): self.close()


if __name__ == "__main__":
    if sys.argv[1:] != ["--guardian"]: raise SystemExit("Import TrainedPairRuntime; caller owns launch admission")
    _guardian_main()
