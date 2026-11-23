"""Linux CPU wiring check. Redirect the hard-coded PFS root to a private fixture."""
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import partition
import reference as ref


def executable(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(0o755)


def shell(root):
    run = "outputs/reground-rotation-reference-20261123-r1"
    sha = "a" * 40
    code = root / "transport"
    here = code / run / "partition"
    runtime = root / run
    attempts = runtime / "reference-native/attempts"
    attempts.mkdir(parents=True)
    fakebin = root / "bin"
    executable(fakebin / "hostname", "#!/bin/sh\necho job-fixture-master-0\n")
    executable(fakebin / "git", '''#!/bin/bash
tree=$2
shift 2
case "$1 $2" in
  "rev-parse --show-toplevel") echo "$FIXTURE_CODE" ;;
  "rev-parse HEAD")
    if [[ $tree == "$FIXTURE_CODE" ]]; then echo "$FIXTURE_PIN"; else echo "${tree##*/src-}"; fi ;;
  "symbolic-ref -q") exit 1 ;;
esac
''')
    # The CPU double permits the original reserve code; probe and dispatch can
    # only leave marker files in the redirected attempt. No CUDA library is loaded.
    executable(runtime / "env/bin/python", '''#!/bin/bash
case "$2" in
  */partition.py)
    if [[ $3 == reserve ]]; then
        exec "$FIXTURE_PYTHON" "$@"
    fi
    [[ $3 == dispatch ]] || exit 99
    printf '%s\\n' "$@" > "$attempt/dispatched-args"
    exit 75 ;;
  *gpu_probe.py)
    printf '{"fixture":true}\\n' > "$attempt/gpu.json" ;;
  *) exit 99 ;;
esac
''')
    here.mkdir(parents=True)
    source = Path(__file__).resolve().parent
    for name in ("partition.py", "collect.py", "fit.py"):
        (here / name).write_bytes((source / name).read_bytes())
    launcher = (source / "launch.sh").read_text()
    original = "root=/mowenjie/fluffy-bassoon-2/$run"
    assert launcher.count(original) == 1
    executable(here / "launch.sh", launcher.replace(original, f"root={runtime}"))
    refdir = runtime / f"src-{partition.NATIVE_RUNNER}" / run
    refdir.mkdir(parents=True)
    (refdir / "reference.py").symlink_to(Path(ref.__file__).resolve())
    (runtime / "runtime").mkdir()
    (runtime / "runtime/activate.sh").write_text('export PYTHONPATH="$REGROUND_REFERENCE_DIR"\n')
    (here.parent / "speed").mkdir()
    import hashlib
    content = (runtime / "runtime/activate.sh").read_bytes()
    (here.parent / "speed/runtime.sha256").write_text(hashlib.sha256(content).hexdigest() + "  runtime/activate.sh\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PATH=f"{fakebin}:{os.environ['PATH']}",
               FIXTURE_CODE=str(code), FIXTURE_PYTHON=sys.executable, FIXTURE_PIN=sha,
               KUBERNETES_SERVICE_HOST="fixture", REFERENCE_CAP_GPUH="1", WORLD_SIZE="1")
    cmd = [str(here / "launch.sh"), sha, "fresh", "1", "100", "fit", "--namespace", "canonical", "--seeds", "0"]
    assert subprocess.run(cmd, env=env).returncode == 75, (attempts / "fresh/launch.log").read_text()
    row = ref.read(attempts / "fresh/reservation.json")
    assert row["max_gpu_seconds"] == 100 and row["gpus"] == 1
    assert ref.read(attempts / "fresh/exit.json")["exit_code"] == 75
    assert (attempts / "fresh/dispatched-args").exists()
    cmd[2] = "bad-pin"
    env["FIXTURE_PIN"] = "b" * 40
    assert subprocess.run(cmd, env=env).returncode != 0
    bad = ref.read(attempts / "bad-pin/reservation.json")
    assert bad["admitted"] is False and bad["max_gpu_seconds"] == ref.read(attempts / "bad-pin/exit.json")["allocated_gpu_seconds"]
    assert not (attempts / "bad-pin/gpu.json").exists()
    return dict(fit_cards=row["gpus"], reserved_gpu_seconds=row["max_gpu_seconds"], stub_busy_status=75,
                failed_pin_before_probe=True, failed_preflight_accounted=True)


def failures(root):
    fake = root / "cpu-lane.py"
    fake.write_text("""import os, signal, sys, time
rank = int(sys.argv[-1])
if rank == 1:
 time.sleep(0.5)
 sys.exit(1)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
time.sleep(30)
""")
    args = SimpleNamespace(gpus=2, attempt=root, phase="collect")
    t = time.monotonic()
    with patch.object(partition, "__file__", str(fake)):
        try:
            partition.dispatch(args)
        except RuntimeError as exc:
            assert "lane failed" in str(exc).lower()
        else:
            raise AssertionError("Failed lane was ignored")
    elapsed = time.monotonic() - t
    assert elapsed < 30, "Dispatcher waited for the surviving fixture's entire sleep"
    return dict(nonfirst_lane_failure_reaped_resistant_sibling_s=round(elapsed, 2))


if __name__ == "__main__":
    assert platform.system() == "Linux" and os.environ["CUDA_VISIBLE_DEVICES"] == ""
    with tempfile.TemporaryDirectory(prefix="reground-launch-") as temp:
        work = Path(temp)
        ref.write(Path(sys.argv[1]), dict(shell=shell(work), cleanup=failures(work), gpu_hours=0))
