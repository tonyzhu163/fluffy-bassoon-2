"""On Baige CPU, exercise the production cache wrapper and rank supervisor against a private fixture store."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

RUN = Path("/mowenjie/fluffy-bassoon-2/outputs/reground-retry-wrist-20261123-r1")
PIN = "ac28987f221e336cef2ea64dc16dfffc9f0e74c9"
SOURCE = RUN / f"src-{PIN}"
RECOVERY = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shim():
    flag, program, *args = sys.argv[1:]
    program = Path(program)
    out = Path(os.environ["attempt"])
    if program.name == "[train]gpu_probe.py":
        assert flag == "-Bu" and program.is_file()
        assert args == ["--baige-container", "--purpose", "training", "--baige-exclusive-training",
                        "--expected-count", "4", "--output", str(out/"gpu.json")]
        (out/"cpu-fixture-probe.json").write_text(json.dumps({"real_gpu_probe": False, "count_requested": 4}))
        return 0
    stage, directory, rank, phase = args
    assert (flag, stage, directory, phase) == ("-Bu", "lane", str(out), "full")
    assert rank in ("0","1","2","3")
    target = SOURCE/"driver.py"
    cache = Path(os.environ["REGROUND_TRANSFORMERS_PATH"])
    assert cache.is_dir() and cache.parent == Path("/tmp")
    assert (cache/"transformers").is_dir()
    receipt = dict(rank=int(rank), received=str(program), expected=str(target), matches=program == target,
                   present=program.is_file(), wrapper_here=os.environ["here"], cache=cache.name)
    if program == target and program.is_file():
        if os.environ["RECOVERY_TRIAL"] == "success" and rank == "0":
            stats = out/"real-driver-readback"
            stats.mkdir()
            subprocess.run([os.environ["RECOVERY_REAL_PY"],"-Bu",str(target),"cpu","readback",str(stats)],check=True)
            passed = json.loads((stats/"passed.json").read_text())
            assert passed["code"] == PIN and passed["original_rows"] == 1400
            receipt["real_driver_historical_rows"] = passed["original_rows"]
        else:
            subprocess.run([os.environ["RECOVERY_REAL_PY"],"-Bu","-c",
                "import driver,sys; from pathlib import Path; assert Path(driver.__file__).resolve()==Path(sys.argv[1])",
                str(target)],check=True)
    (out/f"cpu-rank{rank}.json").write_text(json.dumps(receipt))
    if not receipt["matches"]:
        return 2
    return 43 if os.environ["RECOVERY_TRIAL"] == "rank3-fails" and rank == "3" else 0


def qualify(dest):
    dest.mkdir(exist_ok=False,parents=True)
    assert subprocess.check_output(["git","-C",SOURCE,"rev-parse","HEAD"],text=True).strip()==PIN
    env = dict(os.environ, RECOVERY_SHIM=str(dest/"shim"), CUDA_VISIBLE_DEVICES="-1", JAX_PLATFORMS="cpu",
               BASH_ENV="/etc/bash.bashrc")
    env.pop("PS1",None)
    helper = Path(__file__).resolve()
    (dest/"shim").write_text(f'#!/bin/bash\nexec /usr/bin/python3 -IB "{helper}" "$@"\n')
    (dest/"shim").chmod(0o700)
    cases = []
    for trial, src, expect in [("approved-unfixed", SOURCE/"launch.sh", 2),
                               ("success", RECOVERY/"launch.sh", 0),
                               ("rank3-fails", RECOVERY/"launch.sh", 43)]:
        shell = src.read_text()
        # Preserve the production cache invocation, shell argument expansion, four forks and waits.
        # Redirect only the store and the probe/lane Python executable. Preserve the original $0.
        orig_store = f"store={RUN}/scored"
        env_line = '    source "$here/env.sh"'
        assert shell.count(orig_store) == shell.count(env_line) == 1
        shell = shell.replace(orig_store, f"store={dest}/store").replace(env_line, env_line + '\n    export RECOVERY_REAL_PY="$py"\n    py="$RECOVERY_SHIM"')
        env["RECOVERY_TRIAL"] = trial
        rc = subprocess.run(["bash","-c",shell,str(src),PIN,trial,"4","4950","full"],env=env).returncode
        path = dest/"store/attempts"/trial
        assert rc == expect, (trial,rc,path/"launch.log")
        assert int((path/"exit.txt").read_text().split()[0]) == expect
        assert (path/"cpu-fixture-probe.json").is_file() and not (path/"gpu.json").exists()
        ranks = [json.loads((path/f"cpu-rank{i}.json").read_text()) for i in range(4)]
        assert all(r["matches"] == (trial != "approved-unfixed") for r in ranks)
        assert all(r["wrapper_here"].endswith("/transport") for r in ranks)
        assert all(r["present"] == (trial != "approved-unfixed") for r in ranks)
        assert "PS1: unbound variable" not in (path/"launch.log").read_text()
        cases.append(dict(trial=trial,expected_exit=expect,observed_exit=rc,
                          ranks=ranks,launcher_sha256=sha(src)))
    result = dict(science_pin=PIN, policy_or_simulator_gpu_hours=0, recovery_launcher_sha256=sha(RECOVERY/"launch.sh"),
                  approved_launcher_sha256=sha(SOURCE/"launch.sh"),driver_sha256=sha(SOURCE/"driver.py"),
                  actual_cache_wrapper=True,actual_pinned_environment=True,device_probe_and_gpu_lane_replaced=True,
                  production_scored_store_used=False, trials=cases)
    (dest/"qualified.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({r["trial"]:r["observed_exit"] for r in cases}))


if __name__ == "__main__":
    if sys.argv[1] == "-Bu":
        sys.exit(shim())
    qualify(Path(sys.argv[1]))
