"""Count Transformers model-source open requests in fresh native worker imports."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def worker():
    opened = []

    def count(event, args):
        if event == "open" and isinstance(args[0], str):
            path = args[0]
            if "/transformers/models/" in path and path.endswith(".py"):
                opened.append(path)

    sys.addaudithook(count)
    start = time.perf_counter()
    from lerobot.envs.libero import LiberoEnv  # Same import as TaskEnv.__init__.
    import transformers
    elapsed = time.perf_counter() - start
    source = Path(transformers.__file__).parent
    expected = Path(sys.argv[2]) / "transformers"
    if source != expected or any(not p.startswith(str(source) + "/") for p in opened):
        raise RuntimeError(f"Unexpected import origin {source}, expected {expected}")
    relative = sorted({str(Path(p).relative_to(source)) for p in opened})
    print(json.dumps(dict(origin=str(source), lib=LiberoEnv.__name__, elapsed_s=elapsed, opens=len(opened),
                          distinct_sources=len(relative),
                          source_names_sha256=hashlib.sha256("\n".join(relative).encode()).hexdigest())))


def benchmark():
    staged = os.environ["REGROUND_TRANSFORMERS_PATH"]
    canonical = str(Path(sys.prefix) / "lib/python3.12/site-packages")
    paths = [p for p in os.environ["PYTHONPATH"].split(os.pathsep) if p != staged]
    for arm in ("shared", "scratch", "scratch", "shared", "shared", "scratch", "scratch", "shared"):
        cache = arm == "scratch"
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(paths + ([staged] if cache else [])))
        start = time.perf_counter()
        output = subprocess.check_output([sys.executable, "-B", __file__, "worker",
                                          staged if cache else canonical], env=env, text=True)
        wall = time.perf_counter() - start
        print(json.dumps(dict(arm=arm, process_wall_s=wall, **json.loads(output.splitlines()[-1]))), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["benchmark"]:
        benchmark()
    elif len(sys.argv) == 3 and sys.argv[1] == "worker":
        worker()
    else:
        sys.exit("Usage: probe.py benchmark (inside with-cache.sh)")
