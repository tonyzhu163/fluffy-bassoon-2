"""Wait for the read-only inventory to finish, then exec the qualified archiver."""
from pathlib import Path
import json, os, sys, time
import tree

out=Path(__file__).resolve().parent
probe=json.loads((out/'METADATA-LAUNCH.json').read_text())
while (Path('/proc')/str(probe['pid'])).exists():
    tree.save(out/'DISPATCH-STATE.json',dict(pid=os.getpid(),phase='waiting_for_metadata',checked_unix=time.time()))
    time.sleep(5)
assert (out/'METADATA-SUMMARY.json').exists(), 'Metadata probe exited without a complete inventory; source preserved'
tree.save(out/'LAUNCH.json',dict(pid=os.getpid(),started_unix=time.time(),source_commit=os.environ['TASK_SOURCE'],
                              script_sha256=os.environ['TASK_SCRIPT_SHA256'],plan_sha256=os.environ['TASK_PLAN_SHA256'],GPU_allocations=0))
os.execve(sys.executable,[sys.executable,str(out/'archive.py')],os.environ)
