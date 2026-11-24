import runpy
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path[:0] = [str(here), str(here.parent/"frozen-vla-reachability-20261123-r1")]
sys.argv.pop(0)
runpy.run_path(sys.argv[0], run_name="__main__")
