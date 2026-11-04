"""Archive independently qualified stale environments and other project assets."""
import importlib.util
import json
from pathlib import Path

p = Path('/mowenjie/fluffy-bassoon-2/outputs/storage-audit-baige-20261104-r1')
spec = importlib.util.spec_from_file_location('storage_archive', p / 'archive.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
plan = json.loads((p / 'WORKSPACE-CANDIDATES.json').read_text())
m.ROOT = Path('/mowenjie')
m.AUDIT = p / 'workspace'
m.AUDIT.mkdir(exist_ok=False)
m.BOS = m.BOS / 'workspace'
m.STAGE = m.STAGE / 'workspace'
plan['selected'] = [str(Path(s).relative_to(m.ROOT)) for s in plan['selected']]
m.save(m.AUDIT / 'CANDIDATES.json', plan)
m.main()
