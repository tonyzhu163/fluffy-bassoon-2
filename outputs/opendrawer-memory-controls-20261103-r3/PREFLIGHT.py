"""CPU numerical qualification of the carried window64 prefixes; no simulation."""
import json
import os
from pathlib import Path
import runpy
import time

import torch

root = Path('/mowenjie/fluffy-bassoon-2')
code = Path(os.environ['TASK_CODE'])
out = Path(os.environ.get('TASK_MEMORY_OUTPUT', root / 'outputs/opendrawer-memory-controls-20261103-r3'))
torch.set_num_threads(4)
assert not torch.cuda.is_available()
m = runpy.run_path(str(code / 'outputs/opendrawer-memory-controls-20261102-r1/CONTROL.py'))
b = json.loads((out / 'binding.json').read_text())
assert m['BASE']['head'](code) == b['source'] and m['BASE']['clean'](code)
missing = set(m['pending_keys'](b))
prior = json.loads((root / 'outputs/opendrawer-memory-controls-20261102-r2/TERMINAL-PARTIAL.json').read_text())
assert missing == set(prior['missing_keys']) and len(missing) == 48
assert len(b['recovered']) == 1052 and b['fresh_policy_controls'] == 36000
counts = {str(i): len(m['pending_keys'](b, i)) for i in range(8)}
assert counts == {'0': 0, '1': 11, '2': 23, '3': 14, '4': 0, '5': 0, '6': 0, '7': 0}

from fluffy_bassoon.meta_rl.vla_history import load_policy
from fluffy_bassoon.meta_rl.residual_history import HistoryResidualActor

mb = b['models']['dpt']
policy, _ = load_policy(mb['selected']['candidate']['path'],
    expected_sha256=mb['selected']['candidate']['sha256'], device='cpu')
prefixes = []
for i in sorted({int(key.split('/')[3]) for key in missing if key.startswith('dpt/window64/')}):
    window = HistoryResidualActor(policy, reset_each_episode=False, max_context=64)
    full = HistoryResidualActor(policy, reset_each_episode=False)
    for k in range(1, 6):
        old = m['read_ref'](mb['episodes'][f'ID-{i:03d}'][str(k)])
        if k > 1:
            controller, entry = window.controller, old['memory_entry']
            assert int(controller.state['length'][0]) == entry['cache_length'] == 15 * (k - 1)
            assert controller.previous_reward == entry['previous_reward']
            assert controller.previous_done == entry['previous_done']
            # Compare the same numerical backend. GPU byte hashes are checked
            # by the unchanged worker on GPU, before every missing query.
            assert m['signature'](window) == m['signature'](full)
        if k < 5:
            rows = torch.load(old['records']['path'], map_location='cpu', weights_only=False)
            m['replay'](window, rows)
            m['replay'](full, rows)
    prefixes.append(i)
    print('PREFIX', i, flush=True)
report = dict(complete=True, source=b['source'], carried=1052, missing=48,
    fresh_policy_controls=36000, logical_ranks=counts,
    verified_window64_prefix_cases=prefixes, binding=m['ref'](out / 'binding.json'),
    checked_unix=time.time(), gpu_work=False,
    cpu_numerical_reference='window64 equals full CPU replay before eviction; original boundary lengths and feedback match',
    gpu_exact_signature_check='Retained in worker before each reused prefix and first missing window64 query',
    corrected_preflight_assertion='CPU versus GPU byte hashes are not a valid numerical equality test; original diagnostic log preserved')
m['write'](out / 'CPU-PREFLIGHT.json', report)
print(json.dumps(report), flush=True)
