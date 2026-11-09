"""Check the actor export, native rollout data and pairing to corrected SFT."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

OUTPUTS = Path(__file__).resolve().parents[1]


def module(name, directory):
    spec = importlib.util.spec_from_file_location(name, OUTPUTS/directory/'readback.py')
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def audit(root, clean, randomized):
    native = module('sidney_baseline', 'robotwin-sidney-native-sft-20261109-r1')
    gap = module('sidney_gap', 'robotwin-sidney-gap-20261109-r1')
    result = gap.load(root/'RESULT.json')
    endpoint = gap.load(root/'ENDPOINT.json')
    assert result['endpoint'] == endpoint and endpoint['exact_roundtrip_policy_tensors'] == 667
    assert endpoint['extra_optimizer_steps'] == 0 and endpoint['adam_step'] == 40
    report = native.audit(root)
    report['endpoint'] = endpoint
    for setting, ref in [('demo_clean', clean), ('demo_randomized', randomized)]:
        actor = result['settings'][setting]
        baseline = gap.load(ref/'RESULT.json')['settings'][setting]
        assert len(actor) == len(baseline) == 100
        for a, b in zip(actor, baseline, strict=True):
            assert all(a[k] == b[k] for k in ('seed', 'index', 'actors', 'instruction', 'wrist_near_m'))
            assert np.allclose(a['wrist_near_m'], .01)
            idx = a['index']
            initial = f"{setting}-{idx:03d}-seed-{a['seed']}/initial.npz"
            with np.load(root/initial, allow_pickle=False) as x, np.load(ref/initial, allow_pickle=False) as y:
                assert x.files == y.files and all(np.array_equal(x[k], y[k]) for k in x.files)
            for j, chunk in enumerate(a['chunks']):
                assert chunk['noise_seed'] == 1000+idx*100+j
        report['settings'][setting].update(exact_matched_initial_observations=len(actor),
            sft_successes=sum(b['success'] for b in baseline),
            paired_sft_first_actor_second=gap.pairs([e['success'] for e in baseline],[e['success'] for e in actor]))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('endpoint', 'clean_sft', 'randomized_sft'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.endpoint, args.clean_sft, args.randomized_sft), indent=2))
