"""Run-local numerical qualification against the original dataset writer."""
import importlib.util
import json
from pathlib import Path
import tempfile
import numpy as np
from fluffy_bassoon.meta_rl.vla_history import FIELDS, load_dataset, save_dataset

spec = importlib.util.spec_from_file_location('pool_fast', Path(__file__).with_name('POOL-UNCOMPRESSED.py'))
fast = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fast)
checks = []
with tempfile.TemporaryDirectory(prefix='joint-pool-equivalence-') as td:
    for objective in ('ad', 'dpt', 'dit', 'ic_iql'):
        for split, streams in (('train', 336), ('validation', 48)):
            rng = np.random.default_rng(42)
            trials = []
            for i in range(streams):
                action = rng.normal(size=(4, 6)).astype(np.float32)
                trials.append(dict(observations=rng.normal(size=(4, 12)).astype(np.float32),
                    executed_noise=action, target_noise=action.copy(),
                    rewards=rng.normal(size=4).astype(np.float32),
                    dones=np.array([False, True, False, True]),
                    episode_starts=np.array([True, False, True, False]),
                    teacher_steps=np.array([0, 0, 1, 1], dtype=np.int64)))
            metadata = dict(trial_ids=[f'task-{i // (streams // 3)}/slot-{i % (streams // 3)}' for i in range(streams)], split=split, joint_training=True)
            a = Path(td) / f'{objective}-{split}-reference.npz'
            b = Path(td) / f'{objective}-{split}-uncompressed.npz'
            ma = save_dataset(a, trials, objective=objective, metadata=metadata)
            mb = fast.save_dataset(b, trials, objective=objective, metadata=metadata)
            assert {k:v for k,v in ma.items() if k != 'sha256'} == {k:v for k,v in mb.items() if k != 'sha256'}
            with np.load(a, allow_pickle=False) as aa, np.load(b, allow_pickle=False) as bb:
                assert aa.files == bb.files
                for key in aa.files:
                    assert aa[key].dtype == bb[key].dtype
                    assert aa[key].shape == bb[key].shape
                    assert np.array_equal(aa[key], bb[key]), key
            loaded, manifest = load_dataset(b, expected_sha256=mb['sha256'])
            assert len(loaded) == streams and manifest['objective'] == objective
            for expected, observed in zip(trials, loaded, strict=True):
                for key in FIELDS:
                    assert np.array_equal(expected[key], observed[key])
            checks.append(dict(objective=objective, split=split, streams=streams, arrays='exact', metadata='exact except container SHA256'))
print(json.dumps(dict(passed=True, device='Linux CPU; CUDA hidden', checks=checks), indent=2))
