"""Hash-bound task references and explicit uniform-task/stream sampling."""
import json
from pathlib import Path

TASKS = ('OpenDrawer', 'CloseToasterOvenDoor', 'CoffeeSetupMug',
         'PickPlaceCounterToStove', 'TurnOffStove', 'TurnOnSinkFaucet')


def load_corpus(path, digest, size, objective):
    import numpy as np
    from fluffy_bassoon.meta_rl.vla_history import file_sha256, load_dataset, validate_trial
    assert size in (3, 6) and objective in ('ad', 'dpt', 'dit', 'ic_iql')
    assert file_sha256(path) == digest
    spec = json.loads(Path(path).read_text())
    selected = TASKS[:size]
    assert set(selected) <= spec['tasks'].keys()
    data, ids, sources, bindings = {'train': [], 'validation': []}, {'train': [], 'validation': []}, {}, {}
    identity = None
    for task in selected:
        ref = spec['tasks'][task]
        assert file_sha256(ref['path']) == ref['sha256']
        assembly = json.loads(Path(ref['path']).read_text())
        assert assembly['student_fit_ready'] and assembly['task'] == task
        current = {k: assembly[k] for k in ('codec', 'native_model_sha256', 'native_norm_sha256', 'runtime_commit')}
        if identity is None:
            identity = current
        assert identity == current
        sources.update(assembly['sources']); bindings[task] = {}
        for split, first, count in [('train', 0, 112), ('validation', 112, 16)]:
            item = assembly['datasets'][objective + '_' + split]
            assert file_sha256(Path(item['path']).with_suffix('.json')) == item['manifest_sha256']
            trials, manifest = load_dataset(item['path'], expected_sha256=item['sha256'])
            meta = manifest['metadata']
            assert objective in meta['shared_objectives']
            assert meta['task'] == task and len(trials) == count
            names = [f'{task}/flow-source-slot-{i:03d}' for i in range(first, first + count)]
            assert meta['trial_ids'] == item['trial_ids'] == names
            assert meta['noise_contract'] == identity['codec']
            assert meta['motor_sha256'] == identity['native_model_sha256']
            assert meta['norm_sha256'] == identity['native_norm_sha256']
            assert meta['value_contract']['boundary'] == 'physical_done'
            assert meta['value_contract']['discount'] == .99
            for trial in trials:
                assert validate_trial(trial, objective=objective) == (360, 2073, 600)
                assert np.array_equal(trial['teacher_steps'], np.repeat(np.arange(30), 12))
                assert np.array_equal(trial['episode_starts'], np.arange(360) % 12 == 0)
                assert np.array_equal(trial['dones'], np.arange(360) % 12 == 11)
            data[split].extend(trials); ids[split].extend(names)
            bindings[task][split] = dict(path=item['path'], sha256=item['sha256'], manifest_sha256=item['manifest_sha256'])
    assert not set(ids['train']) & set(ids['validation'])
    assert len(set(ids['train'])) == 112 * size and len(set(ids['validation'])) == 16 * size
    return dict(data=data, ids=ids, sources=sources, bindings=bindings, identity=identity, tasks=selected)


def balanced_batch(factory, trials, policy, rng, *, task_count, streams_per_task, **kwargs):
    """The core sampler receives one chosen stream per lane, then draws windows."""
    import numpy as np
    width = kwargs['batch_size']
    assert len(trials) == task_count * streams_per_task
    tasks = rng.integers(task_count, size=width)
    slots = rng.integers(streams_per_task, size=width)
    indices = tasks * streams_per_task + slots

    class WindowDraws:
        first = True

        def integers(self, *args, **kw):
            if self.first:
                self.first = False
                assert args == (width,) and kw == {'size': width}
                return np.arange(width)
            return rng.integers(*args, **kw)

    batch = factory([trials[int(i)] for i in indices], policy, WindowDraws(), **kwargs)
    if 'query_indices' in batch:
        batch['query_indices'] = [(int(indices[i]), query) for i, query in batch['query_indices']]
    return batch
