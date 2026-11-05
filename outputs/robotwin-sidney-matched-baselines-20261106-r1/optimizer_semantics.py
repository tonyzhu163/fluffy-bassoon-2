"""Exact named Adam comparison for the qualified FSDP local shard layout.

opt01 measured native DCP dropping state only for zero-length, unowned
original parameters. Those parameters receive no synchronized gradient in
the installed FSDP implementation. Preserve their layout and validate their
empty state; compare every owned tensor and all group metadata exactly.
"""
import copy
import hashlib
import json


def without_devices(value):
    if isinstance(value, dict):
        return {k: without_devices(v) for k, v in value.items() if k != 'device'}
    if isinstance(value, list):
        return [without_devices(v) for v in value]
    return value


def canonical_layout(layout):
    parameters = {}
    for name, original in layout['parameters'].items():
        row = copy.deepcopy(original)
        assert row['registered_model_parameter'] and row['requires_grad'], name
        assert row['local_nonempty_parameter'] == (row['parameter_numel'] > 0), name
        state = row['state']
        if row['local_nonempty_parameter']:
            assert row['state_present'] and set(state) == {'step', 'exp_avg', 'exp_avg_sq'}, name
            assert all(s['kind'] == 'tensor' and s['finite'] for s in state.values()), name
            # Tensor descriptors retain dtype, shape and a hash over all bytes.
            row['state'] = without_devices(state)
        else:
            assert row['parameter_shape'] == [0], name
            if row['state_present']:
                assert set(state) == {'step', 'exp_avg', 'exp_avg_sq'}, name
                step = state['step']
                assert step['kind'] == 'tensor' and step['shape'] == [] and step['numel'] == 1, name
                assert step['dtype'] == 'torch.float32' and step['finite'], name
                assert step['values'] == [0.0] and step['nonzero_elements'] == 0, name
                for key in ('exp_avg', 'exp_avg_sq'):
                    moment = state[key]
                    assert moment['kind'] == 'tensor' and moment['shape'] == [0], name
                    assert moment['numel'] == 0 and moment['dtype'] == row['parameter_dtype'], name
                    assert moment['finite'] and moment['nonzero_elements'] == 0 and moment['values'] == [], name
            else:
                assert state == {}, name
            row.pop('state_present')
            row['state'] = 'unowned-zero-length-state-not-used-by-SGD'
        parameters[name] = row
    ordered = [n for g in layout['groups'] for n in g['parameters']]
    assert len(ordered) == len(set(ordered)) == len(parameters)
    assert set(ordered) == set(parameters)
    return dict(version=1, rank=layout['rank'], optimizer_steps=layout['optimizer_steps'],
        groups=without_devices(layout['groups']), parameters=parameters)


def semantic_digest(layout):
    value = canonical_layout(layout)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
