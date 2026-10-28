"""Released exploration-key migration preserves all tensors and rejects ambiguity."""
import json
import struct

import pytest
import torch

from fluffy_bassoon.rl.rlinf_checkpoint_compat import (
    LEGACY_PREFIX, NATIVE_PREFIX, NOISE_SHAPES, checkpoint_has_noise_head,
    map_legacy_noise_head, LEGACY_VALUE_PREFIX, NATIVE_VALUE_PREFIX, VALUE_SHAPES,
    checkpoint_has_value_head, map_legacy_value_head, map_legacy_rl_heads,
)


def head(kind):
    return ((LEGACY_PREFIX, NATIVE_PREFIX, NOISE_SHAPES, map_legacy_noise_head)
            if kind == 'noise' else
            (LEGACY_VALUE_PREFIX, NATIVE_VALUE_PREFIX, VALUE_SHAPES, map_legacy_value_head))


def states(kind='noise'):
    old, new, shapes, _ = head(kind)
    saved = {old + k: torch.full(v, float(i))
             for i, (k, v) in enumerate(shapes.items())}
    saved['motor.weight'] = torch.arange(6).reshape(2, 3).float()
    target = {new + k: torch.empty(v) for k, v in shapes.items()}
    target['motor.weight'] = torch.empty(2, 3)
    return saved, target


@pytest.mark.parametrize('kind', ['noise', 'value'])
def test_mapping_preserves_every_tensor_by_identity_and_does_not_mutate_input(kind):
    old, new, shapes, migrate = head(kind)
    saved, target = states(kind)
    before = dict(saved)
    mapped = migrate(saved, target)
    assert set(saved) == set(before) and len(mapped) == len(saved)
    for name in shapes:
        assert mapped[new + name] is saved[old + name]
        assert saved[old + name] is before[old + name]
    assert mapped['motor.weight'] is saved['motor.weight']
    for k in saved:
        torch.testing.assert_close(saved[k], before[k], rtol=0, atol=0)


@pytest.mark.parametrize('bad', ['missing', 'extra', 'collision', 'target_missing',
                                 'target_extra', 'source_shape', 'target_shape',
                                 'source_dtype', 'target_dtype'])
@pytest.mark.parametrize('kind', ['noise', 'value'])
def test_migration_rejects_partial_changed_or_conflicting_heads(bad, kind):
    old_prefix, new_prefix, shapes, migrate = head(kind)
    saved, target = states(kind)
    suffix = next(iter(shapes))
    old, new = old_prefix+suffix, new_prefix+suffix
    if bad == 'missing': saved.pop(old)
    elif bad == 'extra': saved[old_prefix+'unknown'] = torch.zeros(1)
    elif bad == 'collision': saved[new] = saved[old]
    elif bad == 'target_missing': target.pop(new)
    elif bad == 'target_extra': target[new_prefix+'unknown'] = torch.zeros(1)
    elif bad == 'source_shape': saved[old] = torch.zeros(2)
    elif bad == 'target_shape': target[new] = torch.zeros(2)
    elif bad == 'source_dtype': saved[old] = saved[old].half()
    else: target[new] = target[new].half()
    with pytest.raises(ValueError): migrate(saved, target)


def test_unknown_non_noise_keys_remain_for_strict_native_rejection():
    saved, target = states()
    saved['unknown.weight'] = torch.zeros(1)
    assert map_legacy_noise_head(saved, target)['unknown.weight'] is saved['unknown.weight']
    current = {NATIVE_PREFIX+k: torch.empty(v) for k, v in NOISE_SHAPES.items()}
    assert map_legacy_noise_head(current, target) is current
    empty = {'motor.weight': torch.empty(2, 3)}
    assert map_legacy_noise_head(empty, empty) is empty


def test_complete_rl_heads_preserve_all_sixteen_saved_tensors():
    saved, target = states('noise')
    critic, native_critic = states('value')
    saved.update(critic); target.update(native_critic)
    mapped = map_legacy_rl_heads(saved, target)
    assert set(mapped) == set(target) and len(mapped) == len(saved) == 17
    for kind in ('noise', 'value'):
        old, new, shapes, _ = head(kind)
        for suffix in shapes:
            assert mapped[new+suffix] is saved[old+suffix]
    assert mapped['motor.weight'] is saved['motor.weight']
    saved['unknown.weight'] = torch.zeros(1)
    assert map_legacy_rl_heads(saved, target)['unknown.weight'] is saved['unknown.weight']


@pytest.mark.parametrize('prefix', [LEGACY_PREFIX, NATIVE_PREFIX, 'motor.'])
def test_header_selects_trained_noise_head_only_when_present(tmp_path, prefix):
    metadata = json.dumps({prefix+'weight': dict(shape=[1], dtype='F32', data_offsets=[0, 4])}).encode()
    path = tmp_path/'model.safetensors'
    path.write_bytes(struct.pack('<Q', len(metadata)) + metadata + b'\0'*4)
    assert checkpoint_has_noise_head(path) is (prefix != 'motor.')


@pytest.mark.parametrize('prefix', [LEGACY_VALUE_PREFIX, NATIVE_VALUE_PREFIX, 'motor.'])
def test_header_selects_saved_critic_only_when_present(tmp_path, prefix):
    metadata = json.dumps({prefix+'weight': dict(shape=[1], dtype='F32', data_offsets=[0, 4])}).encode()
    path = tmp_path/'model.safetensors'
    path.write_bytes(struct.pack('<Q', len(metadata)) + metadata + b'\0'*4)
    assert checkpoint_has_value_head(path) is (prefix != 'motor.')


@pytest.mark.parametrize('content', [b'', struct.pack('<Q', 1), struct.pack('<Q', 100)+b'{}',
                                     struct.pack('<Q', 32*1024*1024)])
def test_invalid_headers_fail_before_factory_selection(tmp_path, content):
    path = tmp_path/'broken.safetensors';path.write_bytes(content)
    with pytest.raises(ValueError): checkpoint_has_noise_head(path)


@pytest.mark.parametrize('trained_noise', [False, True])
@pytest.mark.parametrize('trained_value', [False, True])
def test_native_config_constructs_exactly_the_saved_auxiliary_heads(tmp_path, trained_noise, trained_value):
    from fluffy_bassoon.rl.maniskill_pi05 import model_config
    prefix = LEGACY_PREFIX if trained_noise else 'motor.'
    fields = {prefix+'weight': dict(shape=[1], dtype='F32', data_offsets=[0, 4])}
    if trained_value:
        fields[LEGACY_VALUE_PREFIX+'0.weight'] = dict(shape=[1024,2048], dtype='F32', data_offsets=[4,8])
    metadata = json.dumps(fields).encode()
    (tmp_path/'model.safetensors').write_bytes(struct.pack('<Q', len(metadata)) + metadata + b'\0'*4)
    cfg = model_config(tmp_path, tmp_path/'norm_stats.json')
    assert cfg.openpi.noise_method == ('flow_noise' if trained_noise else 'flow_ode')
    assert cfg.openpi.task == 'rl' and cfg.openpi.num_steps == 4
    assert cfg.openpi.action_horizon == 8 and cfg.openpi.action_chunk == 5
    assert cfg.add_value_head is trained_value and cfg.openpi.add_value_head is trained_value
    assert cfg.openpi.value_after_vlm is True and cfg.openpi.torch_compile is False
