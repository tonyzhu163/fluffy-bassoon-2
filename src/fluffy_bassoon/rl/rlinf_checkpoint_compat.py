"""Lossless, fail-closed compatibility for released RLinf auxiliary heads.

RLinf e1a82ebf renamed the trained exploration module from
``reinflow_explore_noise_net`` to ``noise_head``. The motor tensors and the
ExploreNoiseNet equations are unchanged. Native loading must still be strict.
The saved critic uses ``value_proj.value_proj`` rather than ``value_head.mlp``.
"""
import json
from pathlib import Path
import struct


LEGACY_PREFIX = 'reinflow_explore_noise_net.'
NATIVE_PREFIX = 'noise_head.'
NOISE_SHAPES = {
    'logvar_min': (1,), 'logvar_max': (1,),
    'mlp_logvar.moduleList.0.linear_1.weight': (128, 1024),
    'mlp_logvar.moduleList.0.linear_1.bias': (128,),
    'mlp_logvar.moduleList.1.linear_1.weight': (64, 128),
    'mlp_logvar.moduleList.1.linear_1.bias': (64,),
    'mlp_logvar.moduleList.2.linear_1.weight': (32, 64),
    'mlp_logvar.moduleList.2.linear_1.bias': (32,),
}
LEGACY_VALUE_PREFIX = 'value_proj.value_proj.'
NATIVE_VALUE_PREFIX = 'value_head.mlp.'
VALUE_SHAPES = {
    '0.weight': (1024, 2048), '0.bias': (1024,),
    '2.weight': (512, 1024), '2.bias': (512,),
    '4.weight': (256, 512), '4.bias': (256,),
    '6.weight': (1, 256), '6.bias': (1,),
}


def checkpoint_metadata(path):
    """Read a bounded safetensors header without loading tensor data."""
    with Path(path).open('rb') as stream:
        length = stream.read(8)
        if len(length) != 8:
            raise ValueError('Truncated safetensors header length')
        size = struct.unpack('<Q', length)[0]
        if not 2 <= size <= 16 * 1024 * 1024:
            raise ValueError('Invalid safetensors header size')
        header = stream.read(size)
        if len(header) != size:
            raise ValueError('Truncated safetensors header')
    return json.loads(header)


def checkpoint_has_noise_head(path):
    return any(name.startswith((LEGACY_PREFIX, NATIVE_PREFIX))
               for name in checkpoint_metadata(path))


def checkpoint_has_value_head(path):
    return any(name.startswith((LEGACY_VALUE_PREFIX, NATIVE_VALUE_PREFIX))
               for name in checkpoint_metadata(path))


def _map_head(state, target, old_prefix, new_prefix, shapes):
    """Return an alias-only mapping; never drop, clone, cast or initialize tensors.

    Unknown motor/extra keys remain present for the native strict loader to
    reject. Partial, conflicting or changed exploration architectures fail here.
    """
    legacy = {key for key in state if key.startswith(old_prefix)}
    if not legacy:
        return state
    expected = {old_prefix + suffix for suffix in shapes}
    if legacy != expected:
        raise ValueError('Require the complete known legacy head: '+old_prefix)
    native = {key for key in target if key.startswith(new_prefix)}
    if native != {new_prefix + suffix for suffix in shapes}:
        raise ValueError('Target must contain the exact native module: '+new_prefix)
    if any(key.startswith(new_prefix) for key in state):
        raise ValueError('Conflicting legacy and native head checkpoint keys')
    for suffix, shape in shapes.items():
        value, reference = state[old_prefix + suffix], target[new_prefix + suffix]
        if tuple(value.shape) != shape or tuple(reference.shape) != shape:
            raise ValueError('Auxiliary tensor shape differs from the released model')
        if str(value.dtype) != 'torch.float32' or reference.dtype != value.dtype:
            raise ValueError('Auxiliary tensors must retain their saved FP32 dtype')
    mapped = dict(state)
    for suffix in shapes:
        mapped[new_prefix + suffix] = mapped.pop(old_prefix + suffix)
    return mapped


def map_legacy_noise_head(state, target):
    return _map_head(state, target, LEGACY_PREFIX, NATIVE_PREFIX, NOISE_SHAPES)


def map_legacy_value_head(state, target):
    return _map_head(state, target, LEGACY_VALUE_PREFIX, NATIVE_VALUE_PREFIX, VALUE_SHAPES)


def map_legacy_rl_heads(state, target):
    """Alias the sixteen saved RL-head tensors; retain every other tensor."""
    return map_legacy_value_head(map_legacy_noise_head(state, target), target)
