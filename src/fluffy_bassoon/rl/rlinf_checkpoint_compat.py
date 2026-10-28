"""Lossless, fail-closed compatibility for released RLinf FlowNoise tensor names.

RLinf e1a82ebf renamed the trained exploration module from
``reinflow_explore_noise_net`` to ``noise_head``. The motor tensors and the
ExploreNoiseNet equations are unchanged. Native loading must still be strict.
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


def checkpoint_has_noise_head(path):
    """Read safetensors header only; do not create a random head for SFT."""
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
    metadata = json.loads(header)
    return any(name.startswith((LEGACY_PREFIX, NATIVE_PREFIX)) for name in metadata)


def map_legacy_noise_head(state, target):
    """Return an alias-only mapping; never drop, clone, cast or initialize tensors.

    Unknown motor/extra keys remain present for the native strict loader to
    reject. Partial, conflicting or changed exploration architectures fail here.
    """
    legacy = {key for key in state if key.startswith(LEGACY_PREFIX)}
    if not legacy:
        return state
    expected = {LEGACY_PREFIX + suffix for suffix in NOISE_SHAPES}
    if legacy != expected:
        raise ValueError('Require exactly the eight released legacy noise tensors')
    native = {key for key in target if key.startswith(NATIVE_PREFIX)}
    if native != {NATIVE_PREFIX + suffix for suffix in NOISE_SHAPES}:
        raise ValueError('Target must contain the exact native exploration module')
    if any(key.startswith(NATIVE_PREFIX) for key in state):
        raise ValueError('Conflicting legacy and native noise checkpoint keys')
    for suffix, shape in NOISE_SHAPES.items():
        value, reference = state[LEGACY_PREFIX + suffix], target[NATIVE_PREFIX + suffix]
        if tuple(value.shape) != shape or tuple(reference.shape) != shape:
            raise ValueError('Exploration tensor shape differs from the released model')
        if str(value.dtype) != 'torch.float32' or reference.dtype != value.dtype:
            raise ValueError('Exploration tensors must retain their saved FP32 dtype')
    mapped = dict(state)
    for suffix in NOISE_SHAPES:
        mapped[NATIVE_PREFIX + suffix] = mapped.pop(LEGACY_PREFIX + suffix)
    return mapped
