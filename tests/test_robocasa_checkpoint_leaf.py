"""Numerical preservation of native NNX Orbax leaf wrappers; allocated Linux."""
import numpy as np
from rlinf.utils.ckpt_convertor.openpi.jax_to_openpi import _unwrap_nnx_values


def test_nnx_wrappers_become_array_leaves_without_changing_any_weight():
    kernel = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    bias = np.linspace(-1, 1, 4, dtype=np.float32)
    source = {"params": {"PaliGemma": {"img": {"embedding": {
        "kernel": {"value": kernel}, "bias": {"value": bias}}}},
        "already_plain": bias, "ordinary_dict": {"value": kernel, "other": bias}}}
    actual = _unwrap_nnx_values(source)
    assert actual["params"]["PaliGemma"]["img"]["embedding"]["kernel"] is kernel
    assert actual["params"]["PaliGemma"]["img"]["embedding"]["bias"] is bias
    np.testing.assert_array_equal(actual["params"]["PaliGemma"]["img"]["embedding"]["kernel"], np.arange(24, dtype=np.float32).reshape(2, 3, 4))
    assert actual["params"]["already_plain"] is bias
    assert actual["params"]["ordinary_dict"]["value"] is kernel
    assert source["params"]["PaliGemma"]["img"]["embedding"]["kernel"] == {"value": kernel}
