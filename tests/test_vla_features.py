import numpy as np
import pytest

from fluffy_bassoon.meta_rl.vla_features import difference_metrics, partition_differences, pool_prefix


def test_separate_visual_language_pooling_ignores_padding_and_keeps_public_state():
    hidden = np.array([[[1., 2.], [3., 4.], [900., 900.],
                         [10., 20.], [30., 40.], [900., 900.]]])
    mask = np.array([[True, True, False, True, True, False]])
    state = np.arange(32)[None]
    result = pool_prefix(hidden, mask, 3, state)
    assert np.array_equal(result, np.array([[2., 3., 20., 30., 0., 1., 2., 3., 4., 5., 6., 7.]]))


def test_empty_visual_or_language_partition_is_rejected():
    with pytest.raises(ValueError):
        pool_prefix(np.zeros((1, 3, 4)), np.ones((1, 3)), 3, np.zeros((1, 8)))


def test_numerical_diagnostics_preserve_partition_and_exactness():
    reference = np.ones(12, np.float32)
    candidate = reference.copy()
    candidate[1] += .25
    report = partition_differences(reference, candidate, 2)
    assert report["visual"]["maximum_absolute_error"] == .25
    assert report["visual"]["rms_error"] == pytest.approx(np.sqrt(.25 ** 2 / 2))
    assert report["language"]["exactly_equal"]
    assert report["state"]["exactly_equal"]
    with pytest.raises(ValueError, match="Non-finite"):
        difference_metrics(reference, np.full_like(reference, np.nan))
    with pytest.raises(ValueError, match="matching nonempty"):
        difference_metrics(reference, np.ones(3))
