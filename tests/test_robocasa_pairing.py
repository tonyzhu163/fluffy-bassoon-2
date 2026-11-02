from copy import deepcopy

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.robocasa_pairing import compare_native_pair, public_snapshot


def snapshot():
    return public_snapshot(dict(states=np.zeros((1, 25), np.float32), task_descriptions=['Close the fridge door.'],
        main_images=np.full((1, 3, 3, 3), 100, np.uint8), wrist_images=np.full((1, 3, 3, 3), 100, np.uint8),
        extra_view_images=np.full((1, 1, 3, 3, 3), 100, np.uint8)),
        dict(task='CloseFridge', native_seed=17), dict(xml_sha256='scene', state='state', camera_matrix='pose'))


def test_one_level_render_roundoff_preserves_original_observation():
    a = snapshot(); b = deepcopy(a)
    b['images']['wrist_images'][0, 1, 1, 0] += 1
    before = deepcopy(b)
    report = compare_native_pair(a, b)
    assert report['physical_state_exact'] and report['public_state_exact']
    assert report['images']['wrist_images']['differing_components'] == 1
    assert not report['observations_modified']
    assert np.array_equal(b['images']['wrist_images'], before['images']['wrist_images'])


@pytest.mark.parametrize('key', ['case', 'physical', 'task_descriptions'])
def test_changed_physical_case_or_prompt_is_rejected(key):
    a = snapshot(); b = deepcopy(a); b[key] = {'changed': True}
    with pytest.raises(ValueError, match='Native paired case differs'):
        compare_native_pair(a, b)


def test_changed_public_state_is_rejected_even_with_matching_images():
    a = snapshot(); b = deepcopy(a); b['states'][0, 0] = .001
    with pytest.raises(ValueError, match='public state'):
        compare_native_pair(a, b)


@pytest.mark.parametrize('left,right', [(100, 102), (255, 0)])
def test_larger_pixel_change_is_rejected_without_uint8_wraparound(left, right):
    a = snapshot(); b = deepcopy(a)
    a['images']['main_images'][0, 0, 0, 0] = left
    b['images']['main_images'][0, 0, 0, 0] = right
    with pytest.raises(ValueError, match='render roundoff'):
        compare_native_pair(a, b)


def test_camera_geometry_or_presence_change_is_rejected():
    a = snapshot(); b = deepcopy(a); b['images']['main_images'] = np.zeros((1, 4, 3, 3), np.uint8)
    with pytest.raises(ValueError, match='shape/dtype'):
        compare_native_pair(a, b)
    b = deepcopy(a); b['images']['wrist_images'] = None
    with pytest.raises(ValueError, match='presence'):
        compare_native_pair(a, b)


def test_missing_view_cannot_become_a_valid_reference():
    a = snapshot(); a['images']['extra_view_images'] = None
    with pytest.raises(ValueError, match='required view missing'):
        compare_native_pair(a, a)
