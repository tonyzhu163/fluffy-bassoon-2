from copy import deepcopy

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.robocasa_pairing import (
    canonical_case, compare_native_pair, public_snapshot, stable_counter_regions,
)


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
def test_pixel_variation_is_reported_without_uint8_wraparound_or_modification(left, right):
    a = snapshot(); b = deepcopy(a)
    a['images']['main_images'][0, 0, 0, 0] = left
    b['images']['main_images'][0, 0, 0, 0] = right
    before = deepcopy(b)
    report = compare_native_pair(a, b)
    assert report['matched'] and report['physical_state_exact'] and report['public_state_exact']
    assert report['image_comparison'] == 'diagnostic_only' and not report['images_exact']
    image = report['images']['main_images']
    assert image['max_absolute_delta'] == abs(left - right)
    assert image['differing_components'] == 1
    assert image['mean_absolute_delta'] == pytest.approx(abs(left - right) / 27)
    assert not report['observations_modified']
    assert np.array_equal(b['images']['main_images'], before['images']['main_images'])


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


def test_strict_pairing_rejects_pixel_differences_without_replacing_observations():
    a = snapshot(); b = deepcopy(a)
    b['images']['main_images'][0, 0, 0, 0] += 1
    before = deepcopy(b)
    with pytest.raises(ValueError, match='camera pixels differ'):
        compare_native_pair(a, b, require_images_exact=True)
    np.testing.assert_array_equal(b['images']['main_images'], before['images']['main_images'])


def test_shared_canonical_bank_rejects_other_arm_scene_and_preserves_mismatch(tmp_path):
    import pickle
    def save(path, value):
        path.write_bytes(pickle.dumps(value))
    def load(path):
        return pickle.loads(path.read_bytes())
    bank = tmp_path / 'shared' / 'case'
    a = snapshot()
    marker, _ = canonical_case(bank, a, save=save, load=load, source={'arm': 'velocity'},
                               require_images_exact=True)
    other_marker, _ = canonical_case(bank, a, save=save, load=load, source={'arm': 'kv'},
                                     require_images_exact=True)
    assert marker == other_marker
    b = deepcopy(a); b['physical']['xml_sha256'] = 'different-placement'
    with pytest.raises(ValueError, match='physical'):
        canonical_case(bank, b, save=save, load=load, source={'arm': 'kv'}, require_images_exact=True)
    failed = list(bank.glob('mismatch-*.pt'))
    assert len(failed) == 1 and load(failed[0])['snapshot']['physical'] == b['physical']
    assert load(bank / 'initial.pt')['physical'] == a['physical']


def test_counter_region_choice_is_independent_of_upstream_candidate_order():
    class Counter:
        def __init__(self, reverse):
            self.reverse = reverse
            self.rng = np.random.default_rng(39)
        def get_reset_regions(self):
            candidates = [dict(size=[.4, .5], offset=[-1., 0., 1.]),
                          dict(size=np.array([.4, .5]), offset=np.array([1., 0., 1.]))]
            if self.reverse:
                candidates.reverse()
            return {f'geom_{i}': value for i, value in enumerate(candidates)}
    stable_counter_regions(Counter)
    original = Counter.get_reset_regions
    stable_counter_regions(Counter)
    assert Counter.get_reset_regions is original  # Each reset child can call it safely.
    left, right = Counter(False), Counter(True)
    state = deepcopy(left.rng.bit_generator.state)
    a, b = left.get_reset_regions(), right.get_reset_regions()
    assert left.rng.bit_generator.state == state
    for key in a:
        np.testing.assert_array_equal(a[key]['offset'], b[key]['offset'])
    for _ in range(10):
        x, y = left.rng.choice(list(a.values())), right.rng.choice(list(b.values()))
        np.testing.assert_array_equal(x['offset'], y['offset'])
