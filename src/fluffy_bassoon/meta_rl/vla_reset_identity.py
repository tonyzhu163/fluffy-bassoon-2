"""Explicit source40 reset identity with bounded uint8 rendering precision.

No pixel is changed. Physical state/proprio are byte-exact; camera tolerance
is one uint8 least-significant bit and is opt-in through the source contract.
"""
import hashlib
import numpy as np


RESET_IDENTITY = dict(format='source40_exact_physics_camera_lsb_v1', camera_max_uint8_lsb=1,
    nominal_state='exact SHA256', actual_simulator_state='exact dtype/shape/bytes',
    proprioception='exact dtype/shape/bytes', cameras='raw uint8; maximum per-channel difference1; no canonicalization',
    evidence='libero40-ppo-full-20261030-r2/reset-probe: four actual resets, identical92-float64 physical state and8-float32 proprio; two wrist channels at pixel(128,52) differ65/64 and61/60',
    original_observation_hashes=['ae9cf4684fc990e99197cd291b3f2d21177ce8d08e9255e946ce74e60683ff28',
        '360a1e661cb6ffddf67192ee1efe9b37af269190f5ffed81712ce9feaac62885'])


def _hash(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def _exact(first, second):
    first, second = np.asarray(first), np.asarray(second)
    return first.shape == second.shape and first.dtype == second.dtype and first.tobytes() == second.tobytes()


def comparison(first, second, *, camera_max_lsb):
    if camera_max_lsb != 1:
        raise ValueError('Only the explicit measured source40 one-LSB camera profile is supported')
    nominal = bool(first.get('initial_state_vector_sha256')) and first.get('initial_state_vector_sha256') == second.get('initial_state_vector_sha256')
    a, b = np.asarray(first['initial_simulator_state']), np.asarray(second['initial_simulator_state'])
    physical = (_exact(a, b) and np.isfinite(a).all()
        and _hash(a) == first.get('initial_simulator_state_sha256')
        and _hash(b) == second.get('initial_simulator_state_sha256'))
    proprio = (_exact(first['observation']['state'], second['observation']['state'])
        and np.asarray(first['observation']['state']).shape == (8,)
        and np.asarray(first['observation']['state']).dtype == np.float32
        and np.isfinite(first['observation']['state']).all())
    cameras = {}
    for name in ('image', 'wrist'):
        a, b = np.asarray(first['observation'][name]), np.asarray(second['observation'][name])
        if a.shape != b.shape or a.dtype != np.uint8 or b.dtype != np.uint8 or not a.size:
            raise ValueError('Reset camera comparison requires same-shape raw uint8 arrays')
        delta = b.astype(np.int16) - a.astype(np.int16)
        cameras[name] = dict(maximum_absolute_lsb=int(np.abs(delta).max()),
            changed_channels=int(np.count_nonzero(delta)), rms_lsb=float(np.sqrt(np.mean(delta.astype(np.float64) ** 2))),
            reference_sha256=_hash(a), current_sha256=_hash(b))
    return dict(allowed=bool(nominal and physical and proprio and all(row['maximum_absolute_lsb'] <= 1 for row in cameras.values())),
        exact_nominal_state=bool(nominal), exact_actual_simulator_state=bool(physical), exact_proprioception=bool(proprio),
        cameras=cameras, camera_max_uint8_lsb=1,
        reference_observation_sha256=first.get('initial_observation_sha256'),
        current_observation_sha256=second.get('initial_observation_sha256'),
        reference_actual_state_sha256=first.get('initial_simulator_state_sha256'),
        current_actual_state_sha256=second.get('initial_simulator_state_sha256'))


def episode_reset(rows, summary):
    return dict(initial_state_vector_sha256=summary['initial_state_vector_sha256'],
        initial_simulator_state=np.asarray(summary['initial_simulator_state'], dtype=np.float64),
        initial_simulator_state_sha256=summary['initial_simulator_state_sha256'],
        initial_observation_sha256=summary['initial_observation_sha256'], observation=rows[0]['motor_observation'])
