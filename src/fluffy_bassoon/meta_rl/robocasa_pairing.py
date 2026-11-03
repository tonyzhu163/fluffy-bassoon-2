"""Pair native cases by physical state and record renderer differences.

The simulator and motor receive their original observations. This module only
records and verifies them; a transformer feature hash is not a scene identity.
"""
from copy import deepcopy
from functools import wraps
import hashlib
import json
from pathlib import Path
import time

import numpy as np

FORMAT = 'robocasa_native_pair_v1'
IMAGE_KEYS = ('main_images', 'wrist_images', 'extra_view_images')
MODEL_ARRAYS = ('tex_rgb', 'tex_data', 'mat_rgba', 'geom_rgba', 'geom_pos', 'geom_quat',
                'body_pos', 'body_quat', 'light_pos', 'cam_pos', 'cam_quat',
                'mesh_vert', 'mesh_normal', 'mesh_face', 'mesh_texcoord')


def array_ref(value):
    array = np.asarray(value)
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                sha256=hashlib.sha256(array.tobytes()).hexdigest())


def native_fingerprint(env):
    model = env.sim.model
    return dict(xml_sha256=hashlib.sha256(model.get_xml().encode()).hexdigest(),
                state=array_ref(env.sim.get_state().flatten()),
                camera_position=array_ref(env.sim.data.cam_xpos),
                camera_matrix=array_ref(env.sim.data.cam_xmat),
                compiled={key: array_ref(getattr(model, key)) for key in MODEL_ARRAYS if hasattr(model, key)})


def public_snapshot(observation, case, physical):
    def array(value):
        if value is None:
            return None
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        return np.array(value, copy=True)
    return dict(format=FORMAT, case=deepcopy(case), physical=deepcopy(physical),
                states=array(observation['states']),
                task_descriptions=list(observation['task_descriptions']),
                images={key: array(observation.get(key)) for key in IMAGE_KEYS})


def compare_native_pair(reference, current, *, require_images_exact=False):
    for key in ('format', 'case', 'physical', 'task_descriptions'):
        if reference[key] != current[key]:
            raise ValueError(f'Native paired case differs: {key}')
    if reference['format'] != FORMAT:
        raise ValueError('Unknown native pairing snapshot')
    a, b = reference['states'], current['states']
    if (a.shape != b.shape or a.dtype != b.dtype or not np.isfinite(a).all()
            or not np.isfinite(b).all() or not np.array_equal(a, b)):
        raise ValueError('Native paired public state differs')
    if set(reference['images']) != set(IMAGE_KEYS) or set(current['images']) != set(IMAGE_KEYS):
        raise ValueError('Native paired camera set differs')
    report = {}
    for key in IMAGE_KEYS:
        a, b = reference['images'][key], current['images'][key]
        if a is None or b is None:
            raise ValueError(f'Native paired camera presence differs or required view missing: {key}')
        if (a.shape != b.shape or a.dtype != np.uint8 or b.dtype != np.uint8
                or a.ndim != (5 if key == 'extra_view_images' else 4) or a.shape[-1] != 3 or not a.size):
            raise ValueError(f'Native paired camera shape/dtype differs: {key}')
        delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
        maximum = int(delta.max()) if delta.size else 0
        report[key] = dict(max_absolute_delta=maximum, differing_components=int(np.count_nonzero(delta)),
                           total_components=int(delta.size), mean_absolute_delta=float(delta.mean()) if delta.size else 0.)
    # Native state, scene assets and camera geometry establish case identity.
    # Renderer arithmetic can vary across processes/GPUs even when those match.
    # Preserve the actual policy pixels and report their differences, rather
    # than aborting a paired experiment on a maximum-pixel threshold.
    images_exact = all(value['max_absolute_delta'] == 0 for value in report.values())
    if require_images_exact and not images_exact:
        raise ValueError('Native paired initial camera pixels differ; preserve both snapshots')
    return dict(matched=True, physical_state_exact=True, public_state_exact=True,
                image_comparison='exact_required' if require_images_exact else 'diagnostic_only',
                images_exact=images_exact,
                images=report, observations_modified=False)


def stable_counter_regions(counter_class=None):
    """Canonicalize Counter region candidates before its seeded region choice.

    Upstream enumerates XML geometry objects through a set. Object identity
    hashes make region order process-dependent even with seeded RNGs. Sort the
    geometric candidates and renumber only generated geom_N names; retain all
    regions and leave the RNG untouched. No vendor file is modified.
    """
    if counter_class is None:
        from robocasa.models.fixtures.counter import Counter
        counter_class = Counter
    original = counter_class.get_reset_regions
    if getattr(original, '_stable_region_order', False):
        return

    @wraps(original)
    def ordered(self, *args, **kwargs):
        regions = original(self, *args, **kwargs)
        def key(item):
            return json.dumps(item[1], sort_keys=True, separators=(',', ':'),
                              default=lambda value: np.asarray(value).tolist(), allow_nan=False)
        candidates = sorted(regions.items(), key=key)
        if all(name.startswith('geom_') and name[5:].isdigit() for name in regions):
            return {f'geom_{i}': value for i, (_, value) in enumerate(candidates)}
        return dict(candidates)

    ordered._stable_region_order = True
    counter_class.get_reset_regions = ordered


def pairing_native_factory(cfg, *, stable_regions=False):
    """Original native case routing plus an observation-only fingerprint.

    Seed global RNGs in the child as well as the native environment RNG. Native
    decorative geometry colors otherwise depend on fresh-process entropy.
    """
    def factory(case):
        from omegaconf import OmegaConf
        from rlinf.envs.sim.robocasa365.robocasa365_env import Robocasa365Env
        from rlinf.envs.sim.robocasa.venv import RobocasaSubprocEnv
        from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import TASKS
        from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo_env import counted_native_factory

        def captured(fn, horizon):
            original = counted_native_factory(fn, physics_tick_limit=(horizon + 128) * 128)
            def create():
                import random
                random.seed(int(case['native_seed']))
                np.random.seed(int(case['native_seed']))
                if stable_regions:
                    stable_counter_regions()
                env = original()
                reset = env.reset
                def observe_reset(*args, **kwargs):
                    result = reset(*args, **kwargs)
                    env._native_pairing = native_fingerprint(env)
                    if stable_regions:
                        env._native_pairing['placement_protocol'] = 'stable_counter_regions_v1'
                    return result
                env.reset = observe_reset
                return env
            return create

        class CountedNative(Robocasa365Env):
            def _init_env(self):
                self._refresh_task_context()
                horizon = int(self.task_horizons[0])
                self.env = RobocasaSubprocEnv([captured(fn, horizon) for fn in self.get_env_fns()])

        config = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        config.seed = int(case['native_seed'])
        config.seed_strategy = 'same'
        config.auto_reset, config.ignore_terminations = False, True
        config.task_filter = [case['task']]
        horizon, soup, mode, _ = TASKS[case['task']]
        config.task_soup, config.task_mode, config.split = soup, mode, 'target'
        config.max_episode_steps = config.max_steps_per_rollout_epoch = horizon
        return CountedNative(config, num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None)
    return factory


def canonical_case(directory, snapshot, *, save, load, source, require_images_exact=False):
    """Publish once; concurrent model phases share the same fixed case bank."""
    compare_native_pair(snapshot, snapshot)
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    marker = directory / 'complete.json'
    try:
        directory.mkdir()
        owner = True
    except FileExistsError:
        owner = False
    if owner:
        (directory / 'owner.json').write_text(json.dumps(source, indent=2) + '\n')
        save(directory / 'initial.pt', snapshot)
        value = dict(case=snapshot['case'], source=source,
                     snapshot_sha256=hashlib.sha256((directory / 'initial.pt').read_bytes()).hexdigest())
        temporary = directory / 'complete.tmp'
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(marker)
    else:
        end = time.monotonic() + 120
        while not marker.exists():
            if time.monotonic() >= end:
                raise RuntimeError('Incomplete canonical case: preserve owner and diagnose before recovery')
            time.sleep(.1)
    value = json.loads(marker.read_text())
    if (value['case'] != snapshot['case']
            or hashlib.sha256((directory / 'initial.pt').read_bytes()).hexdigest() != value['snapshot_sha256']):
        raise ValueError('Canonical native case integrity differs')
    reference = load(directory / 'initial.pt')
    try:
        comparison = compare_native_pair(reference, snapshot, require_images_exact=require_images_exact)
    except ValueError:
        save(directory / f'mismatch-{time.time_ns()}.pt', dict(snapshot=snapshot, source=source))
        raise
    return marker, comparison
