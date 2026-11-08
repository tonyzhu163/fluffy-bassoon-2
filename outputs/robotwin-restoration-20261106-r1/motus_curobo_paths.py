"""Run-local path repair for the unchanged native cuRobo expert configuration."""
from copy import deepcopy
import hashlib
from pathlib import Path


def prepare(native_yaml, robotwin_root, output):
    import yaml
    native_yaml, robotwin_root, output = map(Path, (native_yaml, robotwin_root, output))
    original_bytes = native_yaml.read_bytes()
    original = yaml.safe_load(original_bytes)
    changed = deepcopy(original)
    kinematics = original['robot_cfg']['kinematics']
    repaired = {}
    for key in ('urdf_path', 'collision_spheres'):
        old = Path(kinematics[key])
        assert old.is_absolute() and not old.exists()
        marker = '/RoboTwin/assets/embodiments/'
        assert str(old).count(marker) == 1
        relative = Path('assets/embodiments')/str(old).split(marker)[1]
        assert '..' not in relative.parts
        actual = (robotwin_root/relative).resolve(strict=True)
        assert actual.is_file() and actual.is_relative_to((robotwin_root/'assets').resolve())
        changed['robot_cfg']['kinematics'][key] = str(actual)
        repaired[key] = dict(original=str(old),actual=str(actual),sha256=hashlib.sha256(actual.read_bytes()).hexdigest())
    restored = deepcopy(changed)
    for key in repaired: restored['robot_cfg']['kinematics'][key] = kinematics[key]
    assert restored == original, 'Only missing asset paths may change'
    # Inspect every nested absolute path, including collision configuration.
    def verify_paths(value):
        if isinstance(value,dict):
            for child in value.values(): verify_paths(child)
        elif isinstance(value,list):
            for child in value: verify_paths(child)
        elif isinstance(value,str) and value.startswith('/'):
            assert Path(value).exists(), value
    verify_paths(changed)
    collision = yaml.safe_load(Path(repaired['collision_spheres']['actual']).read_bytes())
    verify_paths(collision)
    assert isinstance(collision['collision_spheres'],dict) and collision['collision_spheres']
    output.parent.mkdir(parents=True,exist_ok=True)
    encoded = yaml.safe_dump(changed,sort_keys=False).encode()
    if output.exists(): assert output.read_bytes() == encoded
    else:
        with output.open('xb') as stream: stream.write(encoded)
    assert native_yaml.read_bytes() == original_bytes
    return dict(native_yaml=str(native_yaml), native_yaml_sha256=hashlib.sha256(original_bytes).hexdigest(),
        runtime_yaml=str(output), runtime_yaml_sha256=hashlib.sha256(encoded).hexdigest(),
        original_missing_urdf=repaired['urdf_path']['original'], actual_urdf=repaired['urdf_path']['actual'],
        actual_urdf_sha256=repaired['urdf_path']['sha256'],
        actual_collision_yaml=repaired['collision_spheres']['actual'],
        actual_collision_yaml_sha256=repaired['collision_spheres']['sha256'],
        repaired_asset_paths=repaired, only_asset_paths_changed=True, native_yaml_unchanged=True)
