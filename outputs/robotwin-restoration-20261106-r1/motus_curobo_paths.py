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
    old = Path(original['robot_cfg']['kinematics']['urdf_path'])
    assert old.is_absolute() and not old.exists()
    marker = '/RoboTwin/assets/embodiments/'
    assert str(old).count(marker) == 1
    relative = Path('assets/embodiments')/str(old).split(marker)[1]
    assert '..' not in relative.parts
    actual = (robotwin_root/relative).resolve(strict=True)
    assert actual.is_file() and actual.is_relative_to((robotwin_root/'assets').resolve())
    changed['robot_cfg']['kinematics']['urdf_path'] = str(actual)
    restored = deepcopy(changed)
    restored['robot_cfg']['kinematics']['urdf_path'] = str(old)
    assert restored == original, 'Only the missing URDF path may change'
    output.parent.mkdir(parents=True,exist_ok=True)
    encoded = yaml.safe_dump(changed,sort_keys=False).encode()
    if output.exists(): assert output.read_bytes() == encoded
    else:
        with output.open('xb') as stream: stream.write(encoded)
    assert native_yaml.read_bytes() == original_bytes
    return dict(native_yaml=str(native_yaml), native_yaml_sha256=hashlib.sha256(original_bytes).hexdigest(),
        runtime_yaml=str(output), runtime_yaml_sha256=hashlib.sha256(encoded).hexdigest(),
        original_missing_urdf=str(old), actual_urdf=str(actual),
        actual_urdf_sha256=hashlib.sha256(actual.read_bytes()).hexdigest(),
        only_urdf_path_changed=True, native_yaml_unchanged=True)
