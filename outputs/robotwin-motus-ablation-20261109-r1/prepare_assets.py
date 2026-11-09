"""Owned asset view; repair only stale absolute native cuRobo file paths."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

SHARED_ASSETS = Path('/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RoboTwin/assets')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(destination, receipt):
    import yaml
    destination, receipt = Path(destination), Path(receipt)
    assert not destination.exists() and SHARED_ASSETS.is_dir()
    destination.mkdir()
    for source in SHARED_ASSETS.iterdir():
        target = destination / source.name
        if source.name != 'embodiments':
            target.symlink_to(source, target_is_directory=source.is_dir())
            continue
        target.mkdir()
        for robot in source.iterdir():
            if robot.name == 'aloha-agilex':
                shutil.copytree(
                    robot,
                    target / robot.name,
                    copy_function=lambda src, dst: Path(dst).symlink_to(Path(src).absolute()),
                )
            else:
                (target / robot.name).symlink_to(robot, target_is_directory=robot.is_dir())
    records = []
    for side in ('left', 'right'):
        source = SHARED_ASSETS / f'embodiments/aloha-agilex/curobo_{side}.yml'
        target = destination / f'embodiments/aloha-agilex/curobo_{side}.yml'
        original = yaml.safe_load(source.read_bytes())
        changed, paths = deepcopy(original), {}
        for key in ('urdf_path', 'collision_spheres'):
            old = original['robot_cfg']['kinematics'][key]
            marker = '/RoboTwin/assets/'
            assert str(old).count(marker) == 1
            relative = Path(str(old).split(marker)[1])
            assert '..' not in relative.parts
            actual = (SHARED_ASSETS / relative).resolve(strict=True)
            assert actual.is_file() and actual.is_relative_to(SHARED_ASSETS.resolve())
            if Path(old).exists():
                assert sha(old) == sha(actual)
            changed['robot_cfg']['kinematics'][key] = str(actual)
            paths[key] = dict(original=old, actual=str(actual), sha256=sha(actual))
        restored = deepcopy(changed)
        for key, row in paths.items():
            restored['robot_cfg']['kinematics'][key] = row['original']
        assert restored == original
        assert target.is_symlink()
        target.unlink()
        target.write_text(yaml.safe_dump(changed, sort_keys=False))
        records.append(
            dict(
                side=side,
                original_sha256=sha(source),
                runtime_sha256=sha(target),
                paths=paths,
                only_asset_paths_changed=True,
            )
        )
    config = destination / 'embodiments/aloha-agilex/config.yml'
    assert yaml.safe_load(config.read_bytes())['planner'] == 'curobo'
    receipt.write_text(
        json.dumps(
            dict(
                native_planner='curobo',
                robot_config_sha256=sha(config),
                shared_assets_unchanged=True,
                repairs=records,
            ),
            indent=2,
        )
        + '\n'
    )


if __name__ == '__main__':
    prepare(*sys.argv[1:])
