"""Check actual native config/URDF paths on Linux CPU, without a simulator."""
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from motus_curobo_paths import prepare


def main():
    assert os.uname().sysname == 'Linux' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    robotwin = Path(os.environ['ROBOTWIN_PATH'])
    root = Path('/mowenjie/fluffy-bassoon-2/robotwin-curobo-20261108-paths01')
    cold = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1/cpumotuscurobopaths01')
    assert not root.exists() and not cold.exists()
    root.mkdir(); cold.mkdir()
    rows = [prepare(robotwin/'assets/embodiments/aloha-agilex'/f'curobo_{arm}.yml',
        robotwin,root/f'curobo_{arm}.yml') for arm in ('left','right')]
    meshes = {}
    for row in rows:
        urdf = Path(row['actual_urdf'])
        tree = ET.fromstring(urdf.read_bytes())
        for mesh in tree.findall('.//mesh'):
            filename = mesh.attrib['filename']
            # Preserve semantics and fail if an asset needs an additional path repair.
            assert not filename.startswith('package://'), filename
            actual = (urdf.parent/filename).resolve(strict=True)
            assert actual.is_file()
            meshes[str(actual)] = hashlib.sha256(actual.read_bytes()).hexdigest()
    assert meshes and rows[0]['actual_urdf_sha256'] == rows[1]['actual_urdf_sha256']
    result = dict(status='passed_run_local_native_config_path_repair', rows=rows,
        referenced_meshes=meshes, GPU_seconds=0, simulator_scenes_created=0,
        robot_geometry_changed=False, expert_success_qualified=False,
        native_shared_assets_modified=False, source_commit=os.environ['TASK_SOURCE'])
    path = cold/'RESULT.json'
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result['status'],len(meshes),hashlib.sha256(path.read_bytes()).hexdigest(),flush=True)


if __name__ == '__main__': main()
