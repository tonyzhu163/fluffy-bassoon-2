"""Make a lossless state-only working copy of the three original LIBERO HDF5s."""
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np

FILES = {
    "goal3": ("open_the_top_drawer_and_put_the_bowl_inside_demo.hdf5",
              "1b50d99a053a7e4547c9ae9ccd2e3cd762543d27be78f240f13440d7ae487ae8"),
    "long8": ("KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5",
              "e329bb21a8ded3457854faf6a23513c90cb4b34f0e40f3f4e9e70451fc9ba504"),
    "goal6": ("put_the_cream_cheese_in_the_bowl_demo.hdf5",
              "ee58a7d72c143a71bf1448674e70940ac7781c1d1bb10a1b0c7a5c69b26abbef"),
}


def pack(directory, task, destination):
    name, sha = FILES[task]
    path = directory / name
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != sha:
        raise ValueError(path)
    destination.mkdir(exist_ok=True, parents=True)
    with h5py.File(path) as src, h5py.File(destination / (task + ".h5"), "w") as dst:
        dst.attrs.update(source=name, sha256=sha, revision="f13aa24a3da8c43c7225569f28c562979fa0e35a")
        for name in ("demo_0", "demo_1"):
            demo = src["data"][name]
            group = dst.create_group(name)
            model = ET.fromstring(demo.attrs["model_file"]).find("worldbody")
            joints = [el.get("name") for el in model.iter() if el.tag in ("joint", "freejoint")]
            group.attrs["joint_order"] = json.dumps(joints)
            group.attrs["world_bodies"] = json.dumps({el.get("name"): {
                "pos": el.get("pos", "0 0 0"), "quat": el.get("quat", "1 0 0 0")}
                for el in model.findall("body")})
            group.attrs["terminal_reward"] = float(demo["rewards"][-1])
            group.attrs["terminal_done"] = bool(demo["dones"][-1])
            for field in ("states", "obs/ee_pos", "obs/ee_ori", "obs/joint_states", "obs/gripper_states"):
                group.create_dataset(field, data=demo[field][:], compression="gzip", compression_opts=1)
            if not np.array_equal(group["states"][0], demo.attrs["init_state"]):
                raise ValueError(f"Inconsistent first state: {name}")
    print(task, sha, flush=True)


if __name__ == "__main__":
    pack(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]) if len(sys.argv)>3 else Path(sys.argv[1]))
