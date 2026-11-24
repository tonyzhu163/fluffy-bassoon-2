"""Render all genuine poses of one trajectory in a single primary sensor replay."""
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
from selection import ARCHIVE, HERE, select

sys.path.insert(0,str(HERE.parent/'symmeta-camera-pose-pairs-20261124-r1'))
import render
from geometry import dump, pair, project
from world_control import file_sha

ANGLES = {'train':(0,-12,12,-24,24,-42,42,-49,49), 'val':(0,-18,18,-45,45)}


def render_one(out, case):
    angles = ANGLES[case['split']]
    # collect() writes the nominal tape and checks all 15 controls. Its sensor
    # callback renders every moved view at exactly the matched observation clock.
    names = {'nominal':0, **{f'orbit{a:+d}':a for a in angles if a}}
    frames = []
    class SpatialViews(render.Views):
        def extrinsics(self, name, pos, mat):
            rot = render.rz(names[name])
            return self.pivot + rot @ (pos-self.pivot), rot @ mat

        def check_marks(self, world, calib):
            # At wide azimuths the original world-triplet can overlap in projection.
            # Place a visible non-collinear triangle in the actual MuJoCo camera
            # frame; the primary independent rasterizer still verifies calibration.
            gl=np.asarray(self.sim.data._data.cam_xmat[self.cam]).reshape(3,3)
            center=np.asarray(self.sim.data._data.cam_xpos[self.cam])
            markers=center+np.array([[-.09,-.06,-.65],[.02,.09,-.72],[.10,-.07,-.78]])@gl.T
            return super().check_marks(markers,calib)

        def record(self, nominal, settings):
            with patch.object(render,'VIEWS',tuple(names)):
                return super().record(nominal,settings)

        def match(self, raw):
            frame = super().match(raw)
            frames.append(frame)
            return frame

    entry = dict(index=case['index'],physics=case['physics'],embodiment=case['embodiment'],
                 passive=ARCHIVE.parent/case['passive_path'])
    keys = ('shard','suite','task_id','init_state_id','seed')
    with patch.object(render,'Views',SpatialViews), patch.object(render,'VIEWS',('nominal',)):
        row,diag = render.collect(out,case['arm'],{k:case[k] for k in keys},entry,None)
    assert len(frames)==4 and row['id']==case['id'] and row['frame_steps']==[0,5,10,15]
    folder = out/row['id']
    views = row['views']
    world = np.asarray(diag['world_tcp_m'])
    ref = frames[0]['camera']['nominal']
    xyz_ref = project(ref['P'],world)
    for name,angle in names.items():
        cam = frames[0]['camera'][name]
        assert all(f['camera'][name]==cam for f in frames)
        rgb = np.stack([f['images'][name] for f in frames])
        if angle:
            path=folder/f'{name}.npz'
            np.savez_compressed(path,frames=rgb)
            views[name]=dict(angle_deg=angle,frames_npz=str(path),sha256=file_sha(path),**cam,**pair(ref,cam,row['plane_z_m']))
        v=views[name]
        xyz_view=project(cam['P'],world)
        v['eef_registered_residual_px']=np.linalg.norm(project(v['H_view_to_ref'],xyz_view)-xyz_ref,axis=1).tolist()
        v['eef_unregistered_residual_px']=np.linalg.norm(xyz_view-xyz_ref,axis=1).tolist()
    assert len(diag['landmarks'])==len(names)
    row['split']=case['split']
    dump(folder/'diag.json',diag)
    dump(folder/'row.json',row)
    print(row['id'],case['split'],len(views),flush=True)


def main():
    out=Path(sys.argv[1])
    slot=int(sys.argv[2])
    assert os.sched_getaffinity(0)=={42+slot} and not os.environ['CUDA_VISIBLE_DEVICES']
    assert select(ARCHIVE.read_bytes())==json.loads((HERE/'source-ids.json').read_bytes())
    render.setup(render.SOURCE)
    for c in select(ARCHIVE.read_bytes())['rows'][slot::4]:
        render_one(out,c)


if __name__=='__main__':
    main()
