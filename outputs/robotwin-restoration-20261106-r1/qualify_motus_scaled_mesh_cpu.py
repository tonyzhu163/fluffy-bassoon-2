"""Actual CPU PhysX/FCL geometry versus analytic box collision/distance reference."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import sapien
from sapien.physx import PhysxCpuSystem, PhysxRigidDynamicComponent, PhysxCollisionShapeConvexMesh, PhysxMaterial
from mplib.sapien_utils import SapienPlanningWorld
from mplib.collision_detection.fcl import FCLObject, CollisionObject, Sphere, DistanceRequest, distance, collide
from mplib import Pose
import torch

from motus_scaled_mesh import scaled_converter

ROOT=Path(__file__).resolve().parent
BASE=Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')


def main(identity):
    assert os.uname().sysname=='Linux' and os.environ['CUDA_VISIBLE_DEVICES']==''
    assert not torch.cuda.is_initialized()
    code=ROOT.parents[1];source=os.environ['TASK_SOURCE']
    assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==source
    request=DistanceRequest(distance_tolerance=1e-10)
    original=SapienPlanningWorld.convert_physx_component
    digest=hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
    fixed=scaled_converter(original,digest)
    # Physics-only CPU scene: no renderer and no task/model instance.
    scene=sapien.Scene([PhysxCpuSystem()])
    material=PhysxMaterial(.5,.5,0.)
    cube='''v -1 -1 -1
v 1 -1 -1
v 1 1 -1
v -1 1 -1
v -1 -1 1
v 1 -1 1
v 1 1 1
v -1 1 1
f 1 3 2
f 1 4 3
f 5 6 7
f 5 7 8
f 1 2 6
f 1 6 5
f 4 8 7
f 4 7 3
f 1 5 8
f 1 8 4
f 2 3 7
f 2 7 6
'''
    rows=[]
    with tempfile.TemporaryDirectory(prefix='rt-scaled-mesh-') as tmp:
        mesh=Path(tmp)/'cube.obj';mesh.write_text(cube)
        for scale in ([1.,1.,1.],[.078,.078,.078],[.078,.156,.039]):
            shape=PhysxCollisionShapeConvexMesh(str(mesh),np.asarray(scale,dtype=np.float32),material)
            body=PhysxRigidDynamicComponent();body.attach(shape)
            entity=sapien.Entity();entity.name='scaled_geometry_fixture';entity.add_component(body);scene.add_entity(entity)
            converted=fixed(body)
            if scale==[1.,1.,1.]: upstream=original(body)
            else:
                try:original(body)
                except AssertionError:pass
                else:raise AssertionError('Expected native nonunit scale rejection')
            for axis in range(3):
                point=np.zeros(3);point[axis]=3.
                sphere=FCLObject('probe',Pose(p=point),[CollisionObject(Sphere(.05))],[Pose()])
                measured=float(distance(converted,sphere,request).min_distance)
                expected=3.-float(shape.scale[axis])-.05
                assert abs(measured-expected)<2e-6,(scale,axis,measured,expected)
                if scale==[1.,1.,1.]:assert float(distance(upstream,sphere,request).min_distance)==measured
                for offset,expected_hit in ((.025,True),(.1,False)):
                    point[axis]=float(shape.scale[axis])+offset
                    sphere=FCLObject('probe',Pose(p=point),[CollisionObject(Sphere(.05))],[Pose()])
                    assert bool(collide(converted,sphere).is_collision())==expected_hit
                rows.append(dict(scale=scale,axis=axis,distance=measured,reference=expected,near_collision=True,far_collision=False))
            scene.remove_entity(entity)
    assert not torch.cuda.is_initialized()
    report=dict(status='passed_actual_CPU_PhysX_FCL_scaled_convex_geometry_reference',source_commit=source,
        native_converter_source_sha256=digest,cases=rows,unit_geometry_matches_native=True,
        native_nonunit_rejection_reproduced=True,scaled_geometry_analytic_reference_passed=True,
        collision_checks_retained=True,distance_request_tolerance=1e-10,CPU_physics_only_scene=True,renderers_created=0,task_scenes_created=0,
        models_loaded=0,GPU_seconds=0,triangle_mesh_numerical_qualification=False,GPU_expert_qualification_required=True)
    assert identity in ('cpumotusmeshscale02','cpumotusmeshscale03')
    out=BASE/identity;out.mkdir();p=out/'RESULT.json';p.write_text(json.dumps(report,indent=2)+'\n')
    h=hashlib.sha256(p.read_bytes()).hexdigest();print('MESH_SCALE_CPU_PASS',p,[h[i:i+16] for i in range(0,64,16)],flush=True)
    print('NATIVE_CONVERTER_SHA256',[digest[i:i+16] for i in range(0,64,16)],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--identity',required=True)
    main(parser.parse_args().identity)
