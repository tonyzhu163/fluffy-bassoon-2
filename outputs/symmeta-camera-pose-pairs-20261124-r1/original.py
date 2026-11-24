"""Index the untouched orbit release with measured model intrinsics and both calibration directions."""
import json
from pathlib import Path
import sys

from geometry import camera, dump, pair

HERE=Path(__file__).resolve().parent
index=json.loads((HERE/"original-index.json").read_text())
lookup={"cont_45_nom":(45,1,1),"cont_135_075_inv":(135,.75,-1),
        "cont_m135_135_inv":(-135,1.35,-1),"cont_m45_075":(-45,.75,1)}
rows=[]
for old in index["rows"]:
    # Directly inspected the two compiled reset models. Scalar-FOV perspective camera,
    # sensorsize=[0,0], fovy=45 degrees, opengl convention on each task.
    cams={a:camera(v["center_m"],v["camera_to_world"],45.) for a,v in old["camera_extrinsics"][0].items()}
    assert all(b==old["camera_extrinsics"][0] for b in old["camera_extrinsics"])
    z=old["orbital_pivot_m"][2]
    yaw,xy,grip=lookup[old["embodiment"]]
    views={}
    for name,a in [("nominal","0"),("orbit-","-30"),("orbit+","30")]:
        v=old["views"][a]
        views[name]=dict(passive_npz=v["path"],sha256=v["sha256"],angle_deg=int(a),
                         **cams[a],**pair(cams["0"],cams[a],z))
    rows.append(dict(id=f"UR5e-{old['index']:04d}",arm="UR5e",role="heldout",
                     **{k:old[k] for k in ("index","shard","suite","task_id","init_state_id","seed","archive_npz","archive_sha256","physics","frame_times_s")},
                     native_control_id=old["embodiment"],control_yaw_deg=yaw,xy_scale=xy,gripper_sign=grip,
                     frame_steps=[0,5,10,15],replay="archived_chunk_mean_held_for_five",original_orbit=True,plane_z_m=z,views=views))
dump(sys.argv[1],dict(rows=rows))
