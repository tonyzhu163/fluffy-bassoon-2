"""Map source mesh surface colours to homologous link positions along the target chain."""
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np


def links(env):
    """Root and joint-bearing links in native proximal-to-distal order. Excludes gripper/mount."""
    robot = env.robots[0]
    model = env.sim.model
    chain = [model.body_name2id(robot.robot_model.root_body)]
    chain.extend(int(model.jnt_bodyid[model.joint_name2id(name)]) for name in robot.robot_joints)
    arm_ids = {model.body_name2id(name) for name in robot.robot_model.bodies}
    # Panda's root and first visible fixed link are distinct bodies.
    by_link = [[] for _ in chain]
    for geom in range(model.ngeom):
        b = int(model.geom_bodyid[geom])
        if b not in arm_ids or model.geom_group[geom] != 1 or model.geom_type[geom] != 7:
            continue
        while b not in chain and b in arm_ids:
            b = int(model.body_parentid[b])
        if b not in chain:
            raise ValueError(f"Unassigned arm visual geom {geom}")
        by_link[chain.index(b)].append(geom)
    if len(chain) != 8 or any(not geoms for geoms in by_link):
        raise ValueError(f"Expected eight mesh-bearing chain segments: {by_link}")
    return by_link


def surface_area(model, mesh):
    v0, nv = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    f0, nf = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
    vertices = model.mesh_vert[v0:v0+nv].astype(np.float64)
    triangles = vertices[model.mesh_face[f0:f0+nf]]
    return float(np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1).sum()/2)


def source_palette(env):
    model = env.sim.model
    colors, areas = [], []
    for geoms in links(env):
        shades, weights = [], []
        for geom in geoms:
            material = int(model.geom_matid[geom])
            rgba = model.mat_rgba[material] if material >= 0 else model.geom_rgba[geom]
            if rgba[3] != 1:
                raise ValueError("Expected opaque source arm mesh")
            shades.append(rgba[:3])
            weights.append(surface_area(model, model.geom_dataid[geom]))
        colors.append(np.average(shades, weights=weights, axis=0))
        areas.append(sum(weights))
    return dict(arm="Panda", rgb=np.asarray(colors).tolist(), area_m2=areas,
                uniform=np.average(colors, weights=areas, axis=0).tolist())


@contextmanager
def appearance(mode, palette):
    """Apply simulation-side visual material changes before settling and the first observation."""
    import arm_env
    identity = arm_env._identity

    def apply(env, arm, case_id="BASE"):
        info = identity(env, arm, case_id)
        if mode == "raw":
            return info
        if arm != "Kinova3" or mode not in ("uniform", "linkwise") or palette["arm"] != "Panda":
            raise ValueError(f"Unexpected palette intervention {arm} {mode}")
        model = env.sim.model
        by_link = links(env)
        for index, geoms in enumerate(by_link):
            # Opaque per-geom colours take precedence after detaching any material.
            rgb = palette["uniform"] if mode == "uniform" else palette["rgb"][index]
            model.geom_matid[geoms] = -1
            model.geom_rgba[geoms] = [*rgb, 1.]
        info["visual_intervention"] = dict(method=mode, counts=list(map(len, by_link)))
        return info

    with patch.object(arm_env, "_identity", apply):
        yield
