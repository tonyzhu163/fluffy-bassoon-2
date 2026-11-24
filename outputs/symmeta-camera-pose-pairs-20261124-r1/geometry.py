"""Pinhole calibration in the original pooled agentview raster. No scene labels."""
import json
from pathlib import Path

import numpy as np


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n")
    tmp.replace(path)


def camera(center, gl_to_world, fovy, size=64):
    # Offscreen OpenGL row 0 is bottom, col 0 is left. robosuite opengl leaves it
    # untouched. The archive reverses BOTH axes then averages 4x4 blocks.
    f = 0.5 * size / np.tan(np.deg2rad(fovy) / 2)
    c = (size - 1) / 2
    K = np.array([[-f, 0, c], [0, f, c], [0, 0, 1]])
    gl_to_world = np.asarray(gl_to_world)
    np.testing.assert_allclose(gl_to_world.T @ gl_to_world, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(np.linalg.det(gl_to_world), 1., atol=1e-9)
    world_to_cv = np.eye(4)
    R = np.diag([1., -1., -1.]) @ gl_to_world.T
    world_to_cv[:3] = np.column_stack([R, -R @ center])
    return dict(K=K.tolist(), world_to_cv=world_to_cv.tolist(), P=(K @ world_to_cv[:3]).tolist(),
                center_m=list(center), gl_to_world=gl_to_world.tolist(), fovy_deg=float(fovy))


def as_h(matrix):
    return matrix / matrix[2, 2]


def pair(ref, view, z):
    def basis(cam):
        P = np.asarray(cam["P"])
        return np.column_stack([P[:, 0], P[:, 1], P[:, 3] + P[:, 2] * z])
    H = as_h(basis(view) @ np.linalg.inv(basis(ref)))
    T = np.asarray(view["world_to_cv"]) @ np.linalg.inv(ref["world_to_cv"])
    return dict(H_ref_to_view=H.tolist(), H_view_to_ref=as_h(np.linalg.inv(H)).tolist(),
                T_view_from_ref_cv=T.tolist())


def project(matrix, points):
    points = np.asarray(points)
    hom = np.c_[points, np.ones(len(points))] @ np.asarray(matrix).T
    if np.any(hom[:, 2] <= 0):
        raise ValueError("Point behind camera or homography horizon")
    return hom[:, :2] / hom[:, 2:]


def load_view(row, view):
    with np.load(row["views"][view]["passive_npz"], allow_pickle=False) as tape:
        return {k: tape[k] for k in ("img_before", "img_after", "frame0_img", "valid",
                                     "issued_mean", "delta_eef", "delta_jaw", "chunk_steps", "frame0_state")}
