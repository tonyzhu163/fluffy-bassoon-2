"""Pixel-center homographies and PR7 visual observations. No motor or label inputs."""
import cv2
import numpy as np
from fluffy_bassoon.xemb.baselines import compute_dis_flow_maps

SIZE = 64
C = (SIZE - 1) / 2
PIXEL = np.array([[C, 0, C], [0, C, C], [0, 0, 1.]], float)
UNIT = np.linalg.inv(PIXEL)
FULL = np.ones((SIZE, SIZE), np.uint8)
TRAIN = [(0, 0)] + [(a*s, 0) for a in (12, 35, 60, 80) for s in (-1, 1)]
TRAIN += [(0, -18), (0, 18), (-35, -18), (-35, 18), (35, -18), (35, 18)]
VALIDATION = [(0, 0), (-60, 0), (60, 0), (-25, 0), (25, 0), (0, -24), (0, 24)]
TARGET = [(0, 0)] + [(a, 0) for a in (-90, -45, -20, 20, 45, 90)] + [(0, -30), (0, 30)]


def plane_projection(azimuth):
    """Pinhole camera aimed at plane origin, unit radius, 45° elevation, 45° vertical FOV."""
    a = np.radians(azimuth)
    center = np.array([np.cos(a), np.sin(a), 1.]) / np.sqrt(2)
    forward = -center
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    axes = np.stack([right, down, forward])
    k = np.array([[32 / np.tan(np.pi/8), 0, C], [0, 32 / np.tan(np.pi/8), C], [0, 0, 1.]])
    return k @ np.column_stack([axes[:, 0], axes[:, 1], -axes @ center])


def homography(roll, orbit):
    roll_h = np.vstack([cv2.getRotationMatrix2D((C, C), roll, 1.), [0, 0, 1.]])
    h = roll_h if orbit == 0 else roll_h @ plane_projection(orbit) @ np.linalg.inv(plane_projection(0))
    return h / h[2, 2]


def encoding(h):
    centered = UNIT @ h @ PIXEL
    return (centered / centered[..., 2:3, 2:3]).reshape(*centered.shape[:-2], 9).astype(np.float32)


def warp(arr, src_to_dest, fill=127):
    return cv2.warpPerspective(arr, src_to_dest, (SIZE, SIZE), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(fill,)*3)


def render(frames, roll, orbit):
    """Only the digital observation renderer can access original reference pixels."""
    h = homography(roll, orbit)
    if orbit == 0 and roll % 90 == 0:
        return np.ascontiguousarray(np.rot90(frames, roll//90, axes=(-3, -2))), FULL.copy(), h
    mask = warp(FULL.astype(np.float32), h, 0) >= 1.
    observed = np.stack([warp(f, h) for f in frames])
    observed[:, ~mask] = 127
    return observed, mask.astype(np.uint8), h


def view_inputs(frames, masks, supplied_h, arm):
    """Four actual observed boundary frames with a static or per-boundary calibration."""
    frames = np.asarray(frames)
    if frames.shape != (4, SIZE, SIZE, 3) or frames.dtype != np.uint8:
        raise ValueError("Expected four chronological 64x64 RGB byte frames")
    matrices = np.broadcast_to(supplied_h, (4, 3, 3))
    masks = np.broadcast_to(masks, (4, SIZE, SIZE))
    if arm == 'A':
        inverse = np.linalg.inv(matrices)
        values = np.stack([warp(frame, h) for frame, h in zip(frames, inverse, strict=True)])
        masks = np.stack([warp(m.astype(np.float32), h, 0) >= 1. for m,h in zip(masks, inverse, strict=True)])
    elif arm == 'B':
        values = frames.copy()
    else:
        raise ValueError(arm)
    values[~masks.astype(bool)] = 127
    # The two frames defining a flow see the same valid pixels. Pad both identically.
    common = np.asarray(masks[:-1] & masks[1:], dtype=np.uint8)
    before, after = values[:-1].copy(), values[1:].copy()
    before[common == 0] = 127
    after[common == 0] = 127
    safe = np.stack([cv2.erode(m, np.ones((3,3), np.uint8), borderType=cv2.BORDER_REPLICATE) for m in common])
    flow = compute_dis_flow_maps(before, after, np.ones(3, bool))
    flow[safe == 0] = 0
    # Static H on the registered panels. Reject unmodeled within-chunk camera motion for B.
    if not np.allclose(encoding(matrices), encoding(matrices[0]), rtol=1e-6, atol=1e-6):
        raise ValueError("Moving camera during support requires an explicit endpoint-pose encoding")
    return dict(img_before=before, img_after=after, dis_flow=flow, pixel_valid=safe,
                calibration=encoding(matrices[0]),
                grid_h=encoding(matrices[0] if arm == "A" else np.linalg.inv(matrices[0])))


def frames_of(batch, i):
    b, a = batch['img_before'][i], batch['img_after'][i]
    if not np.array_equal(a[:-1], b[1:]):
        raise ValueError("Noncontiguous image pairs")
    return np.concatenate([b[:1], a])


def digital(batch, i, pose, arm, reported_pose=None):
    observed, mask, h = render(frames_of(batch, i), *pose)
    return view_inputs(observed, mask, h if reported_pose is None else homography(*reported_pose), arm)
