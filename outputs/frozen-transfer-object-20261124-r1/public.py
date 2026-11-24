"""Causal features. Neither the image pipeline nor the baselines receive simulator truth."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

CLOCK = ["control", "close_age"]
KIN = CLOCK + ["aperture_mm", "aperture_change_mm", "translation_mm", "vertical_mm", "rotation_rad"]
VISION = ["flow_global", "change_global", "texture_global", "texture_center", "texture_corners"]
METHODS = {
    "clock": CLOCK,
    "proprio": KIN,
    "corners": KIN + VISION + ["flow_corners", "change_corners"],
    "center": KIN + VISION + ["flow_center", "change_center"],
}


def observation(obs):
    rgb = obs["robot0_eye_in_hand_image"]
    assert rgb.dtype == np.uint8 and rgb.shape == (256, 256, 3) and rgb.max() != rgb.min()
    image = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (96, 96), interpolation=cv2.INTER_AREA)
    fingers = obs["robot0_gripper_qpos"]
    return image, obs["robot0_eef_pos"].copy(), obs["robot0_eef_quat_xyzw"].copy(), float(fingers[0] - fingers[1])


def image_features(prev, now):
    flow = cv2.calcOpticalFlowFarneback(prev, now, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    speed = np.hypot(flow[..., 0], flow[..., 1])
    change = cv2.absdiff(prev, now)
    texture = np.abs(cv2.Sobel(now, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(now, cv2.CV_32F, 0, 1))
    stats = {}
    for name, field, fn in (("flow", speed, np.median), ("change", change, np.mean), ("texture", texture, np.mean)):
        def measure(region):
            return float(np.log1p(fn(region)))
        stats[name + "_global"] = measure(field)
        stats[name + "_center"] = measure(field[24:72, 24:72])
        stats[name + "_corners"] = float(np.mean([measure(field[y:y+48, x:x+48])
                                                  for y in (0, 48) for x in (0, 48)]))
    return stats


def features(previous, current, control, close_age):
    img0, xyz0, q0, jaw0 = previous
    img1, xyz1, q1, jaw1 = current
    delta = xyz1 - xyz0
    angle = (Rotation.from_quat(q1) * Rotation.from_quat(q0).inv()).magnitude()
    return dict(control=control, close_age=close_age, aperture_mm=1000 * jaw1,
                aperture_change_mm=1000 * (jaw1-jaw0), translation_mm=1000 * float(np.linalg.norm(delta)),
                vertical_mm=1000 * abs(float(delta[2])), rotation_rad=float(angle), **image_features(img0, img1))
