"""Closed-jaw, inside-out plate push for the simulated Goal 5 Sawyer source."""

import numpy as np


class RimExpert:
    """Create once per reset. Pass the current Session.expert_scene() each tick."""

    def __init__(self, scene):
        # Same destination as the direct controller, for a paired approach test.
        goal = np.asarray(scene["goal"][:2]) + 0.55 * np.asarray(scene["goal_half"][:2])
        self.points = [goal]
        self.phase = "lift"
        self.seat_z = None

    def action(self, scene):
        plate = np.asarray(scene["plate"][:2])
        tcp = np.asarray(scene["tcp"][:2])
        low = float(scene["finger_low"])
        table = float(scene["table_z"])
        rim = float(scene["rim_z"])
        radius = float(scene["plate_radius"])
        hover = max(table + 0.20, rim + 0.05)
        centered = np.linalg.norm(tcp - plate) <= 0.012
        if self.phase == "lift" and low >= hover - 0.012:
            self.phase = "align"
        if self.phase == "align" and centered:
            self.phase = "lower"
        if self.phase == "lower" and centered:
            # A solid plate may stop the fingers at its surface. An open
            # collision model may allow the tips below the lip without contact.
            if (scene["contact"] and low <= rim + 0.003) or low <= table + 0.003:
                self.seat_z = max(table + 0.001, low - 0.002)
                self.phase = "sweep"

        xy = tcp if self.phase == "lift" else plate
        z = hover
        if self.phase == "lower":
            z = table + 0.001
        if self.phase == "sweep":
            # Approach the goal-side wall from inside. As the plate moves,
            # its measured center advances this target and corrects side slip.
            delta = self.points[0] - plate
            direction = delta / max(np.linalg.norm(delta), 1e-8)
            xy = plate + direction * (radius + 0.012)
            z = self.seat_z

        delta_xy = xy - tcp
        max_step = 0.014 if self.phase == "sweep" else 0.04
        delta_xy *= min(1.0, max_step / max(np.linalg.norm(delta_xy), 1e-8))
        # Correct tool tilt in world coordinates. Yaw is free. Use measured
        # lowest finger height because TCP-to-table distance varies with tilt.
        axis = np.asarray(scene["tool_z"])
        tilt = np.cross(axis, [0.0, 0.0, -1.0])
        if np.linalg.norm(tilt) < 1e-8 and axis[2] > 0:
            tilt = np.array([1.0, 0.0, 0.0])
        command = np.r_[1.5 * delta_xy / 0.05, 1.5 * (z - low) / 0.05,
                        np.clip(tilt / 0.5, -0.5, 0.5), 1.0]
        return np.clip(command, -1.0, 1.0).astype(np.float32)
