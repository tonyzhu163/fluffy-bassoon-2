"""Interventions on the isolated, released PandaGripper; no VLA or arm scene."""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from robosuite.models.grippers.panda_gripper import PandaGripper

from screen import HERE, arm_probes, source


class Bench:
    def __init__(self, mounted):
        self.grip = PandaGripper()
        xml = Path(self.grip.file)
        self.sha = hashlib.sha256(xml.read_bytes()).hexdigest()
        assert self.sha == "d9d68123909d6910488310d0fae4e358ed4ac831c6d0957f3e30893a5412baac"
        if mounted:
            # A hand fixed directly to world has world-parent contact filtering
            # in MuJoCo. Keep it a separate body welded to a stationary mount,
            # so its own fingers are again dynamic children of the hand.
            root = ET.parse(xml).getroot()
            for mesh in root.findall(".//mesh"):
                mesh.set("file", str(xml.parent / mesh.get("file")))
            ET.SubElement(root.find(".//body[@name='right_gripper']"), "freejoint")
            ET.SubElement(ET.SubElement(root, "equality"), "weld", body1="right_gripper")
            self.m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        else:
            self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(self.m)
        self.lo, self.hi = self.m.actuator_ctrlrange.T
        assert (self.m.nu, self.m.opt.timestep) == (2, .002)
        self.qadr = [self.m.joint(name).qposadr[0] for name in ("finger_joint1", "finger_joint2")]
        self.vadr = [self.m.joint(name).dofadr[0] for name in ("finger_joint1", "finger_joint2")]
        self.jaw_range_mm = 1000 * (self.hi[0] - self.lo[1])

    def reset(self, width_mm, previous_dw=0., drive=None):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.qadr] = np.array([1., -1.]) * width_mm / 2000
        self.d.qvel[self.vadr] = np.array([1., -1.]) * previous_dw / (2000 * .05)
        if drive is None:
            # Map the public initial width to a symmetric target. No archived
            # simulator target, force, velocity or contact state is read.
            left = np.clip(2 * width_mm / self.jaw_range_mm - 1, -1, 1)
            drive = np.array([left, -left])
        self.grip.current_action = drive.copy()
        mujoco.mj_forward(self.m, self.d)

    def step(self, u):
        for _ in range(25):
            # The upstream controller refreshes grip_action every 2 ms.
            action = self.grip.format_action(np.array([u]))
            self.d.ctrl[:] = (self.hi + self.lo + (self.hi - self.lo) * action) / 2
            mujoco.mj_step(self.m, self.d)
        q1, q2 = self.d.qpos[self.qadr]
        return 1000 * (q1 - q2)

    def sweep(self):
        # Close / open / close. Repeat the identical initial conditions and
        # sign history, changing only absolute command.
        seq = np.repeat([1., -1., 1.], 5)
        traces = {}
        for amp in (0., .0001, .01, .1, .25, .5, 1.):
            self.reset(self.jaw_range_mm / 2)
            traces[amp] = np.array([self.step(amp * s) for s in seq])
        return {"amplitudes": list(traces), "positive_negative_controls_per_amplitude": [10, 5],
                "max_nonzero_vs_unit_width_difference_mm": max(
                    float(np.max(abs(w - traces[1.]))) for amp, w in traces.items() if amp),
                "zero_vs_unit_max_difference_mm": float(max(abs(traces[0.] - traces[1.]))),
                "unit_width_mm_at_5_10_15": traces[1.][4::5].tolist()}

    def state_intervention(self):
        # Fork the *same* midrange qpos and zero qvel. Alter only the stored
        # command state by one preceding open or close command, then issue zero.
        out = {}
        for direction in (-1., 1.):
            self.reset(self.jaw_range_mm / 2)
            self.step(direction)
            drive = self.grip.current_action.copy()
            self.reset(self.jaw_range_mm / 2, drive=drive)
            out[str(direction)] = self.step(0.) - self.jaw_range_mm / 2
        return out

    def transfer(self, e, horizons=(1, 5, 15)):
        # Targets depend only on the initial width and issued history. Reseed
        # position and velocity at each block boundary, leaving the drive alone.
        errors = {}
        for h in horizons:
            self.reset(e.w[0])
            err = []
            for t, u in enumerate(e.u):
                if t % h == 0 and t:
                    self.reset(e.w[t], e.w[t] - e.w[t-1], self.grip.current_action)
                w = self.step(u)
                if (t + 1) % h == 0:
                    err.append(abs(w - e.w[t + 1]))
            errors[f"h{h}"] = float(np.mean(err))
        return errors


def replay_check(bench):
    episodes = source()
    by_task = {}
    for e in episodes:
        by_task.setdefault(e.task, []).append(bench.transfer(e, (1, 5)))
    return {task: {k: float(np.mean([r[k] for r in rows])) for k in ("h1", "h5")}
            for task, rows in by_task.items()}


if __name__ == "__main__":
    assert version("robosuite") == "1.4.0" and version("mujoco") == "3.2.3"
    bench, bare = Bench(mounted=True), Bench(mounted=False)
    contacts = {}
    for name, fixture in [("world_parent", bare), ("welded_mount", bench)]:
        fixture.reset(40.)
        fixture.step(1.)
        contacts[name] = [[fixture.m.geom(a).name, fixture.m.geom(b).name]
                          for a, b in zip(fixture.d.contact.geom1, fixture.d.contact.geom2)]
    report = dict(robosuite=version("robosuite"), mujoco=version("mujoco"),
                  xml_sha256=bench.sha, amplitude=bench.sweep(),
                  fixed_state_zero_command_delta_mm=bench.state_intervention(),
                  midrange_fixture_contacts=contacts,
                  mounted_component_on_policy_tapes_task_mae_mm=replay_check(bench),
                  isolated_physics_cross_arm_mae_mm={e.name: bench.transfer(e) for e in arm_probes()},
                  world_parent_diagnostic_mae_mm={e.name: bare.transfer(e) for e in arm_probes()})
    (HERE / "component.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
