"""Shared real-simulator loop. Only public state and actually executed actions enter PublicHistory."""
from types import SimpleNamespace
import numpy as np
from panel import MANIFEST, RELEASE, TASKS, require
from source_data import PublicHistory
from ur5e_motor import array_digest, run_episode
from fluffy_bassoon.xemb.embodiment import get_embodiment_spec


class Wire:
    # Native pi05_libero action horizon and diffusion width, also enforced at infer.
    _model = SimpleNamespace(action_horizon=10, action_dim=32)

    def __init__(self, sim, motor, mode):
        self.sim = sim
        self.motor = motor
        self.mode = mode
        self.history = PublicHistory()
        self.calls = []
        self.controls = 0
        self.parity_result = None
        self.mechanism = None

    def call(self, op, value):
        if op == "create":
            spec, first = self.sim.call(op, value)
            self.first = first
            self.raw = first["observation"]
            self.history.reset()
            self.lag = self.history.vector().copy()
            require(np.array_equal(self.lag, np.zeros(52, np.float32)), "history leaked across resets")
            return spec, first
        require(op == "chunk", op)
        step = self.sim.call(op, value)
        n = step["steps"]
        require(0 < n <= len(value) and n == len(step["positions"]) == len(step["gripper_positions"]), "chunk length")
        sent = value[:n]
        # The subprocess reports the public XYZ and finger qpos after EACH applied control.
        for issued, xyz, fingers in zip(sent, step["positions"], step["gripper_positions"], strict=True):
            following = {"robot0_eef_pos": xyz, "robot0_gripper_qpos": fingers}
            self.history.append(issued, self.raw, following)
            self.raw = following
        self.raw = step["observation"]
        self.controls += n
        self.calls[-1].update(issued=sent.tolist(), positions=np.asarray(step["positions"]).tolist(),
                              fingers=np.asarray(step["gripper_positions"]).tolist())
        return step

    def infer(self, item, *, noise):
        current = self.history.vector().copy()
        require(current.dtype == np.float32 and current.shape == (52,) and np.isfinite(current).all(), "history schema")
        require(noise.dtype == np.float32 and noise.shape == (10, 32), "diffusion noise schema")
        require(item["prompt"] == self.first["language_instruction"], "instruction changed")
        require(item["observation/state"].shape == (8,) and all(item[k].dtype == np.uint8 and
                item[k].shape == (224, 224, 3) for k in ("observation/image", "observation/wrist_image")), "OpenPI input")
        used = current
        if self.mode == "joint_zero":
            used = np.zeros_like(current)
        elif self.mode == "joint_lagged":
            used = self.lag
        condition = "joint" if self.mode.startswith("joint") else self.mode
        if self.mode == "original" and self.controls == 0:
            self.parity_result = self.motor.parity(item, noise=noise)
            p = self.parity_result
            require(p["stock_sha256"] == p["noop_sha256"] and p["max_abs"] == 0 and p["shape"] == [10, 7],
                    "stock and zero-update injected sampler diverged")
            require(p.get("history_graph_noop_max_abs", p["max_abs"]) == 0,
                    f"explicit no-op history injection mismatch: {p}")
        infer = self.motor.infer
        if self.controls and condition in ("adapter_only", "joint"):
            infer = getattr(self.motor, "history_infer", infer)
        result = infer(item, noise=noise, condition=condition, history=used)
        proposal = np.asarray(result["actions"], dtype=np.float32)
        if self.mode == "joint" and self.controls == 10:
            # Same joint trajectory, same observation/noise. Counterfactual proposals are never issued.
            effects = {}
            for label, h in (("zero", np.zeros_like(current)), ("lagged", self.lag)):
                alt = np.asarray(infer(item, noise=noise, condition="joint", history=h)["actions"])
                diff = np.abs(proposal - alt)
                effects[label] = dict(history=array_digest(h), proposal=array_digest(alt),
                                      first_five_max=float(diff[:5].max()), chunk_max=float(diff.max()))
            self.mechanism = dict(at_control=self.controls, current=array_digest(current),
                                  proposal=array_digest(proposal), alternatives=effects)
        self.calls.append(dict(t=self.controls, obs={k: array_digest(v) for k, v in item.items() if k != "prompt"},
                               noise=array_digest(noise), history=current.tolist(), used=array_digest(used),
                               proposal=array_digest(proposal), proposed_prefix=proposal[:5].tolist()))
        self.lag = current
        return result


class UnmatchedReset(ValueError):
    pass


def episode(sim, motor, case, mode, reference, fixture):
    _, horizon, prompt, rows = TASKS[case["suite"], case["task_id"]]
    for attempt in range(1, 4):
        wire = Wire(sim, motor, mode)

        def check(reset):
            full = wire.first["environment_reproducibility"]
            body = wire.first["model_identity"]
            require(all(full[k] == case[k] for k in ("suite", "task_id", "init_state_id", "seed"))
                    and full["released_row_fingerprint"] == rows[case["init_state_id"] - 46]
                    and full["source_pin"] == RELEASE and full["manifest_sha256"] == MANIFEST
                    and full["perturbation_plan"]["case_id"] == "BASE", "released start")
            require(reset["prompt"] == prompt and reset["horizon"] == horizon and reset["worker_step_index"] == 0
                    and body["model_class"] == "Mounted" + case["arm"] and body["arm"] == case["arm"]
                    and body["joint_count"] == 7 and body["control_freq"] == 20 and body["controller"] == "OSC_POSE"
                    and body["gripper"] == "PandaGripper", "native interface")
            require("llvmpipe" in reset["renderer"].lower() if fixture else reset["renderer_vendor"] == "NVIDIA Corporation",
                    "renderer mismatch")
            if reference is not None and (reference["native"]["reset"] != reset or reference["environment"] != full):
                raise UnmatchedReset("unmatched opening before first action")

        try:
            native = run_episode(wire, wire, case, get_embodiment_spec("nom"), check_reset=check,
                                 max_controls=15 if fixture else None, canonicalize_quat=False)
        except UnmatchedReset:
            if attempt == 3:
                raise
            continue
        opening = wire.first["observation"]
        return dict(case=case, mode=mode, fixture=fixture, accepted_reset_attempt=attempt, native=native,
                    initial_xyz=opening["robot0_eef_pos"].tolist(), initial_fingers=opening["robot0_gripper_qpos"].tolist(),
                    environment=wire.first["environment_reproducibility"], body=wire.first["model_identity"],
                    parity=wire.parity_result, mechanism=wire.mechanism, calls=wire.calls)
