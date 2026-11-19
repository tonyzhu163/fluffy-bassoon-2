"""Collect one independent native Goal 2 episode per driver call."""

from collections import defaultdict
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


OUTPUTS = Path(__file__).resolve().parent.parent
GATE = OUTPUTS / "libero-command-grasp-20261119-r1" / "gate.py"
PILOT = OUTPUTS / "libero-bowl-decisions-20261118-r1" / "pilot.py"
PROBE_CONTROLS = 10


def _load(path):
    spec = spec_from_file_location(path.stem, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bounds(reset, horizon):
    schema = reset["initial"]["action_schema"]
    low, high = (np.asarray(schema[k], dtype=np.float32) for k in ("low", "high"))
    if (reset["horizon"] != horizon or not np.array_equal(low, -np.ones(7))
            or not np.array_equal(high, np.ones(7)) or schema["control_mode"] != "relative"
            or schema["translation_frame"] != "world" or schema["rotation_frame"] != "world"
            or schema["gripper_semantics"] != "negative_open_positive_close_zero_hold"):
        raise ValueError("Unexpected native horizon or action schema")
    return low, high


def _check_actuation(step, bounded, wire, decoder, condition, control):
    native, applied = step["native_input"], step["applied"]
    expected = bounded.copy()
    expected[0] *= decoder * wire
    if not np.array_equal(native, expected):
        raise ValueError(f"Incorrect native wire transform on control {control}")
    if applied.shape != (7,) or not np.array_equal(applied[:6], native[:6]):
        raise ValueError(f"Incorrect applied motion on control {control}")
    override = applied[-1] != native[-1]
    if override and (condition != "D17" or applied[-1] != -1):
        raise ValueError(f"Unexpected applied gripper on control {control}")
    return bool(override)


class Runtime:
    def __enter__(self):
        self.gate = gate = _load(GATE)
        self.pilot = pilot = _load(PILOT)
        devices = pilot.jax.devices()
        if len(devices) != 1 or devices[0].platform != "gpu":
            raise RuntimeError("Expected one scheduler-visible CUDA device for OpenPI")
        if gate.revision(pilot.SRC) != gate.OPENPI_PIN or gate.revision(pilot.PRO) != gate.PRO2_PIN:
            raise ValueError("Frozen OpenPI or Pro2 source differs from the qualified release")
        self.motor = pilot.policy_config.create_trained_policy(
            pilot.config.get_config("pi05_libero"), str(pilot.CKPT))
        self.resets = defaultdict(dict)
        self.openings = {}
        self.simulator = gate.simulator(pilot)  # Starts gate's qualified grasp_simulator.py.
        self.pipe = self.simulator.__enter__()
        return self

    def __exit__(self, *exc):
        return self.simulator.__exit__(*exc)

    def episode(self, folder, pose, wire, condition, decode, grip, decide=None):
        if (wire not in (-1, 1) or condition not in ("BASE", "D17")
                or decode not in (None, -1, 1) or grip not in (0, 1)):
            raise ValueError("Invalid wire, condition, decoder or grasp flag")
        pose, wire, grip = int(pose), int(wire), int(grip)
        decode = None if decode is None else int(decode)
        folder.mkdir(parents=True, exist_ok=False)
        pilot, gate = self.pilot, self.gate
        seed = 19260000 + pose
        reset = pilot.call(self.pipe, "reset", (pose, seed, condition, wire))
        low, high = _bounds(reset, gate.HORIZON)
        first, prompt = reset["initial"], reset["prompt"]
        raw = first["observation"]
        physics = first["environment_reproducibility"]["initial_state_fingerprint"]
        if not physics:
            raise ValueError(f"Missing native reset fingerprint at {folder}")
        initial_public = {k: gate.digest(v) for k, v in raw.items()}
        noise_bank = np.random.default_rng(seed).standard_normal(
            (gate.HORIZON // gate.CADENCE, 10, 32)).astype(np.float32)
        noise_bank.setflags(write=False)
        pose_key = pilot.jax.random.PRNGKey(seed)
        identity = dict(initial_physics=physics, initial_public=initial_public,
                        prompt=prompt, noise_bank=gate.digest(noise_bank))
        checked = dict(identity, initial_public={k: v for k, v in initial_public.items()
                                                if not k.endswith("_image")})
        gate.check_common(self.resets[pose], checked, folder)

        states = [gate.public_pose(raw)]
        commands, rewards = [], []
        chosen_grip = grip
        probe_sign = choice_after = close12 = start = pregrasp = None
        opening = {}
        opening_differences = []
        consecutive = queries = recovery_controls = gripper_overrides = 0
        for t in range(gate.HORIZON):
            if t < PROBE_CONTROLS:
                selected = np.zeros(7, dtype=np.float32)
                selected[0] = 0.45 if t < 5 else -0.45
                selected[-1] = -1
                decoder = 1  # Pulse commands go directly onto the public wire.
            else:
                if t == PROBE_CONTROLS:
                    dx = states[5][0] - states[0][0]
                    probe_sign = int(np.sign(dx))
                    if probe_sign == 0:
                        raise ValueError(f"No first-pulse displacement at {folder}")
                    decoder = probe_sign if decode is None else decode
                if t % gate.CADENCE == 0:
                    if close12 is not None and choice_after is None:
                        choice_after = t
                        if decide is not None:
                            flag = decide(np.stack(commands), np.stack(states))
                            if flag not in (0, 1):
                                raise ValueError(f"Invalid public-prefix decision at {folder}")
                            chosen_grip = int(flag)
                        if chosen_grip:
                            start = t
                            pregrasp = max(0, close12 - gate.CLOSE_CONTROLS - gate.PREGRASP_LOOKBACK)
                            retreat_target = states[t].copy()
                            retreat_target[2] += gate.LIFT_M
                    recovering = start is not None and t < start + 2 * gate.PHASE_CONTROLS
                    if not recovering:
                        slot = t // gate.CADENCE
                        item = pilot.observation(raw, prompt)
                        noise = noise_bank[slot]
                        key = pilot.jax.random.fold_in(pose_key, slot)
                        if t == PROBE_CONTROLS:
                            opening["first_motor_input"] = dict(rgb=gate.digest(gate.images(item)),
                                state=gate.digest(item["observation/state"]), noise=gate.digest(noise),
                                jax_key=gate.digest(np.asarray(key)))
                        self.motor._rng = key
                        proposal = np.asarray(self.motor.infer(item, noise=noise)["actions"])
                        if proposal.shape != (10, 7) or not np.isfinite(proposal).all():
                            raise ValueError(f"Invalid motor proposal at {folder}, control {t}")
                        queries += 1
                        if t == PROBE_CONTROLS:
                            opening["first_proposal"] = gate.digest(proposal)
                            # These inputs follow ten live controls. Record paired differences
                            # so the driver can count the independently measured outcomes.
                            ref = self.openings.setdefault((pose, wire, condition), opening)
                            opening_differences = [k for k in opening if opening[k] != ref[k]]
                if recovering:
                    target = retreat_target if t < start + gate.PHASE_CONTROLS else states[pregrasp]
                    selected = gate.open_toward(states[-1], target)
                    recovery_controls += 1
                else:
                    selected = proposal[t % gate.CADENCE]

            bounded = np.clip(np.asarray(selected, dtype=np.float32), low, high)
            if not np.isfinite(bounded).all():
                raise ValueError(f"Nonfinite command at {folder}, control {t + 1}")
            command = bounded.copy()
            command[0] *= decoder
            step = pilot.call(self.pipe, "step", command)
            gripper_overrides += _check_actuation(step, bounded, wire, decoder, condition, t + 1)
            commands.append(command)
            states.append(step["state"])
            rewards.append(float(step["reward"]))
            if t >= PROBE_CONTROLS and close12 is None:
                consecutive = consecutive + 1 if command[-1] > 0 else 0
                if consecutive == gate.CLOSE_CONTROLS:
                    close12 = t + 1
            raw = step["observation"]
            if step["terminated"] or step["truncated"]:
                break
        else:
            raise RuntimeError(f"Missing native terminal record at {folder}")

        if not opening:
            raise ValueError(f"Episode ended before its first post-probe motor query at {folder}")
        diag = pilot.call(self.pipe, "diagnostics", None)
        if diag["case_id"] != condition:
            raise ValueError(f"Received diagnostics for the wrong native case at {folder}")
        opportunities = [e for e in diag["events"] if e["event"] == "grasp_failure_window_started"]
        fired = [int(e["step"]) + 1 for e in diag["events"] if e["event"] == "recoverable_grasp_failure"]
        n = len(commands)
        if any(not 1 <= c <= n for c in fired):
            raise ValueError(f"Native grasp event outside the executed horizon at {folder}")
        states, rewards = np.stack(states), np.asarray(rewards)
        if states.shape != (n + 1, 9) or not np.isfinite(states).all() or not np.isfinite(rewards).all():
            raise ValueError(f"Invalid public tape at {folder}")
        success = bool(step["success"])
        terminated, truncated = bool(step["terminated"]), bool(step["truncated"])
        if (terminated == truncated or (truncated and n != gate.HORIZON)
                or (success and not terminated) or success != bool(rewards[-1] > 0)
                or (rewards[:-1] > 0).any()):
            raise ValueError(f"Inconsistent native terminal flags or rewards at {folder}")
        result = dict(
            pose=pose, wire=wire, condition=condition, decode=decode, grip=grip, reactive=decide is not None,
            seed=seed, **identity, **opening, first_query_after=PROBE_CONTROLS,
            opening_differences=opening_differences,
            probe_sign=probe_sign, applied_decode=decoder, chosen_grip=chosen_grip, choice_after=choice_after,
            close12_control=close12, regrasp_start_after=start, pregrasp_state_index=pregrasp,
            regrasp_controls=recovery_controls, queries=queries, controls=n,
            success=success, terminated=terminated, truncated=truncated,
            native_return=float(rewards.sum()), d17_opportunity=bool(opportunities) if condition == "D17" else None,
            d17_triggered=bool(fired) if condition == "D17" else None, d17_event_control=fired[0] if fired else None,
            gripper_overrides=gripper_overrides, manifest_sha256=reset["manifest"], bddl_sha256=reset["bddl"],
            renderer=reset["renderer"],
        )
        np.savez_compressed(folder / "tape.npz", commands=np.stack(commands), states=states, rewards=rewards)
        gate.write_json(folder / "result.json", result)
        return result
