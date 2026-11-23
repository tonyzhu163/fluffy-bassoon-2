"""CPU validation using software EGL on a separate development task/start."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from arm_env import ArmEnv
from adapter import appearance, source_palette
from panel import case


def summarize_images(first, other):
    stats = {}
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        diff = np.abs(first[key].astype(np.int16) - other[key].astype(np.int16))
        stats[key] = dict(changed_pixels=int(np.any(diff, axis=2).sum()), mean_abs=float(diff.mean()))
    return stats


def main(out):
    out = Path(out)
    c = case("Kinova3", "libero_goal", 4, 16)
    probe = np.array([[.3, -.2, .1, 0, .05, .1, -.8], [-.2, 0, .1, .1, 0, 0, .8]], np.float32)
    with ArmEnv("Panda", c["suite"], c["task_id"]) as source:
        s = source.reset(c["init_state_id"], c["seed"])
        palette = source_palette(source._env)
        palette["source_xml_sha256"] = s["model_identity"]["xml_sha256"]
        palette["source_scene"] = s["environment_reproducibility"]["scene_fingerprint"]
    (out / "palette.json").write_text(json.dumps(palette, indent=2) + "\n")
    observations, traces, stats = {}, {}, {}
    for mode in ("raw", "uniform", "linkwise"):
        with appearance(mode, palette), ArmEnv(c["arm"], c["suite"], c["task_id"]) as target:
            first = target.reset(c["init_state_id"], c["seed"])
            lineage = first["environment_reproducibility"]
            if lineage["scene_fingerprint"] != palette["source_scene"]:
                raise ValueError("Source and target object layout mismatch")
            states = [target._env.get_sim_state().copy()]
            observations[mode] = [first["observation"]]
            rewards = []
            for action in probe:
                step = target.step(action)
                observations[mode].append(step["observation"])
                states.append(target._env.get_sim_state().copy())
                rewards.append([step["reward"], step["terminated"], step["truncated"]])
            traces[mode] = (states, rewards)
            if not np.array_equal(states, traces["raw"][0]) or rewards != traces["raw"][1]:
                raise ValueError("Appearance changed the physics or native reward")
            for raw, changed in zip(observations["raw"], observations[mode]):
                for key in raw.keys():
                    if not key.endswith("_image") and not np.array_equal(raw[key], changed[key]):
                        raise ValueError(f"Nonvisual observation changed {key}")
            stats[mode] = dict(physics_hash=lineage["initial_state_fingerprint"],
                               whole_physics_sequence_sha256=hashlib.sha256(np.asarray(states).tobytes()).hexdigest(),
                               native=rewards, raw_image_deltas=[summarize_images(a,b) for a,b in zip(observations["raw"],observations[mode])],
                               identity=first["model_identity"])
    comparison = [summarize_images(a,b) for a,b in zip(observations["uniform"], observations["linkwise"])]
    if not all(d["agentview_image"]["changed_pixels"] > 0 for d in comparison):
        raise ValueError("No visible intervention between proposed architecture and matched control")
    report = dict(case=c, controls=probe.tolist(), methods=stats, linkwise_vs_uniform=comparison)
    (out / "cpu.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(comparison), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
