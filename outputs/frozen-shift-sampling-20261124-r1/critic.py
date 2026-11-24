"""Public progress critic fitted on archived source-arm drawer episodes. CPU, archived data only."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parent
POSE = OUTPUTS / "frozen-transfer-pose-20261124-r1/evidence/treatment"
DEMOS = OUTPUTS / "frozen-vla-reachability-20261123-r1/demos/goal3.h5"
ARTIFACT = HERE / "critic.json"
SCALE = .1  # metres per feature unit
RIDGE = 1e-2


def features(displacement, grip):
    """Displacement from the reset pose (..., 3) in metres and the commanded gripper state (...,) in {0, 1}."""
    d = np.asarray(displacement, dtype=np.float64) / SCALE
    g = np.asarray(grip, dtype=np.float64)[..., None]
    x, y, z = d[..., :1], d[..., 1:2], d[..., 2:]
    return np.concatenate((np.ones_like(g), d, g, d * d, x * y, x * z, y * z, g * d), axis=-1)


def value(weights, displacement, grip):
    return features(displacement, grip) @ np.asarray(weights)


def predicted(displacement, proposal, gain):
    """State after the first five commands of a proposal under a linear response."""
    window = np.asarray(proposal, dtype=np.float64)[:5]
    return np.asarray(displacement) + window[:, :3].sum(axis=0) @ np.asarray(gain), float(window[-1, 6] > 0)


def score(weights, displacement, proposal, gain):
    return float(value(weights, *predicted(displacement, proposal, gain)))


def response_gain(inputs, responses, fallback):
    """3x3 displacement per unit summed command from the three calibration bursts."""
    commanded = 5 * np.asarray(inputs, dtype=np.float64)[:, :3]
    moved = np.asarray(responses, dtype=np.float64)
    if commanded.shape != (3, 3) or moved.shape != (3, 3) or np.linalg.cond(commanded) > 1e3:
        return np.asarray(fallback, dtype=np.float64)
    return np.linalg.solve(commanded, moved)


def load_sources():
    """Panda episodes of libero_goal/3. No UR5e, IIWA or Kinova3 trajectory enters."""
    episodes = []
    for path in sorted(POSE.glob("libero_goal-3-*-Panda-*.npz")):
        data, meta = np.load(path), json.loads(path.with_suffix(".json").read_text())
        cmd = data["cmd"].astype(np.float64)
        grip = np.r_[0., (cmd[:-1, 6] > 0).astype(float)]
        episodes.append(dict(
            name=path.stem, kind="pi05", success=bool(meta["success"]), layout=meta["case"]["init_state_id"],
            displacement=data["before_p"] - data["before_p"][0], grip=grip, cmd=cmd,
            moved=data["end_p"] - data["before_p"]))
    if DEMOS.is_file():
        import h5py
        with h5py.File(DEMOS) as handle:
            for key in sorted(handle, key=lambda name: int(name.split("_")[1])):
                position = handle[key]["obs"]["ee_pos"][:]
                jaws = handle[key]["obs"]["gripper_states"][:]
                aperture = np.abs(jaws[:, 0] - jaws[:, 1])
                episodes.append(dict(
                    name=f"demo-{key}", kind="human", success=True, layout=None,
                    displacement=position - position[0], grip=(aperture < .5 * aperture[0]).astype(float)))
    return episodes


def fit(episodes):
    rows = [features(e["displacement"], e["grip"]) for e in episodes]
    labels = [np.arange(len(e["grip"])) / len(e["grip"]) for e in episodes]
    design, target = np.concatenate(rows), np.concatenate(labels)
    penalty = RIDGE * len(design) * np.eye(design.shape[1])
    penalty[0, 0] = 0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


def fit_gain(episodes):
    cmd = np.concatenate([e["cmd"][:, :3] for e in episodes if "cmd" in e])
    moved = np.concatenate([e["moved"] for e in episodes if "cmd" in e])
    return np.linalg.lstsq(cmd, moved, rcond=None)[0]


def ordered_pairs(values, gap=25):
    i, j = np.triu_indices(len(values), k=gap)
    return int((values[j] > values[i]).sum()), len(i)


def window_rank(weights, episode, gain):
    """Executed five-control window against windows from a quarter, half and three quarters of the episode later."""
    length, hits, total, by_phase = len(episode["cmd"]), 0, 0, [[0, 0] for _ in range(4)]
    for start in range(0, length - 5, 5):
        offsets = [start] + [(start + (k * length) // 4) % (length - 5) for k in (1, 2, 3)]
        scores = [score(weights, episode["displacement"][start], episode["cmd"][o:o + 5], gain) for o in offsets]
        hit = int(np.argmax(scores) == 0)
        hits, total = hits + hit, total + 1
        phase = by_phase[min(3, 4 * start // length)]
        phase[0] += hit
        phase[1] += 1
    return hits, total, by_phase


def evaluate(episodes):
    wins = [e for e in episodes if e["success"]]
    fails = [e for e in episodes if not e["success"]]
    report = dict(episodes=[], pairs=dict(pi05=[0, 0], human=[0, 0]), windows=[0, 0],
                  windows_by_quarter=[[0, 0] for _ in range(4)])
    final = {}
    for held in wins:
        train = [e for e in wins if e is not held and (held["layout"] is None or e["layout"] != held["layout"])]
        weights, gain = fit(train), fit_gain(train)
        values = value(weights, held["displacement"], held["grip"])
        hit, total = ordered_pairs(values)
        report["pairs"][held["kind"]][0] += hit
        report["pairs"][held["kind"]][1] += total
        row = dict(name=held["name"], train=len(train), ordered=[hit, total])
        if "cmd" in held:
            wh, wt, phases = window_rank(weights, held, gain)
            report["windows"][0] += wh
            report["windows"][1] += wt
            for mine, theirs in zip(report["windows_by_quarter"], phases):
                mine[0] += theirs[0]
                mine[1] += theirs[1]
            row["windows"] = [wh, wt]
            final[held["name"]] = float(values[-20:].mean())
            # Failures on the same layout are scored by the same held-out-layout fit.
            for other in fails:
                if other["layout"] == held["layout"]:
                    final[other["name"]] = float(value(weights, other["displacement"], other["grip"])[-20:].mean())
        report["episodes"].append(row)
    good = [final[e["name"]] for e in wins if e["name"] in final]
    bad = [final[e["name"]] for e in fails if e["name"] in final]
    report["success_over_failure"] = [int(sum(g > b for g in good for b in bad)), len(good) * len(bad)]
    report["final_values"] = final
    return report


def main():
    episodes = load_sources()
    wins = [e for e in episodes if e["success"]]
    weights, gain = fit(wins), fit_gain(wins)
    artifact = dict(
        weights=weights.tolist(), source_gain=gain.tolist(), scale=SCALE, ridge=RIDGE,
        sources=[dict(name=e["name"], kind=e["kind"], success=e["success"], layout=e["layout"],
                      frames=len(e["grip"])) for e in episodes],
        fitted_on=[e["name"] for e in wins], held_out=evaluate(episodes))
    ARTIFACT.write_text(json.dumps(artifact, indent=1) + "\n")
    held = artifact["held_out"]
    print(json.dumps({k: held[k] for k in ("pairs", "windows", "windows_by_quarter", "success_over_failure", "final_values")}, indent=1))
    print(len(wins), "successes fitted,", len(episodes) - len(wins), "failures held out")


if __name__ == "__main__":
    main()
