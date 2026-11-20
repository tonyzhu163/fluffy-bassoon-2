"""Run the nine CPU command-frame prediction fits registered in FIT.md."""

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch import nn


FAMILIES = ("Panda", "Sawyer")
SOURCE = ("nominal", "yaw_p90", "yaw_m90", "gain065")
HELD = ("yaw_p45", "yaw_p90_gain065")
IDEAL_INVERSE_YAW_DEG = (-45, -90)
ROLES = ("support", "query")
BANKS = ("narrow_a", "narrow_b", "broad")
SEEDS = (41, 42, 43)
CONTROLS = ("correct", "wrong", "empty")
STEPS, BATCH = 100, 32
BASE = "0fbc5e3167b362bc33b741e6bbfa20724aa93234"


def array(value, dims):
    a = np.asarray(value)
    if a.shape != (16, dims) or a.dtype.kind not in "ifu":
        raise ValueError(f"expected 16 x {dims} real numbers")
    a = a.astype(np.float32)
    if not np.isfinite(a).all():
        raise ValueError("nonfinite tape value")
    return a


def load_tapes(paths):
    tapes, hashes, metadata = {}, [], {}
    block_ids = ("layout_seed", "probe_seed", "initial_state_sha256", "scene_hash",
                 "movable_xy_hash", "mobile_joint_xy_hash")
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for line_no, line in enumerate(stream, 1):
                digest.update(line)
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    f, s, r, i = key = tuple(row[k] for k in ("family", "setting", "role", "index"))
                    if f not in FAMILIES or s not in SOURCE + HELD or r not in ROLES:
                        raise ValueError(f"unregistered tape {key}")
                    if type(i) is not int or not 0 <= i < (12 if s in SOURCE else 8):
                        raise ValueError(f"invalid layout index {key}")
                    split = "train" if s in SOURCE else "eval"
                    if row.get("split", split) != split:
                        raise ValueError(f"split disagrees with setting {key}")
                    if key in tapes:
                        raise ValueError(f"duplicate tape {key}")
                    tapes[key] = np.column_stack((array(row["commands"], 7), array(row["motion"], 3)))
                    metadata[key] = {k: row[k] for k in block_ids if k in row}
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"{path}:{line_no}: {exc}") from exc
        hashes.append({"path": str(path), "sha256": digest.hexdigest()})

    seen_blocks = {k: set() for k in block_ids if k != "probe_seed"}

    def pack(settings, n):
        # Keep both roles as whole tapes. Axes: family, setting, role, layout, step, channel.
        keys = list(product(FAMILIES, settings, ROLES, range(n)))
        missing = [key for key in keys if key not in tapes]
        if missing:
            raise ValueError(f"missing {len(missing)} tapes, first {missing[0]}")
        for f, s, r, i in keys:
            ref = (FAMILIES[0], settings[0], r, i)
            if not np.array_equal(tapes[f, s, r, i][:, :7], tapes[ref][:, :7]):
                raise ValueError(f"unmatched public commands: {f}/{s}/{r}/{i}")
            if metadata[f, s, r, i] != metadata[ref]:
                raise ValueError(f"unmatched layout/probe metadata: {f}/{s}/{r}/{i}")
        for r, i in product(ROLES, range(n)):
            fields = metadata[FAMILIES[0], settings[0], r, i]
            for name in seen_blocks.keys() & fields.keys():
                if fields[name] in seen_blocks[name]:
                    raise ValueError(f"reused {name} for {settings[0]}/{r}/{i}")
                seen_blocks[name].add(fields[name])
        return np.stack([tapes[key] for key in keys]).reshape(2, len(settings), 2, n, 16, 10)

    return pack(SOURCE, 12), pack(HELD, 8), hashes


class Reader(nn.Module):
    """The stage-one ReaderAdapter with three measured coordinates."""

    def __init__(self, scale):
        super().__init__()
        self.register_buffer("motion_scale", torch.from_numpy(scale.copy()))
        self.reader = nn.Sequential(nn.Linear(31, 48), nn.Tanh(), nn.Linear(48, 16), nn.Tanh())
        self.response = nn.Linear(16, 8 * 3)

    def encode(self, history):
        action = history[..., :7]
        motion = history[..., 7:] / self.motion_scale
        products = (action.unsqueeze(-1) * motion.unsqueeze(-2)).flatten(-2)
        return self.reader(torch.cat((action, motion, products), dim=-1)).mean(dim=1)

    def coefficients(self, context):
        return self.response(context).reshape(-1, 8, 3)

    def forward(self, history, commands):
        coeff = self.coefficients(self.encode(history))
        return torch.bmm(commands[:, None, :], coeff[:, :7]).squeeze(1) + coeff[:, 7]


def draw_schedule(seed):
    rng = np.random.default_rng(seed)
    # Four independent draws from each of eight family x slot x layout-parity strata.
    strata = np.repeat(list(product(range(2), repeat=3)), BATCH // 8, axis=0)
    schedule = []
    for _ in range(STEPS):
        family, slot, parity = strata[rng.permutation(BATCH)].T
        support = 2 * rng.integers(6, size=BATCH) + parity
        query = 2 * rng.integers(6, size=BATCH) + parity
        step = rng.integers(16, size=BATCH)
        schedule.append((family, slot, support, query, step))
    return schedule


def setting_ids(bank, family, slot, layout):
    if bank == "narrow_a":
        return slot
    if bank == "narrow_b":
        return 2 + slot
    # At each layout use one setting from A, one from B. Mirror parity between arms.
    return 2 * slot + ((layout % 2) ^ family)


def train(bank, seed, schedule, tapes, scale):
    torch.manual_seed(seed)
    model = Reader(scale)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    losses = []
    seen_support, seen_query = set(), set()
    for family, slot, support, query, step in schedule:
        setting = setting_ids(bank, family, slot, support)
        seen_support.update(zip(family, setting, support))
        seen_query.update(zip(family, setting, query))
        histories = torch.from_numpy(tapes[family, setting, 0, support])
        labels = torch.from_numpy(tapes[family, setting, 1, query, step])
        opt.zero_grad(set_to_none=True)
        prediction = model(histories, labels[:, :7])
        loss = (prediction - labels[:, 7:] / model.motion_scale).square().mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        opt.step()
        losses.append(loss.item())
    if len(seen_support) != 48 or len(seen_query) != 48:
        raise RuntimeError("sampled tape count differs from registered bank budget")
    if not np.isfinite(losses).all() or not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError(f"nonfinite fit: {bank}/{seed}")
    return model.eval().requires_grad_(False), losses


def design(tape):
    return np.column_stack((tape[:, :7], np.ones(len(tape))))


def inverse_direction(coeff, expected_yaw):
    # Row-vector map in physical units. Hold angular/jaw commands fixed and cancel drift
    # by differencing: d_world = d_command @ J.
    jacobian = coeff[:3]
    rank = int(np.linalg.matrix_rank(jacobian))
    info = {"rank": rank, "ideal_inverse_yaw_deg": expected_yaw}
    if rank == 3:
        correction = np.linalg.solve(jacobian.T, 0.001 * np.eye(3)).T
        x, y = correction[0, :2]
        info.update(condition=float(np.linalg.cond(jacobian)),
                    commands_for_positive_world_mm=correction.tolist())
        if x != 0 or y != 0:
            angle = float(np.degrees(np.arctan2(y, x)))
            info.update(plus_x_azimuth_deg=angle, signed_yaw_error_deg=(angle - expected_yaw + 180) % 360 - 180)
    return info


def scores(coeff, query, scale):
    prediction = design(query) @ coeff
    return float(np.square((prediction - query[:, 7:]) / scale).mean())


def analytic(support, query, scale, expected_yaw):
    x, xq = design(support), design(query)
    y = support[:, 7:].astype(np.float64)
    # One SVD defines the min-norm solution, rank, estimable subspace and leverage.
    u, singular, vh = np.linalg.svd(x, full_matrices=False)
    keep = singular > singular[0] * max(x.shape) * np.finfo(np.float64).eps
    basis = vh[keep]
    pinv = (basis.T / singular[keep]) @ u[:, keep].T
    coeff = pinv @ y
    rank = int(keep.sum())
    residual = y - x @ coeff
    covariance = residual.T @ residual / (len(x) - rank)
    projector = basis.T @ basis
    identified = np.all(np.isclose(xq @ projector, xq, rtol=1e-8, atol=1e-10), axis=1)
    leverage = np.square(xq @ pinv).sum(axis=1)
    stderr = np.sqrt(leverage[:, None] * np.diag(covariance))
    xyz_identified = np.allclose(projector[:3], np.eye(8)[:3], rtol=1e-8, atol=1e-10)
    return {
        "mse": scores(coeff, query, scale), "coefficients_m": coeff.tolist(),
        "prediction_m": (xq @ coeff).tolist(), "design_rank": rank,
        "singular_values": singular.tolist(), "residual_df": len(x) - rank,
        "residual_covariance_m2": covariance.tolist(),
        "mean_prediction_se_m": [v.tolist() if ok else None for v, ok in zip(stderr, identified)],
        "xyz_coefficients_identified": bool(xyz_identified),
        "inverse": inverse_direction(coeff, expected_yaw) if xyz_identified else None,
    }


@torch.inference_mode()
def evaluate(model, tapes, scale):
    panel = np.empty((2, 2, 8, 3))
    directions = []
    for f, s, i in product(range(2), range(2), range(8)):
        query = tapes[f, s, 1, i]
        supports = tapes[f, [s, 1 - s], 0, i]
        correct_wrong = model.encode(torch.from_numpy(supports))
        contexts = torch.cat((correct_wrong, torch.zeros_like(correct_wrong[:1])))
        coeffs = model.coefficients(contexts).double().numpy() * scale
        for c, coeff in enumerate(coeffs):
            panel[f, s, i, c] = scores(coeff, query, scale)
        directions.append({
            "family": FAMILIES[f], "setting": HELD[s], "index": i,
            "inverse": {control: inverse_direction(coeff, IDEAL_INVERSE_YAW_DEG[s])
                        for control, coeff in zip(CONTROLS, coeffs)},
        })
    return panel, directions


def analytic_panel(tapes, scale):
    panel = np.empty((2, 2, 8, 3))
    rows = []
    for f, s, i in product(range(2), range(2), range(8)):
        query = tapes[f, s, 1, i]
        row = {"family": FAMILIES[f], "setting": HELD[s], "index": i}
        for c, donor in enumerate((s, 1 - s)):
            result = analytic(tapes[f, donor, 0, i], query, scale, IDEAL_INVERSE_YAW_DEG[s])
            panel[f, s, i, c] = result["mse"]
            row[CONTROLS[c]] = result
        panel[f, s, i, 2] = scores(np.zeros((8, 3)), query, scale)
        rows.append(row)
    return {"rows": rows, "layout_mse": panel.tolist(), "cell_mse": panel.mean(axis=2).tolist()}


def summarize(values):
    # seed x bank x family x held-out setting x layout x context (all equal cell sizes).
    narrow = values[:, :2, :, :, :, 0].mean(axis=1)
    broad = values[:, 2]
    delta = narrow - broad[..., 0]
    by_seed = delta.mean(axis=(1, 3))
    by_layout = delta.mean(axis=(0, 1))
    narrow_mean = narrow.mean(axis=(0, 1, 3))
    broad_mean = broad[..., 0].mean(axis=(0, 1, 3))
    margin = (narrow_mean > 0) & (broad_mean <= 0.90 * narrow_mean)
    family = broad.mean(axis=(0, 2, 3))
    controls = family[:, 0, None] < family[:, 1:]
    consistent = (by_seed > 0).all(axis=0)
    return {
        "cell_mse": values.mean(axis=4).tolist(),
        "settings": {name: {
            "narrow_mse": narrow_mean[s], "broad_mse": broad_mean[s],
            "relative_reduction": float(1 - broad_mean[s] / narrow_mean[s]) if narrow_mean[s] > 0 else None,
            "paired_gain_by_seed": by_seed[:, s].tolist(),
            "gain_range_over_seeds": [by_seed[:, s].min(), by_seed[:, s].max()],
            "paired_gain_by_layout": by_layout[s].tolist(),
            "layout_gain_se": by_layout[s].std(ddof=1) / np.sqrt(8),
            "meets_margin": bool(margin[s]), "all_seeds_improve": bool(consistent[s]),
        } for s, name in enumerate(HELD)},
        "families": {name: {"broad_mse": dict(zip(CONTROLS, family[f])),
                            "beats_wrong": bool(controls[f, 0]), "beats_empty": bool(controls[f, 1])}
                     for f, name in enumerate(FAMILIES)},
        "advance": bool(margin.all() and consistent.all() and controls.all()),
    }


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tapes", nargs="+", required=True, type=Path, help="JSONL tape file(s)")
    parser.add_argument("--out", required=True, type=Path, help="fresh result directory")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("run the CPU fits on allocated Linux")
    source, held, inputs = load_tapes(args.tapes)
    rms = np.sqrt(np.square(source[:, :, 0, :, :, 7:].astype(np.float64)).mean(axis=(0, 1, 2, 3)))
    scale = np.maximum(rms, 1e-6).astype(np.float32)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True, exist_ok=False)
    models, training = {}, []
    for seed in SEEDS:
        schedule = draw_schedule(seed)
        for bank in BANKS:
            model, losses = train(bank, seed, schedule, source, scale)
            models[seed, bank] = model
            filename = f"{bank}-{seed}.pt"
            torch.save(model.state_dict(), args.out / filename)
            training.append({"bank": bank, "seed": seed, "minibatch_mse": losses, "checkpoint": filename})

    values = np.empty((3, 3, 2, 2, 8, 3))
    for t, b in product(range(len(SEEDS)), range(len(BANKS))):
        seed, bank = SEEDS[t], BANKS[b]
        values[t, b], directions = evaluate(models[seed, bank], held, scale)
        write_json(args.out / f"inverse-{bank}-{seed}.json", directions)
    write_json(args.out / "analytic.json", analytic_panel(held, scale))
    repo_head = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                               capture_output=True, text=True, check=False)
    report = {
        "design_base": BASE, "fit_py_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkout_revision": repo_head.stdout.strip() if repo_head.returncode == 0 else None,
        "inputs": inputs, "argv": sys.argv, "torch": str(torch.__version__), "numpy": np.__version__,
        "motion_rms_scale_m": scale.tolist(), "training": training,
        "source_tapes": int(np.prod(source.shape[:4])), "evaluation_tapes": int(np.prod(held.shape[:4])),
        "tapes_per_fit": 96, "steps": STEPS, "batch_size": BATCH,
        "axis_order": ["seed", "bank", "family", "setting", "index", "control"],
        "seeds": SEEDS, "banks": BANKS, "families": FAMILIES, "settings": HELD,
        "controls": CONTROLS, "layout_mse": values.tolist(), "summary": summarize(values),
    }
    write_json(args.out / "report.json", report)
    print(json.dumps({"report": str(args.out / "report.json"), "advance": report["summary"]["advance"]}))


if __name__ == "__main__":
    main()
