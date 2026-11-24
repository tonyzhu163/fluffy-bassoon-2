"""Main native OOD retry comparison. The caller supplies the paper's rcParams and palette."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


PANELS = {
    "gpu09test": r"UR5e  $\pi_{0.5}$",
    "gpu09iiwa": r"IIWA  $\pi_{0.5}$",
    "gpu06kinova3": r"Kinova3  $\pi_{0.5}$",
    "gpu08pi0fix": r"UR5e  $\pi_0$",
}


def _verified_ood(native_dir: Path, readback_dir: Path):
    summary_bytes = (native_dir / "summary.json").read_bytes()
    summary = json.loads(summary_bytes)
    receipt = json.loads((readback_dir / "readback.json").read_bytes())
    if receipt["execution"]["attempt"] != "native02g4090":
        raise ValueError("Expected the independent native02g4090 readback")
    if hashlib.sha256(summary_bytes).hexdigest() != receipt["verified_summary_sha256"]:
        raise ValueError("The official summary differs from the independently verified bytes")
    csv_bytes = (readback_dir / "pairs.csv.gz").read_bytes()
    if hashlib.sha256(csv_bytes).hexdigest() != receipt["pairs_csv_gz_sha256"]:
        raise ValueError("The paired CSV differs from the independent readback")

    for source in (summary, receipt):
        if (source["complete"] is not True or source["completed_pairs"] != 1152
                or source["expected_pairs"] != 1152 or source["panels"].keys() != PANELS.keys()):
            raise ValueError("Expected 1,152 matched pairs across all four panels")
    ood = {}
    for panel in PANELS:
        result = summary["panels"][panel]
        if result["missing_index_seed"]:
            raise ValueError(f"{panel} has missing query–seed pairs")
        for split, per_seed in (("overall", 96), ("ood", 64)):
            group = result[split]
            sizes = {s: v["n"] for s, v in group["per_seed"].items()}
            if (group["complete"] is not True or group["n"] != 3 * per_seed
                    or sizes != dict.fromkeys(("41", "42", "43"), per_seed)):
                raise ValueError(f"Incomplete {panel}/{split} denominators")
        group = result["ood"]
        verified = receipt["panels"][panel]["ood"]
        if any(verified[k] != group[k] for k in group):
            raise ValueError(f"{panel} OOD statistics disagree with the readback")
        counts = group["successes"]
        rates = {method: 100 * counts[method] / group["n"] for method in ("tuned", "symmeta")}
        delta = 100 * (counts["tuned"] - counts["symmeta"]) / group["n"]
        lo, hi = group["task_cluster_ci95_pp"]
        if (rates != group["success_pct"] or delta != group["tuned_minus_symmeta_pp"]
                or not -100 <= lo <= hi <= 100):
            raise ValueError(f"Inconsistent {panel} rates, paired difference or CI")
        ood[panel] = (rates["tuned"], rates["symmeta"], delta, lo, hi)
    return ood


def _signed(x):
    return f"{x:+.1f}".replace("-", "−") if x else "0.0"


def fig4_native_retry(native_dir: Path, readback_dir: Path, palette: dict):
    """Return a Matplotlib figure from the strict reducer and independent readback."""
    rows = _verified_ood(native_dir, readback_dir)
    fig = plt.figure(figsize=(6.615, 2.02))
    # Absolute rates have separate columns so ties and near-ties stay legible.
    table = fig.add_axes([0, 0.235, 1, 0.535])
    forest = fig.add_axes([0.52, 0.235, 0.19, 0.535], sharey=table, facecolor="none")
    table.set(xlim=(0, 1), ylim=(3.5, -0.5))
    table.set_axis_off()
    fig.text(0.012, 0.92, "Native OOD retry success", weight="bold", size=10.5)
    fig.text(0.50, 0.92, "Tuned Perceiver − fresh SymMeta (pp)", weight="bold", size=10)
    tuned_color, symmeta_color = palette["perceiver"], palette["ours_v2"]
    for x, label, align, color in [
        (0.012, "Arm / policy", "left", "black"),
        (0.242, "Tuned Perceiver", "center", tuned_color),
        (0.408, "Fresh SymMeta", "center", symmeta_color),
        (0.615, "Paired difference", "center", "black"),
        (0.865, "Estimate [95% CI]", "center", "black"),
    ]:
        fig.text(x, 0.80, label, ha=align, color=color, fontsize=8.8)

    forest.axvline(0, color="#999999", lw=0.9, ls="--", zorder=0)
    for y, (panel, (tuned, symmeta, delta, lo, hi)) in enumerate(rows.items()):
        table.hlines(y - 0.5, 0.012, 0.988, color="#dedede", lw=0.65)
        table.text(0.012, y, PANELS[panel], va="center")
        table.text(0.242, y, f"{tuned:.1f}%", ha="center", va="center", color=tuned_color)
        table.text(0.408, y, f"{symmeta:.1f}%", ha="center", va="center", color=symmeta_color)
        ink = palette["oracle"]
        forest.hlines(y, lo, hi, color=ink, lw=1.5)
        forest.plot([lo, hi], [y, y], "|", color=ink, ms=5)
        forest.plot(delta, y, "o", color=ink, ms=4.6)
        table.text(0.865, y, f"{_signed(delta)}  [{_signed(lo)}, {_signed(hi)}]",
                   va="center", ha="center", fontsize=9)

    extent = max(1, *(abs(v) for row in rows.values() for v in row[2:]))
    forest.set_xlim(-1.08 * extent, 1.08 * extent)
    forest.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    forest.tick_params(axis="x", labelsize=8, length=2.5, pad=2)
    forest.yaxis.set_visible(False)
    forest.spines["left"].set_visible(False)
    fig.text(0.012, 0.035, "64 OOD queries × 3 reader seeds (41–43) per panel. CIs resample eight whole tasks.",
             fontsize=8.5, color=palette["oracle"])
    return fig
