"""Draw the ICML figures from archived simulator outcomes and reader summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#cccccc",
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
    "figure.dpi": 180,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "pdf.fonttype": 42,
})

PALETTE = {
    "oracle": "#2d3436",
    "ours_v2": "#0066cc",
    "ours_homog": "#2e9cca",
    "brightness": "#ad5c19",
    "dis": "#f39c12",
    "broyden": "#b7950b",
    "perceiver": "#c0392b",
    "perceiver_lg": "#7b241c",
    "no_equiv": "#8e44ad",
    "affine": "#7f8c8d",
    "swapped": "#bdc3c7",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def plot_fig1_multiarm_multivla(ev_dir: Path, out_dir: Path):
    """Compare protocols with the same seed-41 reader on the same scored queries."""
    panels = [
        ("UR5e  " + r"$\pi_{0.5}$", "ur5e-pi05"),
        ("IIWA  " + r"$\pi_{0.5}$", "iiwa-pi05"),
        ("Kinova3  " + r"$\pi_{0.5}$", "kinova3-pi05"),
        ("UR5e  " + r"$\pi_0$", "ur5e-pi0"),
    ]
    fig, ax = plt.subplots(figsize=(4.1, 2.1), layout="constrained")
    for row, (_, panel) in enumerate(panels):
        result = load_json(ev_dir / f"{panel}-summary.json")["by_regime"]["overall"]
        if result["trials"] != 96:
            raise ValueError(f"Expected the full 96-query panel for {panel}")
        successes = result["successes"]
        single = successes["ours_equiv_v2_online"]
        retry = successes["ours_equiv_v2_passive"]
        ax.hlines(row, single, retry, color="#b4c2ca", lw=2)
        for count, label, marker, color, dx in [
            (single, "Single attempt", "o", "#77582e", -6),
            (retry, "Calibrated retry", "s", PALETTE["ours_v2"], 6),
        ]:
            ax.plot(count, row, marker=marker, color=color, ms=6, ls="none",
                    label=label if row == 0 else None)
            ax.annotate(str(count), (count, row), (dx, 0), textcoords="offset points",
                        ha="left" if dx > 0 else "right", va="center", color=color, fontsize=10)
    ax.set_yticks(range(len(panels)), [label for label, _ in panels])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(len(panels) - 0.5, -0.5)
    ax.set_xlim(-2, 96)
    ax.set_xticks([0, 24, 48, 72, 96])
    ax.set_xlabel("Native successes out of 96 queries")
    ax.set_axisbelow(True)
    ax.grid(axis="x")
    ax.legend(loc="lower center", bbox_to_anchor=(0.48, 1), ncol=2, frameon=False,
              handlelength=0.7, columnspacing=1.2, fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig1_multiarm_multivla.{ext}")
    plt.close(fig)


def plot_fig2_scaling_law(ev_dir: Path, out_dir: Path):
    data = load_json(ev_dir / "stage3-scaling-summary.json")
    m_levels = data["m_levels"]
    table = data["table"]

    series = [
        ("equiv_v2_dis_dz", "SymMeta", PALETTE["ours_v2"], "o", "-", 2.6),
        ("equiv_flow_homog", "EquivFlow", PALETTE["ours_homog"], "s", "--", 2.0),
        ("factored_no_equiv_v2", "Factored No-Equiv (Ablation)", PALETTE["no_equiv"], "^", "-.", 1.9),
        ("perceiver_icba_w192", "Local Perceiver (w192)", PALETTE["perceiver"], "D", "-", 2.1),
        ("perceiver_icba_w384", "Local Perceiver (w384, 5.31× parameters)", PALETTE["perceiver_lg"], "v", ":", 2.1),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.3))

    for key, label, color, marker, ls, lw in series:
        row = table[key]
        xs = np.array(m_levels, dtype=float)
        yaw_m = np.array([row[str(M)]["ood_combined"]["yaw_mae_mean"] for M in m_levels])
        yaw_s = np.array([row[str(M)]["ood_combined"]["yaw_mae_std"] for M in m_levels])
        ok_m = np.array([row[str(M)]["ood_combined"]["all_ok_mean"] for M in m_levels])
        ok_s = np.array([row[str(M)]["ood_combined"]["all_ok_std"] for M in m_levels])

        ax1.plot(xs, yaw_m, label=label, color=color, marker=marker, linestyle=ls, linewidth=lw, markersize=6.5)
        ax1.fill_between(xs, np.maximum(yaw_m - yaw_s, 0), yaw_m + yaw_s, color=color, alpha=0.14)

        ax2.plot(xs, ok_m, label=label, color=color, marker=marker, linestyle=ls, linewidth=lw, markersize=6.5)
        ax2.fill_between(xs, np.clip(ok_m - ok_s, 0, 100), np.clip(ok_m + ok_s, 0, 100), color=color, alpha=0.14)

    ax1.axhline(90.0, color="#999999", linestyle=":", linewidth=1.1, label="Uniform yaw (90°)")
    ax1.axvline(7.0, color="#666666", linestyle="--", linewidth=0.9, alpha=0.6)
    ax1.text(7.5, 113.0, "M=7 (scale & grip enter)", fontsize=7.8, color="#444444", va="top")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(m_levels)
    ax1.set_xticklabels([str(m) for m in m_levels])
    ax1.set_xlabel("Number of source training shifts")
    ax1.set_ylabel("Mean absolute yaw error (°)")
    ax1.set_title("(a) Planar yaw estimation")
    ax1.set_ylim(0, 120)
    ax1.grid(True)

    ax2.axvline(7.0, color="#666666", linestyle="--", linewidth=0.9, alpha=0.6)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(m_levels)
    ax2.set_xticklabels([str(m) for m in m_levels])
    ax2.set_xlabel("Number of source training shifts")
    ax2.set_ylabel("All three factors correct (%)")
    ax2.set_title("(b) Recovery of yaw, gain and gripper sign")
    ax2.set_ylim(-2, 106)
    ax2.grid(True)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.10), fontsize=8.8)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig2_scaling_law.{ext}")
    plt.close(fig)


def plot_fig3_horizon_and_so3(ev_dir: Path, out_dir: Path):
    hor = load_json(ev_dir / "ablation-horizon-ur5e.json")
    so3 = load_json(ev_dir / "stage2d-so3-summary.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.7))

    ctrls = [5, 10, 15]
    h_series = [
        ("equiv_v2_dis_dz", "SymMeta", PALETTE["ours_v2"], "o", "-"),
        ("equiv_v2_dis_nodz", "SymMeta (no parallax correction)", "#5dade2", "^", "--"),
        ("equiv_flow_homog", "EquivFlow", PALETTE["ours_homog"], "s", "-."),
        ("analytic_visual_hs", "Brightness-translation", PALETTE["brightness"], "D", "-"),
        ("dense_flow_dis", "DIS + Procrustes", PALETTE["dis"], "P", ":"),
        ("perceiver_icba", "Seven-shift Perceiver", PALETTE["perceiver"], "v", "-"),
    ]
    for key, lbl, col, mk, ls in h_series:
        ys = [hor[str(c)][key]["ood_combined"]["yaw_mae_deg"] for c in ctrls]
        ax1.plot(ctrls, ys, label=lbl, color=col, marker=mk, linestyle=ls, linewidth=2.2, markersize=7)
    ax1.set_xticks(ctrls)
    ax1.set_xticklabels(["5 ctrl\n(0.25 s)", "10 ctrl\n(0.50 s)", "15 ctrl\n(0.75 s)"])
    ax1.set_xlim(4, 16)
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Mean absolute yaw error (°)")
    ax1.set_title("(a) Yaw error by trace length")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.50, -0.25), ncol=2, columnspacing=0.9, fontsize=8.5, frameon=False)
    ax1.grid(True)

    # Right: Full 3D SO(3) + Anisotropic Per-Axis Gains on UR5e (n=96)
    regimes = [("familiar", "Familiar\n(n=32)"), ("compositional_ood", "Comp. 3D\nOOD (n=32)"), ("continuous_so3_ood", "Cont. SO(3)\nOOD (n=32)"), ("overall", "Overall\n(n=96)")]
    so3_methods = [
        ("privileged_oracle", "Exact 3D Oracle", PALETTE["oracle"]),
        ("ours_equiv_so3", "SymMeta (3D)", PALETTE["ours_v2"]),
        ("nominal", "Uncorrected", "#95a5a6"),
        ("reground_babbling_3d", "Local affine (3D)", PALETTE["affine"]),
        ("ours_so3_swapped", "Swapped support", PALETTE["swapped"]),
    ]
    x = np.arange(len(regimes))
    width = 0.16
    for idx, (m_key, m_lbl, col) in enumerate(so3_methods):
        vals = [100.0 * so3["by_regime"][r]["rates"][m_key] for r, _ in regimes]
        offs = x + (idx - 2.0) * width
        ax2.bar(offs, vals, width=width, label=m_lbl, color=col, edgecolor="white", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([lbl for _, lbl in regimes])
    ax2.set_ylim(0, 106)
    ax2.set_ylabel("Native task success (%)")
    ax2.set_title("(b) Second-attempt success under 3D shifts")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=9, frameon=False)
    ax2.grid(axis="y")

    fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.34, wspace=0.24)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig3_horizon_and_so3.{ext}")
    plt.close(fig)


def _three_seed_rate(seed41: dict, ms: dict, method: str, regime: str) -> float:
    """Average reader seeds 41–43, including seed 41 from the original panel."""
    original = seed41["by_regime"][regime]
    repeats = ms["by_regime"][regime]
    if original["trials"] != repeats["trials"]:
        raise ValueError(f"Different trial counts for {regime}")
    successes = original["successes"][method]
    for seed in (42, 43):
        successes += repeats["successes"][f"{method}_s{seed}"]
    return 100 * successes / (3 * original["trials"])


def plot_fig4_memorization_cliff(ev_dir: Path, out_dir: Path, scale_ev_dir: Path):
    def _ms(fn):
        return load_json(scale_ev_dir / fn)
    benches = [
        ("UR5e ($\\pi_{0.5}$)", load_json(ev_dir / "ur5e-pi05-summary.json"), _ms("multiseed-zeroshot-ur5e-pi05.json")),
        ("IIWA ($\\pi_{0.5}$)", load_json(ev_dir / "iiwa-pi05-summary.json"), _ms("multiseed-zeroshot-iiwa-pi05.json")),
        ("Kinova3 ($\\pi_{0.5}$)", load_json(ev_dir / "kinova3-pi05-summary.json"), _ms("multiseed-zeroshot-kinova3-pi05.json")),
        ("UR5e ($\\pi_0$)", load_json(ev_dir / "ur5e-pi0-summary.json"), _ms("multiseed-zeroshot-ur5e-pi0.json")),
    ]
    x = np.arange(len(benches))
    w = 0.18

    fig, ax = plt.subplots(figsize=(9.5, 4.1))
    groups = [
        ("ours_equiv_v2_passive", "familiar", "SymMeta · Familiar", PALETTE["ours_v2"], 1.0, None),
        ("ours_equiv_v2_passive", "ood_combined", "SymMeta · OOD", PALETTE["ours_v2"], 0.55, "///"),
        ("perceiver_icba_passive", "familiar", "Seven-shift Perceiver · Familiar", PALETTE["perceiver"], 1.0, None),
        ("perceiver_icba_passive", "ood_combined", "Seven-shift Perceiver · OOD", PALETTE["perceiver"], 0.55, "///"),
    ]
    for i, (m, reg, lbl, col, alp, hat) in enumerate(groups):
        vals = [_three_seed_rate(d, ms, m, reg) for _, d, ms in benches]
        offs = x + (i - 1.5) * w
        ax.bar(offs, vals, width=w, label=lbl, color=col, alpha=alp, hatch=hat, edgecolor="#222222", linewidth=0.5)
        for xi, v in zip(offs, vals):
            ax.text(xi, v + 1.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.8, fontweight="bold" if reg == "ood_combined" else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels([b for b, _, _ in benches], fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, 106)
    ax.set_ylabel("Native task success (%)")
    ax.set_title("Second-attempt success on familiar and held-out command shifts")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=9)
    ax.grid(axis="y")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig4_memorization_cliff_matrix.{ext}")
    plt.close(fig)


def plot_fig5_pro2_visual_stress(ev_dir: Path, out_dir: Path):
    data = load_json(ev_dir / "pro2-stress-ur5e-summary.json")
    variants = [
        ("S13", "S13   Object appearance"),
        ("S14", "S14   Contrast / scene domain"),
        ("S15", "S15   Camera extrinsics"),
        ("S16", "S16   Camera image formation"),
        ("S19", "S19   Observation quality"),
    ]
    # Historical evidence keys are retained for the archived measurements.
    readers = [
        ("ours_equiv_v2_passive_base", "SymMeta", PALETTE["ours_v2"], "o", -0.19),
        ("analytic_visual_hs_passive", "Brightness-translation", PALETTE["brightness"], "s", 0.0),
        ("privileged_oracle", "Exact inverse", PALETTE["oracle"], "|", 0.24),
    ]
    fig, (success, yaw) = plt.subplots(1, 2, figsize=(8.6, 2.8), sharey=True)
    rows = np.arange(len(variants))
    for key, label, color, marker, offset in readers:
        measures = [data["by_variant"][variant] for variant, _ in variants]
        success.scatter([100 * m["rates"][key] for m in measures], rows + offset,
                        label=label, color=color, marker=marker, s=42)
        if key != "privileged_oracle":
            yaw.scatter([m["mean_yaw_err_deg"][key] for m in measures], rows + offset,
                        color=color, marker=marker, s=42)
    success.set_yticks(rows, [label for _, label in variants])
    success.set_ylim(len(rows) - 0.5, -0.5)
    success.set_xlim(-4, 104)
    success.set_xticks([0, 25, 50, 75, 100])
    success.set_xlabel("Native task success (%)")
    success.set_title("(a) Calibrated retry")
    yaw.set_xlim(0, 16)
    yaw.set_xticks([0, 4, 8, 12, 16])
    yaw.set_xlabel("Mean absolute yaw error (°)")
    yaw.set_title("(b) First-attempt estimate")
    for ax in (success, yaw):
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_axisbelow(True)
        ax.grid(axis="x")
        for boundary in rows[:-1] + 0.5:
            ax.axhline(boundary, color="#dedede", lw=0.6, zorder=0)
    fig.legend(*success.get_legend_handles_labels(), ncol=3, loc="lower center",
               bbox_to_anchor=(0.59, -0.08), frameon=False, handlelength=0.8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig5_pro2_visual_stress.{ext}")
    plt.close(fig)


def plot_fig6_fewshot_sft_vs_sysid(ev_dir: Path, scale_ev_dir: Path, out_dir: Path):
    fs = load_json(scale_ev_dir / "fewshot-finetune-ur5e-summary.json")
    rows_path = ev_dir.parents[1] / "xemb-live-wave-20261121-r1/evidence/pi05-rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    matched = [r for r in rows if r["embodiment"] in fs["shifts"]]
    keys = {(r["suite"], r["task_id"], r["embodiment"]) for r in matched}
    tasks = {(suite, task) for suite, task, _ in keys}
    n = fs["n_queries_per_arm"]
    if len(tasks) != fs["n_tasks"] or len(keys) != len(matched) or len(keys) != n:
        raise ValueError("Incomplete or duplicated task–transform pairs in the demo comparison")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.1), sharey=True, layout="constrained")
    for method, name, color, marker in [
        ("sft_mlp_same_shift", "MLP adapter", "#b14a40", "o"),
        ("sft_linear_same_shift", "Linear adapter", "#ad6900", "s"),
    ]:
        records = fs[method]
        shots = sorted(map(int, records))
        successes = [records[str(k)]["successes"] for k in shots]
        ax1.plot(range(len(shots)), successes, marker=marker, label=name, color=color)
    ax1.set_xticks(range(len(shots)), shots)
    ax1.set_xlabel("Successful oracle demonstrations per shift")
    ax1.set_title("(a) Adapter fitted on the query shift")
    ax1.legend(loc="lower right")

    counts = [sum(r["arms"][method]["success"] for r in matched) for method in
              ("ours_equiv_v2_passive", "privileged_oracle", "analytic_visual_hs_passive")]
    counts.insert(1, fs["sft_linear_same_shift"]["1"]["successes"])
    labels = ["SymMeta", "Linear adapter\n(1 demo)", "Exact inverse", "Brightness-\ntranslation"]
    bars = ax2.bar(labels, counts, color=[PALETTE["ours_v2"], "#ad6900", PALETTE["oracle"], PALETTE["brightness"]])
    ax2.bar_label(bars, padding=2)
    ax2.set_title("(b) Same 32 task–transform pairs")
    ax1.set_ylabel(f"Successful queries (of {n})")
    ax1.set_yticks(range(0, n + 1, 8))
    ax1.set_ylim(0, n + 1)
    for ax in (ax1, ax2):
        ax.set_axisbelow(True)
        ax.grid(axis="y")
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig6_fewshot_sft_vs_sysid.{ext}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--scale-evidence-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_fig1_multiarm_multivla(args.evidence_dir, args.out_dir)
    plot_fig2_scaling_law(args.evidence_dir, args.out_dir)
    plot_fig3_horizon_and_so3(args.evidence_dir, args.out_dir)
    plot_fig4_memorization_cliff(args.evidence_dir, args.out_dir, scale_ev_dir=args.scale_evidence_dir)
    plot_fig5_pro2_visual_stress(args.evidence_dir, args.out_dir)
    plot_fig6_fewshot_sft_vs_sysid(args.evidence_dir, args.scale_evidence_dir, args.out_dir)
    print(f"Wrote six PDF/PNG figure pairs to {args.out_dir}")


if __name__ == "__main__":
    main()
