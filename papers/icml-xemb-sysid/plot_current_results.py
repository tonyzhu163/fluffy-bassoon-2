"""Generate clean, readable, plain-English ICML figures from verified LIBERO-Pro2.0 & ManiSkill3 evidence."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11.5,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "#cccccc",
        "figure.dpi": 300,
    })


def plot_fig1_memorization_cliff(scale_live: dict, out_path: Path):
    """Fig 1: Dev split (1,056 live eps on UR5e) — clean, wide spacing, plain English."""
    regimes = ["familiar", "compositional_ood", "continuous_angle_ood"]
    regime_labels = [
        "Seen Shifts\n(n = 32)",
        "New Combinations\n(n = 32)",
        "New Angles (±45°, ±135°)\n(n = 32)",
    ]
    # Keep to the 6 core methods for maximum visual clarity
    arms = [
        ("privileged_oracle", "Oracle (Upper Bound)", "#4c566a", "//"),
        ("analytic_visual_sysid_passive", "Optical Flow (Hand-Crafted)", "#81a1c1", ""),
        ("ours_retained_active", "Ours (Active)", "#1b7837", ""),
        ("ours_retained_passive", "Ours (Passive, No Probe)", "#5aae61", ""),
        ("joint_transformer_active", "Transformer Baseline (Prior)", "#d95f02", ""),
        ("nominal", "No Adaptation", "#999999", ""),
    ]

    x = np.arange(len(regimes))
    width = 0.132
    offsets = (np.arange(len(arms)) - (len(arms) - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(12.6, 5.8))
    for i, (key, label, color, hatch) in enumerate(arms):
        vals = [100.0 * scale_live["by_regime"][r]["rates"][key] for r in regimes]
        bars = ax.bar(
            x + offsets[i],
            vals,
            width=width * 0.88,
            label=label,
            color=color,
            edgecolor="#222222",
            linewidth=0.7,
            hatch=hatch,
            zorder=3,
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                v + 1.6,
                f"{v:.0f}%",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold" if key in ("ours_retained_active", "joint_transformer_active") else "normal",
                color="#111111",
            )

    jt_idx = [a[0] for a in arms].index("joint_transformer_active")
    x_comp = x[1] + offsets[jt_idx]
    ax.annotate(
        "Transformer collapses on new shifts:\n75% Seen → 0% New Combos / 12% New Angles",
        xy=(x_comp, 6.0),
        xytext=(x[1] + 0.05, 83.0),
        ha="center",
        va="center",
        fontsize=9.8,
        fontweight="bold",
        color="#a63603",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff5eb", edgecolor="#d95f02", lw=1.2),
        arrowprops=dict(arrowstyle="->", color="#d95f02", lw=1.6, connectionstyle="arc3,rad=-0.08"),
        zorder=5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(regime_labels, fontsize=11.5, fontweight="bold")
    ax.set_ylabel("Task Success Rate (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title(
        "LIBERO-Pro2.0 Dev Split (New 6-DoF UR5e Arm, Frozen π0.5, 1,056 Episodes)",
        fontsize=13.5,
        fontweight="bold",
        pad=44,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3, fontsize=10.0, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_fig2_overall_and_cis(scale_live: dict, out_path: Path):
    """Fig 2: Overall success + 95% CIs — generous margins, plain English, zero clipping."""
    order = [
        ("privileged_oracle", "Oracle (Upper Bound)", "#4c566a"),
        ("analytic_visual_sysid_passive", "Optical Flow (Passive)", "#81a1c1"),
        ("ours_retained_active", "Ours (Active)", "#1b7837"),
        ("analytic_visual_sysid_active", "Optical Flow (Active)", "#a6bddb"),
        ("ours_retained_passive", "Ours (Passive, No Probe)", "#5aae61"),
        ("joint_transformer_active", "Transformer (Prior)", "#d95f02"),
        ("factored_no_equiv_active", "No Rotation Geometry", "#e7298a"),
        ("analytic_proprio_sysid", "Joint Sensors Only", "#b2abd2"),
        ("nominal", "No Adaptation", "#999999"),
        ("ours_cleared", "Empty Context (Control)", "#bdbdbd"),
        ("ours_swapped_donor", "Wrong Context (Control)", "#d9d9d9"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 6.2), gridspec_kw={"width_ratios": [1.05, 1.05]})

    y = np.arange(len(order))[::-1]
    rates = [100.0 * scale_live["by_regime"]["overall"]["rates"][k] for k, _, _ in order]
    succs = [scale_live["by_regime"]["overall"]["successes"][k] for k, _, _ in order]
    labels = [lbl for _, lbl, _ in order]
    colors = [c for _, _, c in order]

    bars = ax1.barh(y, rates, color=colors, edgecolor="#222222", linewidth=0.7, height=0.68, zorder=3)
    for b, r, s, (k, _, _) in zip(bars, rates, succs, order):
        ax1.text(
            r + 1.4,
            b.get_y() + b.get_height() / 2.0,
            f"{r:.1f}% ({s}/96)",
            va="center",
            ha="left",
            fontsize=9.8,
            fontweight="bold" if k.startswith("ours_retained") else "normal",
        )
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=10.5)
    ax1.set_xlabel("Overall Success Rate (%, n = 96)", fontsize=11.5)
    ax1.set_xlim(0, 88)
    ax1.set_title("A. Overall Success Rate (All 11 Arms)", fontsize=12.5, fontweight="bold", pad=10)

    contrasts = scale_live["paired_contrasts_overall"]["ours_retained_active"]
    c_order = [
        ("ours_swapped_donor", "vs. Wrong Context"),
        ("nominal", "vs. No Adaptation"),
        ("ours_cleared", "vs. Empty Context"),
        ("analytic_proprio_sysid", "vs. Joint Sensors Only"),
        ("factored_no_equiv_active", "vs. No Rotation Geometry"),
        ("joint_transformer_active", "vs. Transformer (Prior)"),
        ("analytic_visual_sysid_active", "vs. Optical Flow (Active)"),
        ("analytic_visual_sysid_passive", "vs. Optical Flow (Passive)"),
        ("privileged_oracle", "vs. Oracle"),
    ]
    cy = np.arange(len(c_order))[::-1]
    ax2.axvline(0.0, color="#333333", linestyle="--", linewidth=1.2, zorder=2)
    for idx, (key, lbl) in enumerate(c_order):
        c = contrasts[key]
        mean = 100.0 * c["mean_diff"]
        lo = 100.0 * c["ci95"][0]
        hi = 100.0 * c["ci95"][1]
        w, l = c["wins"], c["losses"]
        col = "#1b7837" if lo > 0 else ("#555555" if hi >= 0 else "#b2182b")
        ax2.errorbar(
            mean,
            cy[idx],
            xerr=[[mean - lo], [hi - mean]],
            fmt="o",
            color=col,
            ecolor=col,
            elinewidth=2.2,
            capsize=4.5,
            markersize=7.0,
            zorder=4,
        )
        ax2.text(
            hi + 2.5,
            cy[idx],
            f"{mean:+.1f} pp ({w}W/{l}L)",
            va="center",
            ha="left",
            fontsize=9.5,
            color=col,
            fontweight="bold" if lo > 0 else "normal",
        )
    ax2.set_yticks(cy)
    ax2.set_yticklabels([lbl for _, lbl in c_order], fontsize=10.5)
    ax2.set_xlabel("Difference: Ours (Active) − Baseline (pp, 95% CI)", fontsize=11.5)
    ax2.set_xlim(-35, 118)
    ax2.set_title("B. 95% Bootstrap CIs (Ours Active)", fontsize=12.5, fontweight="bold", pad=10)

    fig.suptitle(
        "LIBERO-Pro2.0 Dev Split: Overall Performance & Controls (6-DoF UR5e, 1,056 Episodes)",
        fontsize=13.8,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_fig3_scaling_offline(out_path: Path):
    """Fig 3: More Data Ablation (224 vs 1,344 trials) — legend outside above plot, zero overlap."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.8))

    groups = ["Seen\nShifts", "New\nCombos", "New\nAngles"]
    x = np.arange(len(groups))
    w = 0.19

    jt_8 = [6.3, 106.0, 82.9]
    jt_24 = [2.14, 107.47, 83.60]
    eq_8 = [11.1, 22.2, 22.0]
    eq_24 = [10.96, 16.25, 19.08]

    series = [
        (jt_8, "Transformer (224 trials)", "#fdae6b"),
        (jt_24, "Transformer (1,344 trials)", "#d95f02"),
        (eq_8, "Ours (224 trials)", "#a1d99b"),
        (eq_24, "Ours (1,344 trials)", "#1b7837"),
    ]
    offsets = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]
    for idx, (off, (vals, lbl, col)) in enumerate(zip(offsets, series)):
        bars = ax1.bar(x + off, vals, width=w * 0.88, label=lbl, color=col, edgecolor="#222222", lw=0.7, zorder=3)
        for b, v in zip(bars, vals):
            # Stagger slightly when adjacent orange bars are very close (106 vs 108, 83 vs 84)
            y_off = 2.0 if idx % 2 == 0 else 6.8 if (v > 75) else 2.0
            ax1.text(
                b.get_x() + b.get_width() / 2.0,
                v + y_off,
                f"{v:.0f}°",
                ha="center",
                va="bottom",
                fontsize=9.2,
                fontweight="bold" if idx in (1, 3) else "normal",
            )

    ax1.axhline(90.0, color="#666666", linestyle=":", linewidth=1.3, label="Random Guessing (90°)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, fontsize=11.0, fontweight="bold")
    ax1.set_ylabel("Yaw Angle Error (° ↓ lower is better)", fontsize=11.5)
    ax1.set_ylim(0, 128)
    ax1.set_title("A. Active Probe: 6× Data Does Not Fix\nTransformer Failure on New Shifts", fontsize=12.0, fontweight="bold", pad=46)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.19), ncol=2, fontsize=9.0, frameon=True)

    # Panel B: Passive (no probe) overall yaw error
    p_labels = [
        "Optical\nFlow",
        "Ours\n(1,344 RGB)",
        "Ours\n(1,344 Gray)",
        "Ours\n(224 RGB)",
        "No Rotation\nGeometry",
        "Transformer\n(Prior)",
        "Joint\nSensors",
    ]
    p_vals = [11.9, 20.35, 28.77, 52.9, 45.58, 67.92, 89.3]
    p_cols = ["#81a1c1", "#005a32", "#1b7837", "#a1d99b", "#e7298a", "#d95f02", "#b2abd2"]
    px = np.arange(len(p_labels))
    pbars = ax2.bar(px, p_vals, color=p_cols, edgecolor="#222222", lw=0.7, width=0.62, zorder=3)
    for b, v in zip(pbars, p_vals):
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            v + 2.0,
            f"{v:.0f}°",
            ha="center",
            va="bottom",
            fontsize=9.8,
            fontweight="bold",
        )
    ax2.axhline(90.0, color="#666666", linestyle=":", linewidth=1.3)
    ax2.annotate(
        "6× more tasks cuts\npassive error by -33°",
        xy=(1.0, 29.0),
        xytext=(1.8, 76.0),
        ha="center",
        va="center",
        fontsize=9.8,
        fontweight="bold",
        color="#1b7837",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#e5f5e0", edgecolor="#1b7837", lw=1.2),
        arrowprops=dict(arrowstyle="->", color="#1b7837", lw=1.5),
        zorder=5,
    )
    ax2.set_xticks(px)
    ax2.set_xticklabels(p_labels, fontsize=9.2, fontweight="bold")
    ax2.set_ylabel("Passive Yaw Error (° ↓ lower is better)", fontsize=11.5)
    ax2.set_ylim(0, 108)
    ax2.set_title("B. Passive Mode (No Probe): More Tasks\nSharply Lower Our Angle Error", fontsize=12.0, fontweight="bold", pad=18)

    fig.suptitle(
        "Effect of 6× More Training Data on New 6-DoF UR5e Arm (224 vs. 1,344 Source Trials)",
        fontsize=13.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_fig4_official_and_ms3(out_path: Path):
    """Fig 4: 3-Seed x 576-Episode Official LIBERO-Pro2.0 Test Split & ManiSkill3 (SAPIEN) — clean, plain English."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.8, 6.0), gridspec_kw={"width_ratios": [1.42, 1.0]})

    # Panel A: 3-Seed x 576-Episode Official LIBERO-Pro2.0 Test Split (n=192 per regime)
    regimes = ["Seen Shifts\n(n = 192)", "New Combos\n(n = 192)", "New Angles\n(n = 192)"]
    series = [
        ("Oracle (Upper Bound)", [81.2, 81.2, 89.1], "#4c566a", "//"),
        ("Dense LK Optical Flow", [65.6, 39.1, 57.8], "#81a1c1", None),
        ("Ours (Active, 3-seed)", [58.3, 52.1, 42.2], "#1b7837", None),
        ("Ours (Passive, 3-seed)", [52.1, 32.3, 47.9], "#5aae61", None),
        ("Yang Transf. + Cont. Aug", [78.6, 5.7, 21.9], "#e66101", None),
        ("Transformer (Unaug.)", [80.7, 0.0, 8.3], "#fdb863", None),
        ("RMA-GRU (Unaug.)", [62.0, 0.0, 22.9], "#b2abd2", None),
    ]
    x = np.arange(len(regimes))
    n_s = len(series)
    w = 0.118
    for idx, (lbl, vals, col, hatch) in enumerate(series):
        off = (idx - (n_s - 1) / 2.0) * w
        bars = ax1.bar(
            x + off,
            vals,
            width=w * 0.88,
            label=lbl,
            color=col,
            edgecolor="#222222",
            lw=0.65,
            hatch=hatch,
            zorder=3,
        )
        for b, v in zip(bars, vals):
            ax1.text(
                b.get_x() + b.get_width() / 2.0,
                v + 1.5,
                f"{v:.0f}%",
                ha="center",
                va="bottom",
                fontsize=7.8,
                fontweight="bold" if idx in (2, 4) else "normal",
            )

    ax1.annotate(
        "Even with continuous aug:\nYang Transf. drops to 6% New Combos\n(Ours +33 pp on unseen shifts)",
        xy=(1.0 + (4 - 3.0) * w, 12.0),
        xytext=(1.08, 102.0),
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color="#a63603",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff5eb", edgecolor="#e66101", lw=1.2),
        arrowprops=dict(arrowstyle="->", color="#e66101", lw=1.5),
        zorder=6,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(regimes, fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Task Success Rate (%)", fontsize=11.5)
    ax1.set_ylim(0, 118)
    ax1.set_title("A. Official LIBERO-Pro2.0 Test Split (3 Seeds × 576 Eps/Method)", fontsize=11.8, fontweight="bold", pad=54)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, fontsize=8.3, frameon=True)

    # Panel B: ManiSkill3 (SAPIEN) 7-DoF Panda+Fetch -> 6-DoF xArm6-Robotiq
    ms_labels = ["Ours", "Optical\nFlow", "No Rotation\nGeometry", "Transformer\n(Prior)", "Joint\nSensors"]
    ms_vals = [32.95, 71.25, 80.50, 89.44, 90.01]
    ms_cols = ["#1b7837", "#81a1c1", "#e7298a", "#d95f02", "#b2abd2"]
    mx = np.arange(len(ms_labels))
    mbars = ax2.bar(mx, ms_vals, color=ms_cols, edgecolor="#222222", lw=0.7, width=0.58, zorder=3)
    for b, v in zip(mbars, ms_vals):
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            v + 2.2,
            f"{v:.1f}°",
            ha="center",
            va="bottom",
            fontsize=9.8,
            fontweight="bold",
        )
    ax2.axhline(90.0, color="#666666", linestyle=":", linewidth=1.2)
    # Place callout directly above bar 0 (Ours) so arrow goes straight down without crossing any bar/label!
    ax2.annotate(
        "Beats optical flow by -38°\n#1 live success (23.6% vs 1.4%)",
        xy=(0.0, 42.5),
        xytext=(0.95, 105.0),
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        color="#1b7837",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#e5f5e0", edgecolor="#1b7837", lw=1.2),
        arrowprops=dict(arrowstyle="->", color="#1b7837", lw=1.5),
        zorder=6,
    )
    ax2.set_xticks(mx)
    ax2.set_xticklabels(ms_labels, fontsize=9.8, fontweight="bold")
    ax2.set_ylabel("Yaw Angle Error (° ↓ lower is better)", fontsize=11.5)
    ax2.set_ylim(0, 120)
    ax2.set_title("B. ManiSkill3 (SAPIEN): New 6-DoF xArm6", fontsize=12.0, fontweight="bold", pad=16)

    fig.suptitle(
        "Main Cross-Benchmark Results: Official LIBERO-Pro2.0 Test Split & ManiSkill3",
        fontsize=13.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale-live", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    setup_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scale_live = json.loads(args.scale_live.read_text())

    plot_fig1_memorization_cliff(scale_live, args.out_dir / "fig1_memorization_cliff_ood.png")
    plot_fig2_overall_and_cis(scale_live, args.out_dir / "fig2_overall_and_causal_controls.png")
    plot_fig3_scaling_offline(args.out_dir / "fig3_scaling_memorization_vs_sysid.png")
    plot_fig4_official_and_ms3(args.out_dir / "fig4_official_test_and_maniskill3.png")
    print("WROTE 4 FIGURES TO:", args.out_dir)


if __name__ == "__main__":
    main()
