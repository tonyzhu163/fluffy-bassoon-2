"""Summarize the complete, predeclared fifteen-fit exploration grid."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev

CONDITIONS = ("demo", "blind", "attempts", "corrections", "linked")
CURVES = ("learner", "bayes_on_learner_history", "oracle_own_attempts")


def summarize(root):
    summary = {}
    for cond in CONDITIONS:
        runs = [json.loads((root/f"{cond}-s{seed}"/f"{cond}.json").read_text()) for seed in range(3)]
        for seed, run in enumerate(runs):
            for key, value in dict(cond=cond, seed=str(seed), K="6", steps="20000", batch="256", eval_n="8192").items():
                assert run["args"][key] == value, (cond, seed, key)
        summary[cond] = {}
        for split in ("train", "heldout"):
            curves = {}
            for key in CURVES:
                values = [r[split][key] for r in runs]
                assert all(len(row) == 7 for row in values)
                curves[key] = dict(mean=[mean(v) for v in zip(*values)],
                                   std=[stdev(v) for v in zip(*values)],
                                   seeds=values)
            curves["bayes_gap_per_seed"] = [
                [b-a for a, b in zip(r[split]["learner"], r[split]["bayes_on_learner_history"])] for r in runs]
            summary[cond][split] = curves
    summary["decision_checks"] = {
        "attempts_within_5pp_bayes_all_heldout_k_each_seed": [max(gaps) <= .05 for gaps in summary["attempts"]["heldout"]["bayes_gap_per_seed"]],
        "attempts_train_gap_over_10pp_at_any_k_each_seed": [max(gaps) > .10 for gaps in summary["attempts"]["train"]["bayes_gap_per_seed"]],
        "blind_minus_demo_heldout_mean": [b-d for b, d in zip(summary["blind"]["heldout"]["learner"]["mean"], summary["demo"]["heldout"]["learner"]["mean"])],
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid", type=Path)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    summary = summarize(args.grid)
    dest = args.grid.parent
    (dest/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    lines = ["# Learning histories: fifteen-fit readout", "",
             "Three seeds per condition, K=6, 20,000 updates, 8,192 evaluation sessions per split and seed.", "",
             "The enumeration reference uses a uniform prior over all 126 rules. Because each evaluation split samples a subset, it is not the split-specific Bayes optimum. Its own attempts are greedy, not information-optimal exploration.", "",
             "| Condition | Held-out k=0 | Held-out k=6 | Reference on own history, k=6 | Reference own attempts, k=6 |", "| --- | ---: | ---: | ---: | ---: |"]
    for cond in CONDITIONS:
        r = summary[cond]["heldout"]
        vals = [r["learner"]["mean"][0]] + [r[key]["mean"][-1] for key in CURVES]
        lines.append("| "+cond+" | "+" | ".join(f"{100*v:.2f}%" for v in vals)+" |")
    lines += ["", "Values are means across seeds. Full curves, seed variation and the predeclared decision checks are in `summary.json`.",
              "This synthetic exploration does not establish adaptation on robot tasks.", ""]
    (dest/"RESULT.md").write_text("\n".join(lines))
    print("\n".join(lines))
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 5, figsize=(16, 6.5), sharex=True, sharey=True)
        styles = (("learner", "Transformer", "#1266b0", "-"),
                  ("bayes_on_learner_history", "Reference, same history", "#d87800", "--"),
                  ("oracle_own_attempts", "Reference, own attempts", "#207b51", ":"))
        for row, split in enumerate(("train", "heldout")):
            for col, cond in enumerate(CONDITIONS):
                ax = axes[row, col]
                for key, label, color, style in styles:
                    r = summary[cond][split][key]
                    ax.plot(range(7), r["mean"], style, color=color, label=label, linewidth=2)
                    if key == "learner":
                        ax.fill_between(range(7), [min(v) for v in zip(*r["seeds"])], [max(v) for v in zip(*r["seeds"])], color=color, alpha=.16)
                ax.axhline(.2, color="#999999", linewidth=.7)
                ax.set(ylim=(.15, 1), xticks=range(0, 7, 2), title=cond if row == 0 else "")
                ax.grid(alpha=.2)
                if col == 0:
                    ax.set_ylabel(f"{split.capitalize()} rules\nQuery accuracy")
                if row == 1:
                    ax.set_xlabel("Past attempts (k)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
        fig.suptitle("Learning from attempts: frozen Transformer vs exact rule enumeration\nMean of three seeds; shading shows the seed range")
        fig.tight_layout(rect=(0, .06, 1, .91))
        fig.savefig(dest/"learning-curves.png", dpi=180)


if __name__ == "__main__":
    main()
