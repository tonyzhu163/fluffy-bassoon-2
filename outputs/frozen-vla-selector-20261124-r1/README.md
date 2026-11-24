# Choosing a continuation for a frozen VLA

Can a robot tell from its own motion and the state of a task object which intervention would improve its chance of finishing? This pilot implements that choice for frozen OpenPI π₀.₅ on the LIBERO-Pro2.0 BASE drawer-and-bowl task. The end-to-end driver collects its training labels from simulated source-arm forks. A fitted selector checkpoint and native success estimates require the Baige run below.

## What would distinguish this from ICBA?

| Prior result | Implication for this experiment |
| --- | --- |
| Yang, Finn and Sadigh jointly train π₀.₅ and a context encoder on WidowX controller variations. On held-out task–controller pairs, controller metadata scores 12%, SysID 27%, and another task's expert behavior under the deployment controller 50%. [ICBA §§IV–VI](https://arxiv.org/html/2610.39280v2) | Expert history can convey an execution strategy. Our test changes robot arms and gives the target 15 prescribed controls. It asks whether the original motor already has a useful continuation and whether public evidence identifies it. |
| ReGround matches predicted nominal motion to achievable target motion. Embodiment-classifier guidance steers frozen π₀ using successful target rollouts. [ReGround](https://arxiv.org/abs/2611.20244) · [Classifier guidance](https://arxiv.org/abs/2610.22576) | A frozen backbone, dynamics identification, and guided proposals each have prior art. The specific outcome here is the incremental value of each intervention against continuing from the same state. |
| Mirage transfers robot appearance, KITE separates contact intent from kinematic execution, and FIT-MAC routes recovery by failure type. [Mirage](https://robot-mirage.github.io/) · [KITE](https://arxiv.org/abs/2606.22113) · [FIT-MAC](https://arxiv.org/abs/2610.28109) | The proposed empirical contribution is whether source-trained intervention effects transfer across arms and whether separate motion and object evidence improves the decision over a capacity-matched pooled reader. |

Two recorded Kinova3 failures motivate the distinction. On layout 14 the replay opened the drawer by 161 mm and never contacted the bowl. On layout 15 it visited the plate before touching the drawer. On a different Panda panel, hindsight selection among unchanged, slowed and regrasp continuations gave the same 27/32 successes as continuing. On a single Sawyer task, a privileged contact-time expert rescued 18 failures and lost seven successes. These motivate measuring both the available alternatives and the cost of choosing the wrong one. [Kinova replay](../frozen-vla-reachability-20261123-r1/RESULT.md) · [Panda menu](../frozen-vla-factory-20261123-r1/contact/RESULT.md) · [Sawyer forks](../embodiment-sawyer-contact-fork-20261120-r1/RESULT.md)

## What runs

1. On layout 11, issue the existing three five-control bursts and measure each displacement and starting joint posture. Fit a local translation response around the archived source prior. Weight each support by its joint-posture similarity to the query. The four-patch camera reader, previously fitted on 30 human Panda demonstrations, estimates whether the drawer is open from the current and reset images. Reclosure can change its answer. The progress features also carry the last observation and jaw state. The model has no bowl-localization or grasp-success head. [Motion fit](../frozen-vla-motion-models-20261123-r1/motion.py) · [Source camera fit](../frozen-vla-phase-transfer-20261123-r1/RESULT.md) · [Signals](signals.py)
2. Replan with frozen π₀.₅ at its ordinary five-control cadence. **Continue** executes the normal proposal. **Correct** adjusts only its XYZ commands with the calibrated inverse. **Cue** re-queries the same VLA on the same images, state and noise with “open the top drawer” or “put the bowl in the top drawer.” The next replan starts with the released task instruction. Those two cues are authored for this task. [Rollout](rollout.py)
3. Fit two heads to the paired differences in binary terminal success between each intervention and continuing. The factorized selector uses 19 motion features for correction and 13 progress features for cueing. The pooled comparator receives the same inputs and fits the same 34 ridge coefficients to the same outcomes. A unique positive estimated gain triggers the corresponding branch. A tie or two nonpositive gains preserves the original action. Both models, their normalizers, and their fixed feature projections serialize to JSON. [Learner](learner.py)

## Registered comparison

Run one calibration on each of Panda, UR5e and Kinova3. Use Panda and UR5e layouts 0–9 for fitting. Fit both selectors before opening any Kinova3 query. Kinova3 layouts 36–39 are held out from selector fitting. These layouts were previously reserved for the drawer prompt test. The seed is 42710300 plus the released layout number. In order within each arm, cycle the first decision boundary through controls 45, 60, 75 and 90.

Each source case has four runs in one process: continue, a repeat of continue, one corrected replan, and one cued replan. Each target case additionally evaluates one decision from either learned selector and repeated decisions from the boundary onward for each selector and the motion-only branch. All branches share the original policy noise bank. Before accepting a fork, the driver checks reset identity, opening proposal, executed prefix, prefix rewards, current nominal proposal, noise, public features and a digest of native integration and controller state. The state digest is audit-only. Early terminations count in the success denominator and supply no training choice. [Driver](experiment.py) · [Native state audit](../embodiment-sawyer-contact-fork-20261120-r1/fork.py)

The primary comparison is one choice followed by ordinary closed-loop execution. The per-start hindsight maximum measures how many raw failures that menu could rescue in these paired realizations. If it rescues none, the menu supplies no selection headroom on the panel. Compare any rescues to untreated-repeat variation. Where headroom exists, compare factorized and pooled selection using paired native success, successes lost, failures rescued, and a layout-level bootstrap interval. The repeated-intervention branches evaluate a different deployment strategy. Their decisions can change subsequent observations, so the one-choice hindsight maximum does not bound them. Four layouts of one task give a pilot estimate conditional on that task. A cross-task claim requires held-out tasks with their own public progress representation.

## Reproduce

From the repository root on Linux, run the CPU checks:

```sh
uv run --no-sync pytest -q outputs/frozen-vla-selector-20261124-r1
```

The integration fixture runs the real five-control motor **loop** with a deterministic policy, simulator and observation packer. It exercises source collection, fitting, model serialization, all branches, same-noise re-query, treatment of early termination and rejection of a mismatched fork. Two additional checks feed recorded Kinova3 camera frames through the cue path. At control 60 it selects the bowl on layout 14 and the drawer on layout 15. These checks establish signal routing and software behavior. [Tests](test_rollout.py)

An independent frozen SmolVLA CPU forward on one Panda frame changed all five opening actions for both cues. Translation RMS differences from that checkpoint's full task prompt were 0.16968 for opening and 0.14822 for placement. Repeating its full prompt reproduced the entire 50-action chunk. The endpoint is prompt sensitivity on a different motor. [Receipt](prompt-check/receipt.json) · [Reproduction](prompt-check/check.py)

In a Baige Distributed Training allocation with one RTX 4090 or RTX PRO 6000D, a checkout at the recorded revision, and the project's native runtime and model mounts:

```sh
bash outputs/frozen-vla-selector-20261124-r1/run.sh "$(git rev-parse HEAD)" native01
```

The launcher creates a fresh attempt directory on shared storage, probes the allocated device, pins the existing simulator and OpenPI checkouts, and caps the process at two hours. The panel has 116 planned query episodes plus three 15-control calibrations. Each completed case stores native rewards and termination, reset identity, prefix and fork receipts, and decisions. The run stores its revision, models, support traces, per-arm summaries, console log and exit status. No target expert or target reward enters a deployed decision.
