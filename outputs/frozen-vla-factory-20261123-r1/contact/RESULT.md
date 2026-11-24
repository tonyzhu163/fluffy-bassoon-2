# Neither post-grasp expert rescued a failed continuation

The unchanged frozen π0.5 tape reached the LIBERO goal predicate in 27 of 32 friction/cadence assignments. Choosing the best of three continuations separately for every assignment also reached 27/32. The contact-conditioned gate selected the unchanged tape on all 28 triggered assignments. Four drawer-opening assignments never issued 12 consecutive closing commands. All four succeeded.

These are **paired tape-replay outcomes** on eight development tasks, one Panda reset per task, two pad frictions and two source cadences. The experiment exercised native simulation physics and rewards with archived proposals. It did not run closed-loop VLA inference or evaluate a held-out robot or official task split.

| Decision | Default source tape /16 | Paced source tape /16 | All /32 | Rescues / losses vs continuing |
| --- | ---: | ---: | ---: | ---: |
| Continue unchanged | 15 | 12 | 27 | — |
| Always slow initial carry | 15 | 10 | 25 | 0 / 2 |
| Always retreat and regrasp | 12 | 2 | 14 | 0 / 13 |
| Other-task best fixed | 15 | 12 | 27 | 0 / 0 |
| Other-task command-only gate | 15 | 12 | 27 | 0 / 0 |
| Other-task contact gate | 15 | 12 | 27 | 0 / 0 |
| Per-assignment hindsight oracle | 15 | 12 | 27 | 0 / 0 |

**What the gate could know.** Eight leave-one-task-out folds excluded the held-out task's reset, both cadences and both materials from fitting and scaling. Each fold supplied 28 source cells on seven tasks, with three measured outcomes per cell. The gate used zero target practice episodes. At the 12th consecutive closing command it used issued translations, jaw opening and wrist displacement up to that control. True material, contact diagnostics and outcome labels stayed outside the router. Flipping every held-out outcome left every decision unchanged. All 28 triggers followed a finger contact according to the original archive's integration-step logger. The trigger alone does not certify bilateral grasp. [Design](PLAN.md) · [Choices and deltas](metrics.json)

**Paired uncertainty.** Gate minus continuing, gate minus command-only and oracle minus continuing were 0 percentage points in every task. The corresponding 95% whole-task bootstrap intervals are [0, 0]. Gate minus always slow was +6.25 points [0, 18.75], and gate minus always regrasp was +40.63 points [28.13, 50.00]. The bootstrap resamples eight complete tasks with the evaluated out-of-fold decisions fixed. The zero interval describes exact agreement on this panel. With one reset per task it cannot bound gains at new starts.

**Failure modes.** Five cells defeated all three continuations. Three were the two-can placement (both paced tapes and the reduced-friction default tape), and two were the paced soup placement. Both slow-carry losses occurred on paced milk placement, where the original hit was at control 272 with eight controls left in the horizon. Slowing can add ten controls. Ten of the thirteen regrasp losses lacked enough remaining budget to reach the unchanged success index after the 80-control rewind and maneuver. That accounting does not establish the outcome under a longer horizon. On reduced-friction milk placement the default tape ended at 127, but a single added zero-motion control with the last jaw command reached the predicate at 128. Counting only the original tape would miss it. [Per-cell stops and hits](panel.json)

**Pairing and provenance.** The three branches of each of 32 cells matched on fresh reset hashes, command and public pose prefixes, and gate features. At the original tape cutoff the local continuing branch reproduced all 32 archived binary outcomes and stop controls. None of the 32 full reset fingerprints matched the older Baige machine's hash. Maximum deviations over overlapping replay controls were 1.48 mm in any wrist coordinate and 1.98 mm in any finger coordinate. The causal contrasts use within-run pairs. Candidate actions came from the recorded frozen OpenPI π0.5 LIBERO default and paced policies at `c45aa965db73ad0644382da578ff03a30b2083d7`. Each original command-array hash matched the source manifest. Source friction replay `a6d6cb8d208bf8df744986e03c802a89343fe8d3`. Research checkout `757a44cd4374833cad790852df8b87ddf412e638`. See [source replay](../../libero-xemb-friction-dev-20261119-r1/RESULT.md) and [source manifest](../../libero-xemb-friction-dev-20261119-r1/cpu01/sources.json).

**Decision.** Keep the unchanged continuation for this menu. Even perfect information about friction and the future could rescue zero failures using these two alternatives. Before training a larger contact router, test a continuation with a different physical capability on fresh, matched post-grasp failures, for example object-relative visual feedback during transport. The measurements here leave that capability and transfer to a different arm open.

## Reproduction

Local Linux CPU, four workers, 96 episodes, 24,434 controls, 154.62 s collection, exit 0, zero GPU-hours. No motor parameters were loaded or updated. The verified receipt, software versions and branch order are in [panel.json](panel.json). From the repository root, reproduce the held-out scoring using NumPy alone:

```sh
run=outputs/frozen-vla-factory-20261123-r1/contact
uv run --no-sync python "$run/score.py" --out /tmp/contact-metrics.json
cmp "$run/metrics.json" /tmp/contact-metrics.json
```

To recollect the panel, set `release` to a checkout of `LIBERO-Pro-Team/LIBERO-Pro2.0` at `e5500c690c1498eca1b7050350654964190cdf70` and `sim_python` to its simulator interpreter. This run used the checkout's bundled simulator and BASE assets, the recorded MuJoCo 3.2.3 / robosuite 1.4.0 / NumPy 1.26.4, and no renderer. Python 3.12.16, SciPy 1.17.1, numba 0.61.2 and Torch 2.11.0 were used locally. Torch reads released initialization rows.

```sh
export PYTHONPATH="$release:$release/simulator-source"
export LIBERO_CONFIG_PATH="$(mktemp -d)"
export MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=1 CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
printf 'n\n' | "$sim_python" -c 'from libero.libero import benchmark'
"$sim_python" "$run/collect.py" --release "$release" --workers 4 --out /tmp/contact-panel.json
"$sim_python" "$run/score.py" --panel /tmp/contact-panel.json --out /tmp/contact-metrics.json
```
