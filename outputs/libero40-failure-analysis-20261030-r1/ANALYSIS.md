# Saved-data diagnosis, 2026-10-30

The inspected failures contain substantial executed steering noise and real
robot motion. The strongest immediate limitations are **no successful target
training experience on tasks0/40**, very short exposure to the learned SAC
policy, and a mismatch between the student sampling dispersion and its label
distribution. None of these establishes a definitive causal explanation or a
negative result for ICRL in general.

Scope: terminal six-case PixelSAC run
`libero40-target-online-dsrl-six-20261030-r1`, native query before training
versus final query after16 training episodes on tasks0/40; the same two cases
from the terminal all40 AD/DPT firstwave; saved final500 tensors and source
label arrays. Original runs/checkpoints were read without modification. No
rollouts, model construction, inference, training, tests or allocations ran.
Checkpoint inspection used CPU-only tensor loading; image exports preserve
unchanged saved pixels.

## Actual training exposure and reward

|Task |Training successes /16 |Training chunks |Final SAC updates |Frozen query outcomes at0/1/4/8/13/16 episodes |
| --- | ---: | ---: | ---: | --- |
|0, close top drawer |0 |640 |3,200 |fail at every budget |
|10 |12 |352 |0 |success at every budget; queries remain native |
|20 |14 |244 |0 |success at every budget; queries remain native |
|30 |10 |517 |800 |success at every budget; only final query uses SAC |
|40, put frying pan on shelf |0 |640 |3,200 |fail at every budget |
|50 |15 |258 |0 |success at every budget; queries remain native |

On each failed case all640 training chunk rewards were **−1**, and all6,400
primitive rewards were **0**. All16 episodes timed out at400 primitives.
The first13 complete episodes used unbounded Gaussian warmup; episode13's
postepisode burst made the first800 optimizer updates. Only episodes14–16
physically used the learned SAC policy, followed by the final3,200-update
query. Thus the flat4/6 curve contains three cases with no SAC updates and
only two failed cases with repeated postwarmup exposure. It is a bounded
learning-onset observation, not converged DSRL performance.

Weights/updates did advance. On tasks0/40 the final actor-mean norms in saved
update metrics were0.0401/0.0400; actor distribution std averaged0.8762/0.8764
before its bounded sampling transform. Final executed query noise std was
0.5928/0.5901, with mean−0.0157/+0.0049. Final critic losses near7.2e−5/6.5e−5
show fitting of the recorded objective; they do not imply task improvement.
The constant failure reward supplies no observed successful trajectory or
reward contrast for selecting the correct object/sequence. That is a plausible
learning limitation, rather than proof that sparse-reward steering cannot work.

## Visible failure behavior

These descriptions follow direct inspection of main and wrist keyframes.
Object identities are visual descriptions; no contact forces, causal language
grounding explanation or unseen object-state trajectory is inferred.

- **Task0:** both native and final SAC sequences lift the small bowl from the
  table and move it into the open top drawer. The arm then stays near that
  bowl/drawer region; the top drawer remains visibly open at400 actions.
  Sampled end-effector span over the last100 primitives is only approximately
  2.9×3.4×1.6mm for native and2.2×0.7×0.5mm for SAC. This is an observed
  wrong-sequence/stalled finish, rather than successful closing.
- **Task40:** the frying pan remains on the table. The arm interacts around
  the stove and small white dish. Native eventually carries that dish toward
  the cabinet; final SAC moves/drops the dish beside the stove. Neither saved
  sequence transfers the frying pan onto the cabinet shelf.

[Task0 native: bowl lifted at100](task00-native-q0-step100-image.png),
[native final open drawer](task00-native-q0-step400-image.png),
[SAC final open drawer](task00-sac-q16-step400-image.png).
[Task40 native final: white dish carried](task40-native-q0-step400-image.png),
[SAC final: pan on table, dish near stove](task40-sac-q16-step400-image.png).
Wrist frames and intermediate frames are preserved alongside these images.
All four query traces start from matching per-case observation/nominal-state
hashes and use the same per-case query seed before/after SAC training.

The steering distribution and decoded actions changed, but the high-level
wrong-object/sequence behavior persisted on these two cases. Whether this is
mainly an insufficiently trained critic/actor, restricted steering reach,
representation loss, or frozen-motor goal behavior requires an intervention;
these images alone do not identify that cause. PixelSAC also uses image64 plus
proprio8, whereas the history controllers consume frozen4104D prefix features.
That representation difference remains a comparison limitation.

## Student dispersion: no deterministic-MSE explanation

Actual code trains **Gaussian sequence NLL**, with a learned global
per-channel `log_std`. Deployment samples `mean + std × Gaussian` with
`deterministic=False`, then clips coordinates to[−1,1]. The dispersion is not
state conditional. Every inspected candidate starts at std0.5 and has500 Adam
updates at learning rate3e−4.

|Final500 model |Checkpoint std min / mean / max |Executed sample std, task0 / task40 |
| --- | --- | --- |
|AD full |0.58417 /0.58481 /0.58567 |0.5564 /0.5568 |
|AD static |0.58434 /0.58498 /0.58583 |0.5515 /0.5585 |
|DPT full |0.42044 /0.42076 /0.42092 |0.4255 /0.4209 |
|DPT static |0.42074 /0.42105 /0.42121 |0.4280 /0.4253 |

These are stochastic, nonzero outputs. Saved samples do not support literal
zero-output or deterministic-action collapse. They do not by themselves
identify the conditional mean: its contribution cannot be separated exactly
from sampled noise using these summary statistics alone.

The exact admitted all40 training bank has31,351 chunks:

|Training labels / actual behavior |Chunks |Empirical coordinate std |
| --- | ---: | ---: |
|AD warmup labels = executed Gaussian |20,436 (65.2%) |1.00075 |
|AD postupdate labels = executed SAC |10,915 (34.8%) |0.58821 |
|AD labels overall |31,351 |0.87936 |
|DPT final-teacher labels overall |31,351 |0.14396 |
|DPT labels at warmup-history observations |20,436 |0.14247 |
|DPT labels at postupdate-history observations |10,915 |0.14672 |

AD's learned global std≈0.585 is below the overall label dispersion, while
DPT's≈0.421 exceeds its label dispersion. The two objectives push the same
initial0.5 dispersion in opposite directions, but500 updates leave a large
contrast with their respective empirical target scales. This is evidence for
investigating optimization/calibration before treating the trained controller
as a faithful teacher imitation. Marginal label dispersion is not the same as
conditional residual variance, so this table alone cannot prove the optimal
sampling std or prescribe its replacement.

The fully verified contemporaneous noise screen scores native29/90, zero25,
unbounded tied28 and clipped tied29. Native versus zero has four wins and no
losses. Thus temporal tying or clipping alone does not explain a large broad
collapse in this screen. The earlier native31/90 remains a separate firstwave
reference. A more specific student-scale/conditional-mean effect remains open.

## Next discriminating check, before scaling

1. Review the existing25/100/500 NLL and saved global-std trajectories against
   these actual label distributions. This is a fitting/calibration check, not
   a new robot claim or a reason to relabel final source actors as experts.
2. If a small robot diagnostic is authorized later, pair the existing frozen
   student with a zero-conditional-mean control retaining **its exact learned
   std**, the same initial state/query RNG and one common motor process. That
   separates conditional steering from its sampling scale. Predeclare cases
   and cost; it is a diagnostic, not outcome-selected efficacy evaluation.
3. A target-online improvement claim needs informative successful/recovery
   experience or a longer prespecified onset curve beyond these three actual
   postwarmup training episodes. The present all-negative failed-task traces
   do not justify scaling the same collection on confidence alone.

No next experiment was launched or reserved by this analysis.

Evidence: `stats.json` contains original trace/group checksums, per-stage action
and noise statistics, per-episode update metrics and keyframe hashes;
`dispersion.json` binds all four exact final export SHAs and the admitted
assembly/label manifests. `read_saved.py` and `read_dispersion.py` reproduce
saved-data inspection only. Label statistics read NPZ noise/teacher-step arrays
without loading feature tensors or rerunning the model; dataset identity is
linked to the admitted assembly manifest, without another full-corpus audit.
