# Balanced native interface and fresh-probe qualification

The user accepted the support→fresh-probe plan on 2026-10-28. This bounded
follow-up keeps the existing frozen ManiSkill RL-finetuned pi0.5 motor,
PutOnPlate task, objects 0/1, Gaussian flow residual, width64 CTM T4 / XL T1,
and K4 contract. It introduces no contact task or new steering interface.

One exclusive Baige GPU, 16 CPUs, 64 GiB RAM and 8 GiB shared memory; 90-minute
scheduler cap and 4,800-second driver cap. No retries, deletion or automatic
PPO extension. This is smaller than the existing two-GPU, three-hour allocation.
The launcher runs focused tests and the actual allocation probe before model
work. The first six-lane collection measures memory and time; output is saved
after every complete batch, fit checkpoint, interface condition and probe.

Two six-lane history batches cover each of {-45,0,+45}×{object0,object1}
exactly once in both training and validation: 12 K4 trials, 3,840 simulator
steps and 768 chunk decisions. Complete history seeds/scenes separate splits;
all training/validation overlays remain 0..11. Private theta metadata selects
labels and separate privileged teacher heads, never student public inputs.
The bounded residual saturation and coordinatewise matching-loss floor are
diagnostics; they do not prove four-step action reachability.

Fit three separate same-interface privileged CTM heads and CTM/XL student
candidates for at most 100 cached-flow updates or 600 seconds each. Preserve
full optimizer/sampler/RNG snapshots and portable candidate weights. Compare
each teacher through the actual four-Euler sampler with paired zero bridge
and external inverse-command labels on two fresh held-out-overlay scenes at
the same training theta. The external label reference bypasses the bridge;
the learned head does not. Six episodes per condition are a competence screen.

Evaluate saved DPT-only and checksum-pinned direct-PPO candidates at held-out
±30° on both objects. Three support attempts precede a fresh attempt-four
probe; normal, memory-reset, valid opposite-theta history and zero bridge share
probe scenes/noise and clocks. Copy the complete boundary state including
pending reward and previous public action. Allow recurrence inside each probe,
restore state between probes, save raw history/donor identities and count every
extra simulator episode. Independent support batches, not cloned probes, are
the uncertainty units. One batch here is a diagnostic, not statistical ICRL
evidence or cross-task generalization.

The driver stops between complete stages/attempts under its cap. Missing phases
remain explicitly incomplete. Longer DPT→PPO fitting and paper-level matched
seeds follow only after this interface/data/evaluation gate is audited.

## Launch receipt

Baige `job-37q79sp1o23m`, submitted 2026-10-28 18:56:16 Shanghai, source
`5ac79dce8ec8bbc962a93d623cf0f56b5b74d73a`. GPU admission passed; **78 tests
passed in 18.02 seconds** in the allocation. The six-lane simulator initialized
and its renderer bound to CUDA's allocated PCI `0000:a3:00.0`. The first balanced
history is pending at the 18:58 check; these receipts establish execution
admission, not bridge competence. Exact snapshots and allocation identity are
in `LAUNCH.json`; raw probe/tests are in `evidence/`.

At 19:03 Shanghai the first complete balanced six-lane history was saved:
1,920 simulator steps /384 chunk decisions in219.14seconds, peak CUDA
allocation8.09GB. Exact replay log-probability and value errors were both zero.
The saved batch SHA256 is in LAUNCH.json; its raw receipt is preserved in
evidence/progress-1904.json. The independent validation batch is collecting.
Required CFM residuals exceed the mean bound on9.57%of active-prefix coordinates;
this is an offline pointwise diagnostic and does not establish live sampler
incompetence or success. Actual same-interface teachers and fresh probes remain
pending. Reward-blind behavior is supported/tested by the evaluator but is not
part of this bounded run's four executed probe conditions.
