# Learning-history recipe screen and fresh confirmation

Dense multi-query supervision with Fourier features improved held-out correction
reading from **67.40% to 75.64%** on fresh evaluation sessions, a gain of
**8.24 percentage points**. Each of the three fits improved. The full-prior
enumeration reference on the same histories reached **94.94%**, leaving a
19.30-point deficit. The recipe improves evidence use but does not pass the
within-five-points-at-every-prefix screen.

## Fresh frozen-checkpoint comparison

The development panel selected `both` (`--dense --fourier 4`, learning rate
3e-4) by mean held-out k=6 accuracy across three fits. Confirmation used RNG
base 90000 instead of 10000, with 8,192 sessions per split and fit. These are
new scenes and sessions from the same rule split, not new rule families or
new training seeds. All checkpoints were frozen at 20,000 updates; evaluation
performed zero training updates and the launcher verified that every source
checkpoint's SHA256 stayed unchanged.

| Recipe | Train k=6 | Held-out k=0 | Held-out k=6 | Same-history reference k=6 | Held-out fit range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original corrections | 68.09% | 45.30% | 67.40% | 94.94% | 67.15–67.59% |
| Dense + Fourier corrections | 84.72% | 44.74% | 75.64% | 94.94% | 75.01–76.00% |
| Original attempts | 58.08% | 45.31% | 57.34% | 86.70% | 53.94–63.38% |
| Dense + Fourier attempts, lr 1e-3 | 64.50% | 44.56% | 60.50% | 87.39% | 56.37–62.88% |

Corrections improved by 8.46, 7.86 and 8.41 points for seeds 0, 1 and 2.
Their mean held-out curve at k=0…6 is 44.74, 53.44, 59.58, 64.83, 68.98,
72.71 and 75.64%. The gain grows mainly at longer histories. The training-rule
gain is 16.63 points, with a residual 10.41-point reference deficit. Thus both
fitting and generalization deficits remain plausible.

Attempts improved by 3.15 points on average, with seed changes of −0.50,
+8.30 and +1.66 points. Their histories depend on each learner's choices, so
this compares closed-loop behavior as well as reading. It is not a matched
feedback intervention. All four confirmation families fail the near-reference
screen in every fit, on both training and held-out rules.

## What the screen establishes

All 15 fits in the [development panel](readout01/RESULT.md) completed. Mean
held-out k=6 accuracy was 75.76% for `both`, 67.90% for `base-lr`, 74.18% for
`both-lr`, 74.18% for `wide`, and 61.15% for `attempts-both-lr`. The larger model
did not lead this screen; size and learning rate changed together, so this is
not a controlled capacity comparison. No longer-duration fit was tested.

Dense training changes the attention mask and objective together, and Fourier
features change the representation. The result cannot attribute the gain to
one component. The enumeration reader uses the full 126-rule prior rather
than the split-specific sampling prior; it is a reference, not a proof of
optimality. Three-fit ranges are descriptive, not confidence intervals.

The recipe is a useful improvement, but the large remaining gap does not
justify a broad five-condition rerun on the assumption that fitting is solved.
Continue the separately budgeted LIBERO motor/interface feasibility check.
That check tests whether a frozen motor can execute distinct history-selected
choices; it does not inherit a claim of successful robotic adaptation from
this synthetic result. No additional toy training has been launched.

## Validation, allocation and retained artifacts

- Trainer and qualification commit: `4b06413e33d1e1279b2431f370ff9e638bb7cab9`.
  Confirmation launcher: `27fe941012964f2ec922aae17f293ad0898032ed`.
- CUDA qualification passed dense prefix logits/gradients and leakage checks,
  bitwise checkpoint continuation including optimizer/scheduler/RNG, exact
  replay of the original curves, and timing/capacity admission.
- Baige jobs: qualification `job-dxc5nv9qq1yw`, grid `job-h9o49h1vu3zt`
  (15 × RTX 4090), confirmation `job-obkpwzu2057v` (1 × RTX 4090).
  All GPU probes passed; all 15 grid workers and confirmation exited zero.
- Grid scheduler duration: 34m06s, 8.525 GPU-hours. Confirmation duration:
  5m31s, 0.09194 GPU-hours; its process clock was 302 seconds. Qualification
  used 0.06444 scheduler GPU-hours. Cumulative toy work used 16.212 scheduler
  GPU-hours, with a 16.606 submission-to-end conservative bound. LIBERO is
  accounted separately under its 5 GPU-hour cap.
- Resumable checkpoints remain on Baige under the original `grid01` output
  directories; none were overwritten or moved. Small evidence was downloaded
  and SHA256 verified. See [execution](EXECUTION.json),
  [retrieval receipt](evidence-retrieval.json), [fresh results](confirmation-readout01/summary.json),
  [checkpoint manifest](confirm01/results/manifest.json), and
  [qualification](qual01/qualification.json).

Reproduce the JSON-only confirmation analysis with
`python3 outputs/learning-histories-fix-20261111-r1/confirmation_readout.py outputs/learning-histories-fix-20261111-r1/confirm01 --out <fresh-output-directory>`.
