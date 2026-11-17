# Does useful history survive the full adapter in closed loop?

Proposed November 17, 2026. No allocation or launch is attached to this plan.
It follows the [native calibration pilot](../robocasa-feedback-advantage-20261117-r1/RESULT.md).
In that pilot, a small DPT reader scored 95.83% on held-out second attempts.
Clearing its history gave 50%, and flipping its feedback gave 4.17%.

The pilot established that feedback carries usable information in one
controlled case. It left four questions open:

1. **No closed loop.** Readers were scored against branch outcomes that had
   already been measured; no trained reader drove the robot.
2. **Small model and narrow choice.** The reader was a two-layer selector
   over compact state tokens, choosing among three corrections.
3. **Privileged timing.** Simulator handle geometry decided when to correct.
4. **Hardware confound.** Training scenes ran on RTX 4090 and held-out scenes
   on Pro6K.

This experiment asks the next question: does the image-conditioned history
bridge on the frozen π0.5 motor make the same use of feedback, executed closed
loop, without a privileged trigger?

## Fixed elements

- **Task and motor.** Native RoboCasa `OpenDrawer`, the public Human300 π0.5
  checkpoint and the pilot's calibration mechanism, unchanged: a signed 4 cm
  lateral shift of the controller goal during approach (`collect_panel.py`,
  `--bias-m 0.04`).
- **Trial structure.** The hidden sign is drawn independently of scene and
  stays fixed across one support episode and one query episode. Simulator
  reset between them; learner memory reset between trials.
- **Support episode.** A left or right probe, matching the pilot's primary
  variant. The probe direction is randomized and balanced within each scene.
- **Outcome.** Native first success within 600 controls on the query episode.
  The hidden sign, handle pose and fingertip error are recorded for the
  evaluator only and never enter the learner.
- **Adapter.** `HistoryFlowSteeringPolicy` with the KV bridge interface
  (`src/fluffy_bassoon/meta_rl/flow_history.py`). The VLM and action expert
  stay frozen. Zero-initialized output keeps exact native parity before
  training.

The bridge acts at every chunk. It has to learn from its normal inputs when
to apply the correction, before or after the trigger point, so the privileged
geometry trigger is not available to it. The small reader keeps its trigger
because it was trained to act only there. The comparison between the two is
reported as such, not as a matched architecture ablation.

## Data

**Training scenes are all Pro6K.** The 16 surplus eligible scenes from the
pilot were collected on Pro6K and never used in any readout. Together with
the pilot's 8 held-out scenes, also Pro6K, they form 24 training scenes. The
pilot's 8 training scenes were collected on 4090. They stay out of training
here; their seeds are used only to draw fresh development resets.

**The pilot tapes lack camera frames.** They store compact action tapes,
public state and rewards. The bridge needs the three camera views at every
chunk along each branch. For each needed branch, replay its exact action tape
on Pro6K and record the camera views and public inputs. Compare the physics
hash at every control and the terminal success. A branch that fails replay is
recorded and excluded, together with its trial, from both arms.

**Training targets** (DPT) follow `constructed_queries.py --action-stats`:

- **Before the decision point:** the native chunk, so the bridge should leave
  the motor unchanged.
- **Inside the 100-control correction window:** the delivered commands of the
  measured successful correction branch.
- **After the first grasp:** the native chunk again.

The decision point stays a labeling convention for training. It is never a
deployment input.

**The memoryless control** gets the same targets, scenes, model width,
optimizer budget, sampler seed and fixed endpoint. It also gets the attempt
index, but no support tokens.

**Fresh evaluation scenes.** Screen new `OpenDrawer` seeds on Pro6K, using
the pilot's nominal admission rule unchanged, until 16 eligible scenes are
reached in seed order. Retain the scenes that fail screening in the admission
ledger. Each eligible scene gets four trials (2 hidden signs × 2 probe
directions), for 64 trials. These scenes select nothing.

## Steps and the decision each protects

**0. CPU readout of the pilot's DIT critic (no GPU).** Rank the two nonzero
corrections within each fixed history, using the saved critics. The pooled
AUC of 0.748–0.800 can come from rejecting the always-failing zero correction
alone. This decides whether a DIT bridge is worth a later arm. It does not
gate the DPT test.

**1. Authority check (development resets only).** Fit one bridge with the
hidden sign given as a privileged input. Run it closed loop on 8 fresh
development resets drawn from the pilot's former 4090 training seeds,
executed on Pro6K: 2 signs × 8 = 16 episodes. Compare with the best fixed
correction and with the known-sign scripted correction on the same resets.

If the privileged bridge does no better than the best fixed correction, the
frozen motor cannot express the correction through this interface. In that
case, stop and report it as a motor-authority result: it says nothing about
learning from history. The bridge's `scale` parameter is fixed before this
step, from the measured action change it can produce. It is not tuned on
outcomes.

**2. Small reader in closed loop.** Run the pilot's three main DPT fit seeds
unchanged, with their trigger, on the 64 fresh trials. This separates "closed
loop breaks the result" from "the adapter loses the information".

**3. Full bridge fits.** Fit DPT and memoryless bridges with three fit seeds
each, to a fixed endpoint without held-out selection. Save complete actor,
optimizer, sampler and RNG state at each milestone.

**4. Closed-loop readout on the 64 fresh trials.** Every query arm branches
from the same completed support record, with the same query reset and motor
noise:

| Query arm | Purpose |
| --- | --- |
| DPT bridge, own history | Primary |
| DPT bridge, cleared history | Reliance on any retained input |
| DPT bridge, random donor | Donor support drawn from another trial of the same task and probe direction, hidden sign at random. Tests whether the *particular* experience matters. |
| DPT bridge, opposite-sign donor | Contradictory history; should push the correction the wrong way |
| DPT bridge, reward masked | Feedback dependence (an evaluation-only input change) |
| Memoryless bridge | Equally trained control |
| Small DPT reader (step 2) | Positive control |
| Frozen π0.5, no correction | Native retry |
| Best fixed correction | Strongest memoryless policy available |
| Known-sign scripted correction | Upper bound |

**Primary contrasts:**
- Own history versus cleared history.
- Own history versus the random donor.
- Own history versus the memoryless bridge.

For each contrast, report paired wins and losses, exact McNemar p-values and
scene-clustered bootstrap intervals. All other contrasts are descriptive.

Alongside native success, record the commanded lateral displacement at the
first grasp. A shift in the correct direction without a success gain shows
sensitivity to history, not adaptation.

## Interpretation

| Outcome | Reading | Next |
| --- | --- | --- |
| Step 1 fails | The interface or motor cannot express the correction | Rework the interface (e.g. velocity bridge, larger scale) before any learner test |
| Small reader holds in closed loop, bridge own ≈ cleared ≈ memoryless | Information is lost in the adapter or objective | Rework representation or objective; do not expand tasks |
| Bridge own > cleared and > random donor; opposite donor reverses it | History survives the full adapter | Build the multi-condition adaptation distribution, then compare AD/DPT/DIT/IC-IQL |
| Small reader also fails closed loop | The pilot's branch-table scoring overstated usable history | Diagnose closed-loop mismatch first |

Whatever the outcome, this remains one hand-designed condition on one task.
It does not establish adaptation to held-out tasks or held-out condition
families.

## Cost and limits

The following are estimates, not measurements. The pilot's parallel
collection averaged roughly 40 GPU-seconds per native episode, including
resets.

| Item | Episodes or fits | Estimate (GPU-hours) |
| --- | --- | ---: |
| Tape replay with camera recording | about 24 scenes × 9 branches | 3 |
| Fresh scene screening | about 32 scenes | 3 |
| Authority check | 16 development episodes, plus one fit | 2 |
| Bridge fits | 7 fits through the frozen expert | 8–12 |
| Closed-loop readout | 64 supports, plus 64 trials × about 20 query arms and seeds | 15 |
| **Total** | | **about 31–35** |

This is within the 50 GPU-hour discretionary limit. Record a 40 GPU-hour
all-attempt ceiling in the execution contract, and request the bridge fit
throughput on the actual allocation before committing to all 7 fits. Bind
inference and rendering to allocated GPUs, and run the GPU probe inside each
job step. Keep training and evaluation on Pro6K.

Known limits that remain:
- One task, a single 4 cm magnitude, and a binary hidden sign.
- Admission conditional on nominal motor success.
- 16 evaluation scenes.
- Three fit seeds, which are not independent datasets.
- No cross-hardware trajectory equivalence is claimed.
