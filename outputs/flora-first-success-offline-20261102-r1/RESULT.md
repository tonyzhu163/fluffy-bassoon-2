# Flora binary success rules, offline

2026-11-02. Claude Code session (tonyzhu163). CPU analysis of existing labels
only: no scorer, simulator, GPU or SGD. Input: the original-prompt labels in the
[rubric replay](../robocasa-flora-rubric-replay-20261102-r1/RESULT.md)
(saved and rerun, 634 videos). Script: [ANALYZE.py](ANALYZE.py); full numbers:
[RESULT.json](RESULT.json).

## Question

Gate 1's Flora arm needs a binary first-success reward to match the RLinf
simulator run. Which binary rule over Flora's grade probabilities keeps false
successes low enough, and how much does the current "Score 5" rule cost?

## Method

Rules: argmax grade = 5 (current), and thresholds on P(5), P(4)+P(5), expected
grade and progress. Each threshold is the smallest value whose false-positive
rate (FPR) is at most 5% or 10% on CALIBRATION/OpenDrawer (50 native episodes,
21 failures). It is then applied unchanged to TRAIN/OpenDrawer (384 Flora-PPO
episodes, 183 failures, 48 cases × 8).

## Results

| Rule on TRAIN/OpenDrawer | Recall | FPR | Precision |
| --- | ---: | ---: | ---: |
| argmax grade = 5 (current) | 0.975 | 0.317 | 0.772 |
| P5 ≥ 0.448 (calibration FPR ≤ 10%) | 0.915 | 0.158 | 0.864 |
| **P5 ≥ 0.545 (calibration FPR ≤ 5%)** | 0.891 | 0.126 | 0.886 |
| P4+P5 ≥ 0.745 (calibration FPR ≤ 5%) | 0.866 | 0.120 | 0.888 |
| expected grade ≥ 4.12 (calibration FPR ≤ 5%) | 0.871 | 0.120 | 0.888 |
| Best on TRAIN itself, recall ≥ 0.8 (reference only) | 0.81–0.86 | 0.104 | ≈0.90 |

1. **The current rule is the weak point, not the scorer.** "Score 5" is an
   argmax, often won at P5 ≈ 0.3–0.4. A calibrated P5 threshold cuts TRAIN false
   successes from 58 to 23 (FPR 32% → 13%) and loses 17 of 196 true Score-5s.
   All four continuous scores perform about the same.
2. **There is a floor of about 10% FPR.** Even a threshold tuned on TRAIN
   cannot go below 19 false successes at recall 0.8. The 23 failures still
   passing P5 ≥ 0.545 include several at P5 > 0.8 (listed in RESULT.json). Only
   drawer ground truth can show whether these are near misses, wrong drawers or
   label noise: Stage A of the [factorial plan](../robocasa-flora-rubric-factorial-20261102-r1/PLAN.md).
3. **Calibration does not transfer cleanly.** The threshold with 4.8% FPR on
   calibration gives 12.6% on TRAIN. The PPO episodes contain more confident
   failures than native rollouts, and the calibration bank has only 21 failures.
   Calibrate gate-1 thresholds on a larger failure bank from the policy being
   trained, not only on native rollouts.
4. **The graded reward used by the earlier Flora-PPO separates outcomes poorly.**
   Mean success reward minus mean failure reward on TRAIN: graded progress
   0.42, argmax-5 0.66, P5 ≥ 0.545 0.76. Within-case ranking is good for every
   continuous score (pairwise accuracy 0.92–0.93 over 655 success/failure pairs),
   so the graded reward ranks well but compresses the gap. This may have
   contributed to the flat Flora-PPO, alongside the issues in the RLinf plan.
5. False successes are spread across 33 of 48 cases and both instructions
   (left 24/90, right 34/93); case 40 has 5 of its 7 failures graded 5. Saved and
   rerun labels agree exactly on the argmax-5 rule.

## Limits

- **No first-success timing.** Flora scored each whole video once (64 sampled
  frames). A first-success rule that ends the episode needs prefix or per-chunk
  scoring, whose false-positive rate can differ from end-of-episode scoring and
  compounds over many checks per episode. That needs a GPU replay on saved
  videos (prefix videos only, no new simulator controls).
- **Horizon mismatch.** These videos are 750 controls; the RLinf simulator run
  uses 400 and the joint plan 600. Recalibrate at the gate-1 horizon.
- OpenDrawer only for TRAIN; other tasks have 50 calibration episodes each.

## Recommended for the gate-1 Flora arm

1. Replace argmax-5 with a calibrated P5 threshold (start at 0.545) as the
   binary success signal, and report its TRAIN FPR during training.
2. Before launch, run prefix scoring on the existing 384 TRAIN videos at the
   chunk boundaries the RLinf run uses, and pick the first-success rule
   (threshold, and K consecutive positive checks) from that replay.
3. Run Stage A1 of the factorial plan on the 23 remaining confident false
   positives first. If the drawer is visibly near fully open, no threshold or
   wording will fix them and the remaining ~10% FPR is the price of Flora.
