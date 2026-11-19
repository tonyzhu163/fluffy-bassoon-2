# A fresh physical probe restored native success under command reversals

The preregistered seed-41 shared-axis MLP chose the correct command convention on 64/64 physical support episodes. Correctly transformed playback of the frozen π0.5 action tapes reached native task success on all 64 convention/query combinations. Clearing support selected nominal commands and reached 16/64. Supplying the physical probe from the opposite convention reached 0/64. The analytic command–motion estimator tied the trained model on every case.

| Physical support and decoder | Nominal /16 | X reversal /16 | Y reversal /16 | Held-out XY reversal /16 |
| --- | ---: | ---: | ---: | ---: |
| Shared-axis MLP | 16 | 16 | 16 | 16 |
| Analytic command–motion estimator | 16 | 16 | 16 | 16 |
| Always nominal | 16 | 0 | 0 | 0 |
| Shared MLP, cleared support | 16 | 0 | 0 | 0 |
| Shared MLP, opposite-convention support | 0 | 0 | 0 | 0 |
| True-convention decoder | 16 | 16 | 16 | 16 |

The native experiment used all 16 query resets on eight development tasks. A different, previously unused support reset supplied two five-control pulses under each physical convention. The reader received the issued commands and the following wrist displacements at the fifth control of each pulse. The known coordinate convention was available to the simulator and the evaluator. The learner saw only the two public command–motion pairs. For each query start, all four possible net command transforms were executed to native termination, truncation or the original tape budget. The method comparisons select from those four measured outcomes. These are open-loop motor tapes generated earlier by a closed-loop policy. They establish a native consequence of the inferred calibration under a constructed fault on BASE environments. They do not measure live VLA replanning, the published benchmark score or the contribution of the pretrained motor to identifying the fault.

All newly measured query branches for a given start had identical simulator-state fingerprints. Twelve query starts also matched the archive's full-state fingerprint. Within that subset, retained support scored 48/48, cleared support 12/48 and opposite support 0/48. On the four Object-suite starts with different archive fingerprints the respective scores were 16/16, 4/16 and 0/16. All five public robot fields matched the archive on every start. Nominal playback reproduced all 16 success outcomes, control counts and reward sequences. The largest nominal wrist-path discrepancy across all controls was 0.021 mm or less. The full-state hash difference remains unexplained. [Verified scores, reset comparisons and hashes](cpu02-scores.json)

## Research decision

Close the sign-reversal mechanism as a positive control. The prior joint learner failed the withheld combination, while sharing one small estimator across the axes solved it and matched the analytic inverse. These inputs contain no images or text, so this ceiling provides no meaningful comparison of pretrained visual representations. [Matched architecture comparison](../libero-axis-factor-20261119-r1/RESULT.md)

Released S18 adds a constant public position bias that cancels out of this reader's displacement inputs. The existing S20 cadence panel gave prior-attempt and current-attempt timing policies the same 11/16 fresh query successes. A VLA initialization comparison needs a development decision for which perceptual interpretation of the earlier attempt changes attainable return, after accounting for what the current attempt can reveal. [Released S18 test](../libero-s18-probe-20261119-r1/RESULT.md) · [Live S20 comparison](../libero-first-controls-20261119-r1/RESULT.md)

Baige development CPU `cpu02` at `b669b0b92b8717e84ba64d857a8e1cdbaee77f72`, exit 0 in 298 seconds. Source archive `menu02` at `b90ee5a0`. The earlier `cpu01` exited during archive validation after 36 short probes and 32 query branches in 160 seconds. Those outcomes were excluded from the second attempt. Both runs used zero GPU-hours. The physical probe traces, per-control query rewards, endpoint records, score readback and logs are under `/mowenjie/fluffy-bassoon-2/outputs/libero-axis-native-20261119-r1/cpu02/`. [Run and retry protocol](PLAN.md) · [Verified readout](score.py)
