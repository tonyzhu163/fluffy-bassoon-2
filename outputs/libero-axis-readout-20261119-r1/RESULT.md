# More of the calibration signal was linearly readable after the support motions

On the held-out double reversal, a linear calibration decoder applied immediately after the two support tokens recovered both signs on 27, 33 and 33 of 60 development variants. Passing the query through the frozen reader and applying the same ridge procedure at that position gave 9, 11 and 22. All counts use the same 15 task/reset pairs under four probe directions. The calibration readouts multiply their two estimated signs by the motor's nominal first command signs. The original head learns that conversion jointly.

| Frozen features or head | Familiar conventions, matched history /180 | Double reversal, matched /60 | Double reversal, opposite /60 |
| --- | --- | --- | --- |
| Original trained command head | 121, 129, 111 | 28, 22, 23 | 0, 0, 0 |
| Trained encoder, final support token | 166, 166, 172 | 27, 33, 33 | 0, 0, 0 |
| Random encoder, final support token | 161, 161, 164 | 25, 26, 15 | 0, 0, 0 |
| Trained encoder, query token | 130, 122, 150 | 9, 11, 22 | 0, 0, 4 |
| Random encoder, query token | 138, 124, 123 | 21, 17, 20 | 0, 0, 8 |
| Analytic command–displacement rule | 180 | 60 | 0 |

Seeds are 41, 42 and 43 in every row. The probe after the second support token exceeded the seed-matched random features on the double reversal in all three fits. It exceeded the original learned command head in two fits. Random features also supplied substantial signal, and the analytic reference still recovered every command. These comparisons localize a linearly accessible signal in the fitted reader. They leave nonlinear information, the effect of broader trajectory diversity, and differences between VLA initializations open.

The original balanced command heads achieved 192/192 on the familiar fitting examples at every seed. Their 111–129/180 on those same conventions on disjoint development tasks exposes a generalization problem even before adding the withheld convention. The probe reused this exploratory development panel; it used fitting data only for its normalization and 12 ridge fits.

Baige development CPU `cpu01`, source `80f0eef854daf5e9b937acac9f67cec0fd6b37df`, exit 0 in four seconds, zero GPU-hours. All 1,440 original matched/opposite prediction rows replayed with identical logits, signs and targets. The readout recomputed the metrics from the saved per-trial arrays. The fit coefficients, provenance and predictions are under `/mowenjie/fluffy-bassoon-2/outputs/libero-axis-readout-20261119-r1/cpu01/probes/`. [Counts and artifact hashes](cpu01-readout.json) · [Implementation](probe.py) · [Original joint-reader experiment](../libero-axis-calibration-20261119-r1/RESULT.md)
