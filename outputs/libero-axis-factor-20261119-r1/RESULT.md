# Sharing an axis estimator recovered the withheld combination

All three shared MLPs recovered both signs on all 60 double-reversal development variants. All three joint MLPs scored zero on the same variants despite high accuracy on the familiar conventions. Both architectures estimated a fixed calibration and applied it algebraically to the same frozen π0.5 proposal. Both trained on exactly the same per-seed batches and number of optimizer updates. There were 16 fitting pairs and 15 eligible development pairs on disjoint groups of eight tasks. Four reflected directions per development pair gave the 60 variants.

| Estimator, seeds 41, 42, 43 | Familiar /180 | Double reversal /60 | Double reversal with opposite support /60 |
| --- | --- | --- | --- |
| Joint MLP | 177, 177, 175 | 0, 0, 0 | 0, 0, 0 |
| MLP reused on each axis | 180, 180, 180 | 60, 60, 60 | 0, 0, 0 |
| Command–motion correlation | 180 | 60 | 0 |

Clearing support gave 0/60 on the double reversal for both learned arms. The architecture aligned the two probes by axis and shared their estimator. The model learned the relation between command and motion from raw values. The representation supplied the known independence of the two axes, which makes the withheld combination representable as two familiar one-axis decisions. This comparison does not isolate parameter tying from axis-local inputs or establish that a VLA discovered the factorization. The analytic estimator already solves this panel.

Baige development CPU `cpu01`, source `6cffa2c0d8b2be4f7acfb1ca827a7b15c7b41c7e`, exit 0, 22 seconds, zero GPU-hours. The run verified identical seed-paired batch hashes and Torch–NumPy agreement on all seed-41 development decisions. The six resumable checkpoints, estimates, portable reader, direction-specific counts and logs are under `/mowenjie/fluffy-bassoon-2/outputs/libero-axis-factor-20261119-r1/cpu01/`. [Counts and configuration](cpu01-summary.json) · [Physical evaluation](../libero-axis-native-20261119-r1/PLAN.md)
