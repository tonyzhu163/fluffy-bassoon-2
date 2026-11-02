# Original versus generalized visual completion rubric

Frozen comparison on634 existing videos:384 Flora-PPO training episodes and250 earlier calibration episodes. Both prompts scored every video with the same model, camera bytes/order, sampler, output format and reward map. No new robot episodes or policy updates. This is development evidence; independent qualification remains separate.

| Video bank | Saved original AUC | Rerun original AUC | Candidate AUC | Original false Score5 | Candidate false Score5 | True Score5 lost / gained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TRAIN/OpenDrawer | 0.925 | 0.927 | 0.919 | 58/183 | 43/183 | 35 / 0 |
| CALIBRATION/OpenDrawer | 0.966 | 0.967 | 0.954 | 3/21 | 1/21 | 6 / 0 |
| CALIBRATION/PickPlaceCounterToCabinet | 0.900 | 0.902 | 0.827 | 0/13 | 0/13 | 4 / 0 |
| CALIBRATION/PickPlaceDrawerToCounter | 0.977 | 0.977 | 0.974 | 3/27 | 2/27 | 1 / 0 |
| CALIBRATION/TurnOffStove | 0.894 | 0.894 | 0.904 | 6/43 | 17/43 | 0 / 0 |
| CALIBRATION/CoffeeSetupMug | 0.923 | 0.930 | 0.927 | 8/33 | 8/33 | 0 / 1 |

Of the58 TRAIN failures originally given Score5, the candidate gives a lower grade to **20** and still gives Score5 to **38**. Rerun-original and candidate grade histograms are retained in RESULT.json, so an original replay difference is distinguishable from a rubric change.

| Video bank | False full successes corrected / introduced | Paired two-sided p | Score5 precision: original → candidate | Score5 recall: original → candidate | Original replay grade changes |
| --- | ---: | ---: | ---: | ---: | ---: |
| TRAIN/OpenDrawer | 20 / 5 | 0.004 | 0.772 → 0.789 | 0.975 → 0.801 | 10 |
| CALIBRATION/OpenDrawer | 2 / 0 | 0.500 | 0.906 → 0.958 | 1.000 → 0.793 | 1 |
| CALIBRATION/PickPlaceCounterToCabinet | 0 / 0 | 1.000 | 1.000 → 1.000 | 0.216 → 0.108 | 2 |
| CALIBRATION/PickPlaceDrawerToCounter | 1 / 0 | 1.000 | 0.885 → 0.917 | 1.000 → 0.957 | 1 |
| CALIBRATION/TurnOffStove | 0 / 11 | 0.001 | 0.500 → 0.261 | 0.857 → 0.857 | 2 |
| CALIBRATION/CoffeeSetupMug | 2 / 2 | 1.000 | 0.667 → 0.680 | 0.941 → 1.000 | 1 |

All58 original full-score failures:

| Original episode | Native instruction | Saved grade | Rerun original | Candidate | Original reward | Candidate reward |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| flora-single-case-000112-rank-06-episode-000001 | Open the left drawer. | 5 | 5 | 5 | 0.9678 | 0.9772 |
| flora-single-case-000064-rank-00-episode-000001 | Open the left drawer. | 5 | 5 | 5 | 0.9664 | 0.9725 |
| flora-single-case-000272-rank-02-episode-000004 | Open the right drawer. | 5 | 5 | 5 | 0.9506 | 0.8595 |
| flora-single-case-000376-rank-07-episode-000005 | Open the right drawer. | 5 | 5 | 5 | 0.9288 | 0.9161 |
| flora-single-case-000216-rank-03-episode-000001 | Open the right drawer. | 5 | 5 | 5 | 0.9317 | 0.9062 |
| flora-single-case-000256-rank-00-episode-000002 | Open the left drawer. | 5 | 5 | 5 | 0.9157 | 0.8338 |
| flora-single-case-000360-rank-05-episode-000007 | Open the right drawer. | 5 | 5 | 5 | 0.9180 | 0.9198 |
| flora-single-case-000184-rank-07-episode-000004 | Open the left drawer. | 5 | 5 | 5 | 0.9132 | 0.9480 |
| flora-single-case-000224-rank-04-episode-000006 | Open the left drawer. | 5 | 5 | 5 | 0.9055 | 0.8607 |
| flora-single-case-000232-rank-05-episode-000006 | Open the right drawer. | 5 | 5 | 5 | 0.8795 | 0.8720 |
| flora-single-case-000040-rank-05-episode-000000 | Open the right drawer. | 5 | 5 | 5 | 0.8842 | 0.8677 |
| flora-single-case-000144-rank-02-episode-000000 | Open the left drawer. | 5 | 5 | 5 | 0.8747 | 0.8670 |
| flora-single-case-000056-rank-07-episode-000002 | Open the right drawer. | 5 | 5 | 5 | 0.8753 | 0.8822 |
| flora-single-case-000040-rank-05-episode-000005 | Open the right drawer. | 5 | 5 | 5 | 0.8752 | 0.8675 |
| flora-single-case-000040-rank-05-episode-000003 | Open the right drawer. | 5 | 5 | 5 | 0.8244 | 0.8153 |
| flora-single-case-000048-rank-06-episode-000001 | Open the right drawer. | 5 | 5 | 5 | 0.8433 | 0.8787 |
| flora-single-case-000240-rank-06-episode-000002 | Open the right drawer. | 5 | 5 | 5 | 0.8235 | 0.8017 |
| flora-single-case-000336-rank-02-episode-000000 | Open the left drawer. | 5 | 5 | 4 | 0.8279 | 0.7677 |
| flora-single-case-000232-rank-05-episode-000005 | Open the right drawer. | 5 | 5 | 5 | 0.8166 | 0.7703 |
| flora-single-case-000168-rank-05-episode-000005 | Open the right drawer. | 5 | 5 | 5 | 0.7965 | 0.7857 |
| flora-single-case-000024-rank-03-episode-000005 | Open the right drawer. | 5 | 5 | 4 | 0.7929 | 0.6040 |
| flora-single-case-000000-rank-00-episode-000004 | Open the right drawer. | 5 | 5 | 5 | 0.7760 | 0.7595 |
| flora-single-case-000248-rank-07-episode-000001 | Open the right drawer. | 5 | 5 | 4 | 0.7802 | 0.7899 |
| flora-single-case-000040-rank-05-episode-000001 | Open the left drawer. | 5 | 5 | 5 | 0.7245 | 0.7658 |
| flora-single-case-000280-rank-03-episode-000001 | Open the left drawer. | 5 | 5 | 5 | 0.7336 | 0.6074 |
| flora-single-case-000336-rank-02-episode-000004 | Open the right drawer. | 5 | 5 | 5 | 0.7340 | 0.7830 |
| flora-single-case-000080-rank-02-episode-000000 | Open the right drawer. | 5 | 5 | 5 | 0.7175 | 0.7169 |
| flora-single-case-000176-rank-06-episode-000006 | Open the right drawer. | 5 | 5 | 4 | 0.6873 | 0.6925 |
| flora-single-case-000360-rank-05-episode-000000 | Open the left drawer. | 5 | 5 | 1 | 0.6830 | 0.4433 |
| flora-single-case-000280-rank-03-episode-000005 | Open the right drawer. | 5 | 5 | 5 | 0.6819 | 0.7633 |
| flora-single-case-000088-rank-03-episode-000000 | Open the left drawer. | 5 | 5 | 5 | 0.6716 | 0.7221 |
| flora-single-case-000304-rank-06-episode-000000 | Open the left drawer. | 5 | 5 | 5 | 0.6618 | 0.7664 |
| flora-single-case-000280-rank-03-episode-000007 | Open the right drawer. | 5 | 5 | 3 | 0.6527 | 0.5473 |
| flora-single-case-000248-rank-07-episode-000006 | Open the right drawer. | 5 | 5 | 4 | 0.6518 | 0.6916 |
| flora-single-case-000000-rank-00-episode-000000 | Open the right drawer. | 5 | 5 | 5 | 0.6334 | 0.6830 |
| flora-single-case-000128-rank-00-episode-000004 | Open the left drawer. | 5 | 5 | 5 | 0.6306 | 0.6987 |
| flora-single-case-000344-rank-03-episode-000005 | Open the right drawer. | 5 | 5 | 4 | 0.6463 | 0.6762 |
| flora-single-case-000376-rank-07-episode-000001 | Open the left drawer. | 5 | 5 | 4 | 0.6188 | 0.6422 |
| flora-single-case-000344-rank-03-episode-000004 | Open the right drawer. | 5 | 5 | 5 | 0.6112 | 0.7555 |
| flora-single-case-000120-rank-07-episode-000006 | Open the right drawer. | 5 | 5 | 5 | 0.6223 | 0.7206 |
| flora-single-case-000328-rank-01-episode-000002 | Open the right drawer. | 5 | 5 | 5 | 0.5958 | 0.6724 |
| flora-single-case-000232-rank-05-episode-000004 | Open the left drawer. | 5 | 5 | 3 | 0.6058 | 0.5575 |
| flora-single-case-000072-rank-01-episode-000002 | Open the right drawer. | 5 | 5 | 5 | 0.5978 | 0.6609 |
| flora-single-case-000112-rank-06-episode-000005 | Open the left drawer. | 5 | 5 | 5 | 0.6048 | 0.6767 |
| flora-single-case-000136-rank-01-episode-000005 | Open the right drawer. | 5 | 5 | 3 | 0.5613 | 0.6229 |
| flora-single-case-000360-rank-05-episode-000002 | Open the left drawer. | 5 | 5 | 4 | 0.5801 | 0.6356 |
| flora-single-case-000216-rank-03-episode-000003 | Open the right drawer. | 5 | 5 | 5 | 0.5507 | 0.6807 |
| flora-single-case-000072-rank-01-episode-000006 | Open the left drawer. | 5 | 5 | 5 | 0.5722 | 0.6051 |
| flora-single-case-000176-rank-06-episode-000007 | Open the left drawer. | 5 | 5 | 5 | 0.5603 | 0.6038 |
| flora-single-case-000376-rank-07-episode-000003 | Open the left drawer. | 5 | 5 | 3 | 0.5480 | 0.5413 |
| flora-single-case-000072-rank-01-episode-000001 | Open the left drawer. | 5 | 5 | 5 | 0.5343 | 0.5613 |
| flora-single-case-000040-rank-05-episode-000006 | Open the right drawer. | 5 | 5 | 2 | 0.5480 | 0.4230 |
| flora-single-case-000256-rank-00-episode-000004 | Open the right drawer. | 5 | 5 | 2 | 0.5294 | 0.4920 |
| flora-single-case-000272-rank-02-episode-000000 | Open the right drawer. | 5 | 5 | 4 | 0.5274 | 0.5857 |
| flora-single-case-000240-rank-06-episode-000007 | Open the left drawer. | 5 | 5 | 2 | 0.5250 | 0.5162 |
| flora-single-case-000032-rank-04-episode-000001 | Open the left drawer. | 5 | 5 | 3 | 0.5362 | 0.5998 |
| flora-single-case-000080-rank-02-episode-000001 | Open the left drawer. | 5 | 5 | 3 | 0.5257 | 0.5751 |
| flora-single-case-000208-rank-02-episode-000003 | Open the right drawer. | 5 | 5 | 1 | 0.5056 | 0.4489 |

A greedy Score5 is a full-success claim. The expected-grade reward remains a soft five-grade quantity; a lower greedy grade does not imply binary-failure reward. AUROC, precision and recall must be considered together. All failed and successful cases remain in the exported paired records.

The text was frozen before replay. These same videos informed the prior error investigation, so this comparison cannot establish generalization to unseen videos or a PPO improvement. Retained original/rerun differences measure numerical and batch-composition reproducibility separately.

[Full paired CSV](PAIRED-EPISODES.csv) · [58-case CSV](FOCUS-58.csv) · [Machine-readable comparison](RESULT.json) · [Frozen manifest](MANIFEST.json)
