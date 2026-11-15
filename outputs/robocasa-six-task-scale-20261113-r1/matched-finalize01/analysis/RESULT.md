# Matched three- versus six-task KV comparison

40 fresh cases per task, all eight fixed-final models and SFT/PPO30 controls, 13,200 audited physical episodes. Native episode-six success and paired history effects are separate from first-attempt competence.

| Model | Panel | Full sixth | Reset sixth | History effect (pp) | Paired 95% interval (pp) |
| --- | --- | ---: | ---: | ---: | --- |
| 3/ad | ID | 45.0% | 47.5% | -2.5 | [-9.2, 5.0] |
| 3/ad | OOD | 36.2% | 38.8% | -2.5 | [-11.2, 6.2] |
| 3/dpt | ID | 50.8% | 47.5% | +3.3 | [-4.2, 10.8] |
| 3/dpt | OOD | 37.5% | 41.2% | -3.8 | [-12.5, 5.0] |
| 3/dit | ID | 41.7% | 45.8% | -4.2 | [-10.8, 2.5] |
| 3/dit | OOD | 36.2% | 41.2% | -5.0 | [-13.7, 2.5] |
| 3/ic_iql | ID | 49.2% | 41.7% | +7.5 | [0.8, 15.0] |
| 3/ic_iql | OOD | 43.8% | 40.0% | +3.8 | [-3.8, 11.3] |
| 6/ad | ID | 45.0% | 44.2% | +0.8 | [-7.5, 9.2] |
| 6/ad | OOD | 40.0% | 41.2% | -1.2 | [-10.0, 7.5] |
| 6/dpt | ID | 45.0% | 46.7% | -1.7 | [-7.5, 4.2] |
| 6/dpt | OOD | 42.5% | 45.0% | -2.5 | [-11.2, 6.2] |
| 6/dit | ID | 46.7% | 45.8% | +0.8 | [-5.8, 7.5] |
| 6/dit | OOD | 40.0% | 35.0% | +5.0 | [-5.0, 15.0] |
| 6/ic_iql | ID | 47.5% | 49.2% | -1.7 | [-9.2, 5.8] |
| 6/ic_iql | OOD | 41.2% | 38.8% | +2.5 | [-5.0, 10.0] |

| Method | Panel | Six − three full sixth (pp) | Paired 95% interval | Change in history effect (pp) | Paired 95% interval |
| --- | --- | ---: | --- | ---: | --- |
| ad | ID | +0.0 | [-9.2, 8.3] | +3.3 | [-7.5, 14.2] |
| ad | OOD | +3.8 | [-5.0, 12.5] | +1.2 | [-11.3, 15.0] |
| dpt | ID | -5.8 | [-14.2, 2.5] | -5.0 | [-15.0, 5.8] |
| dpt | OOD | +5.0 | [-5.0, 15.0] | +1.2 | [-11.3, 13.8] |
| dit | ID | +5.0 | [-3.3, 13.3] | +5.0 | [-4.2, 14.2] |
| dit | OOD | +3.8 | [-6.2, 13.7] | +10.0 | [-5.0, 25.0] |
| ic_iql | ID | -1.7 | [-8.3, 5.0] | -9.2 | [-19.2, 0.8] |
| ic_iql | OOD | -2.5 | [-11.2, 6.2] | -1.2 | [-11.3, 8.8] |

| Corpus | Method contrast | Panel | Full sixth difference (pp) | Paired 95% interval | History-effect difference (pp) | Paired 95% interval |
| --- | --- | --- | ---: | --- | ---: | --- |
| 3 | ad − dpt | ID | -5.8 | [-14.2, 2.5] | -5.8 | [-15.8, 4.2] |
| 3 | ad − dit | ID | +3.3 | [-5.0, 11.7] | +1.7 | [-8.3, 12.5] |
| 3 | ad − ic_iql | ID | -4.2 | [-11.7, 3.3] | -10.0 | [-20.0, -0.0] |
| 3 | dpt − dit | ID | +9.2 | [0.0, 18.3] | +7.5 | [-2.5, 17.5] |
| 3 | dpt − ic_iql | ID | +1.7 | [-5.8, 9.2] | -4.2 | [-13.3, 5.0] |
| 3 | dit − ic_iql | ID | -7.5 | [-15.0, 0.0] | -11.7 | [-21.7, -1.7] |
| 3 | ad − dpt | OOD | -1.2 | [-7.5, 5.0] | +1.2 | [-11.2, 13.7] |
| 3 | ad − dit | OOD | +0.0 | [-7.5, 7.5] | +2.5 | [-7.5, 11.3] |
| 3 | ad − ic_iql | OOD | -7.5 | [-15.0, 0.0] | -6.2 | [-18.8, 6.2] |
| 3 | dpt − dit | OOD | +1.2 | [-7.5, 10.0] | +1.2 | [-11.3, 15.0] |
| 3 | dpt − ic_iql | OOD | -6.2 | [-15.0, 2.5] | -7.5 | [-18.7, 3.7] |
| 3 | dit − ic_iql | OOD | -7.5 | [-16.2, 1.2] | -8.8 | [-20.0, 2.5] |
| 6 | ad − dpt | ID | +0.0 | [-8.3, 8.3] | +2.5 | [-7.5, 12.5] |
| 6 | ad − dit | ID | -1.7 | [-10.0, 6.7] | +0.0 | [-10.8, 10.9] |
| 6 | ad − ic_iql | ID | -2.5 | [-10.0, 5.0] | +2.5 | [-8.3, 12.5] |
| 6 | dpt − dit | ID | -1.7 | [-10.0, 6.7] | -2.5 | [-10.8, 5.8] |
| 6 | dpt − ic_iql | ID | -2.5 | [-10.8, 5.8] | +0.0 | [-10.0, 10.0] |
| 6 | dit − ic_iql | ID | -0.8 | [-9.2, 7.5] | +2.5 | [-8.3, 13.3] |
| 6 | ad − dpt | OOD | -2.5 | [-12.5, 8.7] | +1.2 | [-11.2, 15.0] |
| 6 | ad − dit | OOD | +0.0 | [-8.8, 8.8] | -6.2 | [-18.8, 6.2] |
| 6 | ad − ic_iql | OOD | -1.2 | [-10.0, 7.5] | -3.8 | [-16.3, 10.0] |
| 6 | dpt − dit | OOD | +2.5 | [-6.3, 11.3] | -7.5 | [-21.2, 6.2] |
| 6 | dpt − ic_iql | OOD | +1.2 | [-8.7, 11.2] | -5.0 | [-15.0, 5.0] |
| 6 | dit − ic_iql | OOD | -1.2 | [-11.2, 8.7] | +2.5 | [-11.3, 16.3] |

| Model | Panel | Full sixth − SFT (pp) | Paired 95% interval | Full sixth − PPO30 (pp) | Paired 95% interval |
| --- | --- | ---: | --- | ---: | --- |
| 3/ad | ID | -4.2 | [-12.5, 4.2] | -7.5 | [-17.5, 2.5] |
| 3/ad | OOD | -5.0 | [-16.3, 6.2] | -3.8 | [-13.7, 5.0] |
| 3/dpt | ID | +1.7 | [-6.7, 9.2] | -1.7 | [-10.8, 7.5] |
| 3/dpt | OOD | -3.8 | [-15.0, 7.5] | -2.5 | [-13.8, 8.8] |
| 3/dit | ID | -7.5 | [-15.8, 0.8] | -10.8 | [-19.2, -2.5] |
| 3/dit | OOD | -5.0 | [-15.0, 5.0] | -3.8 | [-13.7, 6.2] |
| 3/ic_iql | ID | +0.0 | [-7.5, 7.5] | -3.3 | [-12.5, 5.8] |
| 3/ic_iql | OOD | +2.5 | [-8.8, 13.8] | +3.8 | [-7.5, 15.0] |
| 6/ad | ID | -4.2 | [-13.3, 4.2] | -7.5 | [-17.5, 2.5] |
| 6/ad | OOD | -1.2 | [-12.5, 10.0] | +0.0 | [-12.5, 12.5] |
| 6/dpt | ID | -4.2 | [-12.5, 4.2] | -7.5 | [-16.7, 1.7] |
| 6/dpt | OOD | +1.2 | [-10.0, 12.5] | +2.5 | [-7.5, 12.5] |
| 6/dit | ID | -2.5 | [-10.0, 5.0] | -5.8 | [-15.0, 3.3] |
| 6/dit | OOD | -1.2 | [-12.5, 10.0] | +0.0 | [-12.5, 12.5] |
| 6/ic_iql | ID | -1.7 | [-9.2, 5.8] | -5.0 | [-14.2, 4.2] |
| 6/ic_iql | OOD | +0.0 | [-11.2, 11.2] | +1.2 | [-10.0, 12.5] |

PPO30 uses the matching task teacher on ID and the fixed OpenDrawer30 teacher on both OOD tasks. This OOD control measures transfer from one declared source policy, not a multi-task PPO policy or an oracle selected on target outcomes.

Task-stratified paired case bootstrap preserving all eight models, equal task weighting, marginal95% intervals without multiplicity correction. All six method pairs within each corpus are descriptive comparisons, not a familywise winner test. One fit per cell; no training-seed or task-population uncertainty.

This compares adding task data at fixed fitting compute. It does not isolate diversity from corpus size. OOD is relative to these meta-training tasks, not necessarily base pretraining. First-attempt success and reset disagreements are reported in RESULT.json; initial competence alone does not establish adaptation.
