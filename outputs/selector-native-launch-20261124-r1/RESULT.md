# Native selector results

The run completed all 116 native frozen-policy episodes. All nine branches on Kinova3 failed on layouts 36–38 and succeeded on layout 39. The two learned selectors chose different one-shot branches on all four layouts, with identical success outcomes. Even choosing the best of continue, correct and cue in hindsight rescued none of the three failures. Every paired difference in success was zero, so the layout bootstrap interval for factorized minus pooled was also zero on these four layouts. The experiment supplies no estimate of selection benefit on layouts where one of these interventions rescues an untreated failure. [Independent case readback](case-readback.json) · [Native summary](summary.json)

| Source arm | Continue | Untreated repeat | Correct | Cue |
| --- | ---: | ---: | ---: | ---: |
| Panda | 10/10 | 10/10 | 10/10 | 10/10 |
| UR5e | 10/10 | 10/10 | 10/10 | 9/10 |

Source fitting observed zero intervention rescues and one loss from cueing. Across the panel, the simulator consumed 24,980 query controls. At the 596 logged decisions it consumed 2,977 controls, including two shortened decision windows. A total of 1,524 consumed controls differed from nominal. The next discriminating question is whether a different intervention or decision time can rescue even one matched failure before fitting another selector. [Fitted selectors](selectors.json) · [Independent verification script](native_readback.py)

## Execution

Baige Distributed Training started `job-umsgs123dma7` on one RTX 4090 on November 24 at 11:21 Shanghai. The scheduler marked it successful at 11:48:21 Shanghai. The launcher ran for 1,564 seconds (0.434 GPU-hours) and exited 0. The panel included 80 source and 36 held-arm query episodes, plus three independent 15-control calibrations. [Exit receipt](exit.txt) The shared attempt directory is `/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-selector-20261124-r1/native01`. [Plan](PLAN.md) · [Baige job](https://console.bce.baidu.com/aihc/infoTaskIndex/detail?clusterUuid=aihc-serverless&k8sNamespace=default&k8sName=job-umsgs123dma7&kind=PyTorchJob&name=frozen-vla-selector-pr10-native01&jobId=job-umsgs123dma7&queueID=aihcq-chqrelndaaj4&from=list)

The job's launcher verified commit `85696bd8231b8115cc0d7959a1662dad46bb71b2`. The allocation probe verified the scheduler-assigned GPU and completed a CUDA computation on the RTX 4090. [Launch pin](launch.txt) · [Device probe](gpu.json)

All 77 selector, simulator, GPU admission and device policy tests passed on Baige Linux. Four short real-simulator branches per embodiment reached the same control-45 fork using a deterministic policy. All twelve had matching resets and valid executed-action receipts. The camera calibration was frozen on Panda and tested against the archived Kinova3 review cases before launch. [Tests](pytest.log) · [Device-mask cases](device-mask.log) · [Simulator branches](selector-smoke.json) · [Camera validation](../frozen-vla-selector-20261124-r1/drawer-evidence.md)

At 11:30 Shanghai, all four Panda layout 0 branches had finished successfully. The control-45 reset, prefix, noise, features and state matched. Nominal and repeat issued identical controls. The largest component change over the five consumed controls was 0.03442 for correction and 0.01779 for the cue. [First complete native case](Panda-0.json)
