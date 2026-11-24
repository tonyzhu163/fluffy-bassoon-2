# Native selector launch

Baige Distributed Training started `job-umsgs123dma7` on one RTX 4090 on November 24 at 11:21 Shanghai. The two-hour attempt uses the registered 80 source and 36 held-arm query episodes with three independent 15-control calibrations. The shared attempt directory is `/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-selector-20261124-r1/native01`. [Plan](PLAN.md) · [Baige job](https://console.bce.baidu.com/aihc/infoTaskIndex/detail?clusterUuid=aihc-serverless&k8sNamespace=default&k8sName=job-umsgs123dma7&kind=PyTorchJob&name=frozen-vla-selector-pr10-native01&jobId=job-umsgs123dma7&queueID=aihcq-chqrelndaaj4&from=list)

The job's launcher verified commit `85696bd8231b8115cc0d7959a1662dad46bb71b2`. The allocation probe verified the scheduler-assigned GPU and completed a CUDA computation on the RTX 4090. [Launch pin](launch.txt) · [Device probe](gpu.json)

All 77 selector, simulator, GPU admission and device policy tests passed on Baige Linux. Four short real-simulator branches per embodiment reached the same control-45 fork using a deterministic policy. All twelve had matching resets and valid executed-action receipts. The camera calibration was frozen on Panda and tested against the archived Kinova3 review cases before launch. [Tests](pytest.log) · [Device-mask cases](device-mask.log) · [Simulator branches](selector-smoke.json) · [Camera validation](../frozen-vla-selector-20261124-r1/drawer-evidence.md)

At 11:30 Shanghai, all four Panda layout 0 branches had finished successfully. The control-45 reset, prefix, noise, features and state matched. Nominal and repeat issued identical controls. The largest component change over the five consumed controls was 0.03442 for correction and 0.01779 for the cue. [First complete native case](Panda-0.json)
