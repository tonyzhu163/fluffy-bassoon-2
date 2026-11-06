# Eight preserved candidate measurements

All eight fixed candidates were measured on the original128 held-out queries with identical noisy actions/timesteps for native/full/reset comparisons. Four KV models and velocity-DPT pass the prospectively fixed1% screen. Original readout admission remains unchanged.

| Model | Loss reduction vs native | Steering RMS / native RMS | History RMS / native RMS | Gate |
| --- | ---: | ---: | ---: | --- |
| kv_bridge/ad | 3.194% | 3.847% | 0.307% | pass |
| velocity_bridge/ad | 0.496% | 2.012% | 0.353% | fail |
| kv_bridge/dpt | 6.807% | 6.344% | 0.380% | pass |
| velocity_bridge/dpt | 1.946% | 3.435% | 0.218% | pass |
| kv_bridge/dit | 10.779% | 6.646% | 0.328% | pass |
| velocity_bridge/dit | 0.528% | 2.041% | 0.145% | fail |
| kv_bridge/ic_iql | 3.208% | 3.853% | 0.273% | pass |
| velocity_bridge/ic_iql | 0.521% | 1.922% | 0.068% | fail |

KV-AD native/full/reset losses are0.13427240/0.12998429/0.12996786. The fitted controller is not numerically identical to native: its steering RMS is3.847% of native velocity RMS. History has a much smaller effect (0.307% of native RMS), and clearing history slightly improves this held-out imitation loss. This does not establish useful robot adaptation.

These losses are unweighted active-coordinate flow MSE. DPT has different targets; DIT/IC-IQL fitting used value weights. Do not rank methods by these raw cross-objective losses or equate DIT/IQL measurements to their weighted fit validation scores.

No weights changed and no simulator controls or optimizer updates occurred. Actual job-step GPU probing passed. All candidate SHAs match the finalized-fit index; the native identity and parameters were checked. CPU qualification passed35 tests, including fully frozen export measurement.

Job`job-k4fln7grvrtd` succeeded/released at16:36:07 Shanghai. Creation-to-release456GPU-s plus the earlier66GPU-s probe failure gives522GPU-s (0.145GPUh). Parent140.271111 plus this diagnostic is140.416111/150GPUh, leaving9.583889.

[Raw measurements](EIGHT-CANDIDATE-RESULT.json) · [GPU probe](DIAGNOSTIC02-PROBE.json) · [CPU tests](CPU-QUALIFICATION.log).
