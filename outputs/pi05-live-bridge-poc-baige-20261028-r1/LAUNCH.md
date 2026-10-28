# Native pi0.5 live bridge PPO launch receipt

Run `pi05-live-bridge-poc-baige-20261028-r1` was submitted to Baige as **job-61p1rncjnr1x** at
**2026-10-28 16:45:46 Shanghai** and entered Running at **16:45:57**.
Source is pinned to `c3e9cca68b93b6e777cbe5314e130bcaf85c8e14`. Evidence below was copied and checked
at **16:48:15 Shanghai**.

- Allocation: **2 NVIDIA RPBZZZ6 GPUs, 32 CPUs, 128 GiB RAM, 16 GiB shared memory**;
  scheduler wall limit **180 minutes**. Automatic retries and deletion are disabled.
- Launcher: `/mowenjie/fluffy-bassoon-2/launchers/pi05-live-bridge-poc-baige-20261028-r1/launch.sh`;
  verified SHA256 `5436f9dc4b04341ec86ff04d865296e815e8122a0a9539da47fa439d61f2adaf`.
- Remote outputs: `/mowenjie/fluffy-bassoon-2/outputs/pi05-live-bridge-poc-baige-20261028-r1`.
- Parent allocation-local two-GPU probe passed, including scheduler UUID membership,
  CUDA/NVML PCI mapping and real CUDA arithmetic. Qualification: **32 passed in 3.71 s**.
- Both children loaded the frozen motor and wrote their pinned configuration and actual
  renderer bindings. Their selected GPUs and renderer PCI addresses are distinct and match
  the parent probe. This receipt establishes admission; optimizer progress and adaptation
  results require the saved training metrics.

| Worker | CTM ticks | Parameters | State bytes/lane | CUDA visibility | PCI address |
| --- | ---: | ---: | ---: | --- | --- |
| persistent CTM, seed 0 | 4 | 202,055 | 4,864 | `0` | `0000:62:00.0` |
| Transformer-XL, seed 0 | 1 | 264,865 | 16,392 | `1` | `0000:a2:00.0` |

Each worker targets 100 updates with a 9,000-second training cap: two full trials per
update, four attempts per trial, 80 simulator steps per attempt and five executed actions
per chunk. The recurrent reader advances once per chunk. Stochastic Gaussian residual
velocities act on the first seven normalized model-action coordinates at four Euler passes;
residual scale is 1.0 and fixed sigma is 0.05. PPO uses four epochs, learning rate 1e-4,
clip 0.2 and a declared joint-KL early-stop cap of 0.05. Deterministic evaluation at updates
0, 10, 50 and 100 uses normal, memory-wiped and reward-blind conditions. Training angles
are -45/0/45 degrees; evaluation angles are -30/30 degrees with disjoint overlay splits.
The parameter budgets differ, so this initial run does not establish a matched ranking.

The frozen motor file SHA256 is
`0adb5b40f064760ff245f53ae0a4e078b86680b65f77d1f5991a173543edd5a1`.
Both children recorded the same loaded-motor fingerprint
`ed862a89a39052dad591d6542959ef0a72866b2c85eac2c84767bf55f52205dc`.

Local evidence: [machine-readable receipt](run-receipt.json),
[copy hashes](mirror-manifest.json), [parent GPU probe](gpu-probe.json),
[qualification log](qualification.log),
[CTM configuration](persistent_ctm-seed0/config.json),
[CTM renderer binding](persistent_ctm-seed0/renderer-binding.json),
[Transformer-XL configuration](transformer_xl-seed0/config.json),
[Transformer-XL renderer binding](transformer_xl-seed0/renderer-binding.json).
