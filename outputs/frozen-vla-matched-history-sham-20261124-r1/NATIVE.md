# Execution history changed actions on all eight cases and success on none

The frozen JOINT policy succeeded on **8/8 with real history, 8/8 with the runtime-zero sham, and 8/8 with the plain sampler**. Real minus sham produced 0 rescues, 0 losses and 8 ties. Sham minus plain also produced 0 rescues, 0 losses and 8 ties. Each condition succeeded on 4/4 Panda and 4/4 Kinova3 cases. Both paired success differences were zero percentage points on each robot and after excluding any one of the four shared task/layout starts. [Paired counts and leave-one-start-out sensitivity](evidence/native-contrasts.json)

Both zero controls reached the native-success ceiling. These fixed cases could reveal a loss from real history, and none occurred. They offered no baseline failure that history could rescue. Each task/layout/robot had one registered noise bank. Generalization to other starts, noises and tasks remains unresolved.

## Native endpoints

Every episode terminated on native success, returned 1 and finished before its deadline. The three middle columns give executed controls to success. The last two columns give maximum absolute differences across the first five proposed actions at control 10, evaluated on identical inputs and never issued. “Content” compares real with sham on the history executable. “Graph” compares sham with plain at zero addition. [Full native reward, proposal and execution tapes](evidence/native-episodes.tar.gz) · [Paired audit](evidence/native-paired.json)

| Task | Row | Robot | Real | Sham | Plain | Content | Graph |
|---|---:|---|---:|---:|---:|---:|---:|
| Alphabet soup | 48 | Kinova3 | 212 | 244 | 183 | 0.004884 | 0.001239 |
| Alphabet soup | 48 | Panda | 133 | 139 | 139 | 0.005043 | 0.001090 |
| Alphabet soup | 49 | Kinova3 | 145 | 145 | 145 | 0.004434 | 0.001541 |
| Alphabet soup | 49 | Panda | 133 | 132 | 133 | 0.005517 | 0.001579 |
| Akita bowl | 48 | Kinova3 | 105 | 106 | 106 | 0.005693 | 0.002681 |
| Akita bowl | 48 | Panda | 104 | 105 | 105 | 0.006047 | 0.001330 |
| Akita bowl | 49 | Kinova3 | 74 | 79 | 79 | 0.006047 | 0.001884 |
| Akita bowl | 49 | Panda | 68 | 68 | 68 | 0.007264 | 0.001333 |

## What the controls establish

All eight triplets matched on reset, observation, diffusion noise, empty history, opening joint kernel, proposal and the first five issued controls. The first post-opening observations and computed offsets also matched. Both contrasts first changed an issued command at zero-based control 5 in all eight cases. Subsequent replans used each trajectory's own observations and public execution history and the registered exogenous noise at that control index. Replay reconstructed the history and the bank from the saved tapes. All first reset attempts were accepted. [Paired audit](evidence/native-paired.json) · [Readback code](audit.py)

The 189 post-opening real calls and 198 sham calls used one precompiled history executable and one frozen JOINT kernel in one process. The sham computed a nonzero offset on every post-opening call, then passed an identically zero array as the runtime argument. The plain arm used the plain executable throughout. At control 10, both repeated samples (real and sham) were bitwise reproducible on every case. Changing history content and switching the arithmetic graph each changed the proposal at fixed observation, noise, RNG and kernel in 8/8 cases. The content difference exceeded the graph difference on the first five proposed actions on each of the eight inputs. Downstream times to success include the effects of diverging trajectories. [Sampler](sham.py) · [Executable identities](evidence/native-graphs.json) · [Recorded proposals](evidence/native-episodes.tar.gz)

The fit remained sealed at SHA-256 `4855546591381f149d7509a6af3d45db32e9e82a732788c8376d504f19417dbc`. Independent disk reads before and after recovered the same JOINT factors, head archive, checkpoint manifest and normalization. The GPU-resident stock, heads and joint kernel also matched before and after. Fitting used 192 source windows from 21 Panda trajectories. The two task identities were familiar from source fitting. The registered evaluation layouts and Kinova3 rollouts supplied no fitting examples or model-selection feedback. [Independent fit receipt before](evidence/fit-independent-before.json) · [Independent fit receipt after](evidence/fit-independent-after.json) · [Resident receipts and verified hashes](evidence/native-readback.txt) · [Enrollment contract](../frozen-vla-cotraining-20261124-r1/matched-task/contract.json)

## Execution and verification

The [plan](PLAN.md) was committed at `3c693063e7f0f9a39b64cc9a0c6542d5623af6df` before opening rollout outcomes. The first CPU fixture stopped during JAX lowering on an OpenPI argument-type check. The qualified revision scoped OpenPI's type-check override to ahead-of-time compilation. All 24 short CPU fixtures then passed on two Baige cores using the real simulator, trained projection factors and lightweight upstream embeddings. [CPU readback](evidence/cpu-paired.json)

Baige Distributed Training job `job-tgrezp06uwab` ran pin `8b0230bfb852416d036a04db8ecc83ae90b7e900` in one exclusive RTX 4090 attempt. The launcher exited 0 after 596 seconds (0.1656 launcher GPU-hours). A separate Linux CPU process reproduced the complete paired audit at 12:34:15 Shanghai time on November 24. The local lossless archive verified against all 24 per-episode SHA-256 digests. Full PFS output is under `/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-matched-history-sham-20261124-r1/gpu01`. [Device receipt](evidence/native-gpu.json) · [Exit and timing](evidence/native-exit.txt) · [Replay receipt](evidence/native-readback.txt) · [Pinned paths and command](evidence/submission-handoff.txt)
