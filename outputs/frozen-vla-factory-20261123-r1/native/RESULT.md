# Camera calibration recovered most of the moved-view yaw accuracy on two UR5e tasks

The seed-41 frozen SymMeta readout reached 7.61° mean absolute yaw error with known relative camera orientation, versus 28.98° with the raw readout. Its own upright error was 7.76°. Across the same ±30° views, the calibrated classical brightness-flow reader reached 12.81°. This is an offline reanalysis of eight previously recorded support trajectories. The target camera poses came directly from the simulator. There were zero new native task episodes, scheduler submissions or GPU-hours.

| Method | Upright MAE | −30° MAE | +30° MAE | Moved-view physical XY command error |
| --- | ---: | ---: | ---: | ---: |
| Unadapted | 90.00° | 90.00° | 90.00° | 0.604 |
| Classical brightness flow | 7.86° | 23.03° | 32.01° | 0.212 |
| Brightness flow + camera rotation | 7.86° | 13.38° | 12.24° | 0.113 |
| SymMeta | 7.76° | 23.73° | 34.22° | 0.225 |
| SymMeta + camera rotation | 7.76° | 8.33° | 6.89° | 0.074 |
| Privileged command-interface oracle | 0° | 0° | 0° | 0.006 |

**Pairing and uncertainty.** Reset 11 supplies four continuous interfaces per task on Long 5 and Spatial 0. Each recording replays 15 frozen π₀.₅ controls as three repeated five-step means. The paired camera renders share the same physical states. All 24 support/view rows enter the screen. Applying the relative camera-to-world rotation improved SymMeta by 21.94° on Long and 20.80° on Spatial, averaging both moved views. The paired average was 21.37°. The exact empirical 95% percentile interval from resampling two whole tasks is [20.80°, 21.94°]. Its endpoints are just the two task effects. A broad population effect cannot be estimated reliably here. The calibrated learned reader's gains over calibrated classical flow were 2.25° and 8.15° respectively.

**What the command score measures.** Each estimated map acts on the same 1,148 recorded commands from eight *other* LIBERO development tasks. We verified each NPZ command-array digest against the archived frozen π₀.₅ manifest, clipped the canonical actions to motor bounds, applied the estimated inverse and the true forward interface with native clipping, and scored mean Euclidean XY error in normalized command units. The eight tapes receive equal weight. This is open-loop algebra on replayed commands. It excludes contact dynamics, changed perception, and motor replanning. Even the oracle has residual error when inverse commands saturate. Public support displacement supplies translation scale. The grip coordinate has no effect on this XY endpoint. Native retries additionally use the real support jaw deltas for polarity.

**Qualification on November 23, 2026.** The Beijing Distributed Training console at `https://console.bce.baidu.com/aihc/tasks` redirected this session to `login.bce.baidu.com`. The VM has no configured `baige-dev` alias or project mount. A batch-mode SSH check failed to resolve the hostname. The board at the source pin reports overlapping Perceiver-retry and other jobs. Their live status, available queue capacity and quota could not be read. No existing allocation was used. The live qualifier exited 1 and lists the missing runtime, weights and support-tape locations in [qualification.log](qualification.log).

**CPU qualification.** The exact pinned motor `run_episode` and the new paired driver ran all 16 moved-view cells × six interventions, with two five-step replans per arm. The fixture substituted a deterministic policy, simulator and observation packer. Three negative controls all failed as intended when the full reset, opening proposal or terminal validity was corrupted. These 96 fixture trajectories exercise the evaluation code. Their reward is synthetic. [Fixture receipt](fixture.json)

**Decision.** Advance pose-compensated SymMeta to the prepared paired native retry when calibrated extrinsics are available. The camera affects the support reader alone in this comparison. A different experiment is needed for an unknown camera pose or a changed motor view. Native rescues over raw SymMeta on the same starts, alongside oracle headroom, would justify investigating task transfer further. The CPU score alone cannot establish this.

## Reproduction

From the repository root on the lane commit, run the following commands. The local run used Python 3.12.16 and NumPy 2.5.4. The CPU fixture reads the motor episode implementation from Git object `ff2ddf0ab03e6fbe243979556d8d8f2ebb9bc29c`, which requires that commit in the checkout's history.

```bash
run=outputs/frozen-vla-factory-20261123-r1/native
uv run --no-sync python -B "$run/orbit.py" /tmp/orbit-screen.json
uv run --no-sync python -B "$run/fixture.py" /tmp/orbit-fixture.json
cmp /tmp/orbit-screen.json "$run/screen.json"
cmp /tmp/orbit-fixture.json "$run/fixture.json"
bash "$run/launch.sh" qualify
```

The offline scorer checks the compressed JSON and action-array digests. Its [metrics](screen.json) include every support/view row, both per-task effects, all controls and sign-specific comparisons. The input revision is `757a44cd4374833cad790852df8b87ddf412e638`. [Orbit inputs and reader provenance](../../symmeta-camera-command-20261123-r1/RESULT.md) · [π₀.₅ tape provenance](../../libero-xemb-friction-dev-20261119-r1/PLAN.md)

For the native run, stage the clean deployment checkout on the project's PFS. Use the authenticated Beijing console's Distributed Training **Pro6K → 算法Z组** queue (`aihcq-rh0wwk9muei4`), checking current admission and remaining quota before submitting. Request one worker with one RTX PRO 6000D, the validated PyTorch 2.7.0 CUDA 12.8 image, read-write `/mowenjie`, graphics capability, a two-hour scheduler maximum and automatic retries disabled. In this exclusive allocation run `bash "$run/launch.sh" run "$DEPLOYMENT_SHA" orbit01` with a fresh attempt name and the verified literal deployment SHA. This would consume at most 2 of the lane's 12 GPU-hours. [Queue and image provenance](../../symmeta-original-pro-20261123-r1/RUNTIME.md)

The launcher checks runtime Git pins, motor manifest and weight file hashes, the Baige GPU probe, and the archived support-tape hashes. It pairs the six arms at release `e5500c69` BASE, query start 14, on the same native reset, prompt, horizon, noise and opening proposal. The frozen motor retains its normal query camera. Each verified cell produces both the scores and native trajectories' reward/termination records on PFS. A complete run emits paired success differences by task, wins/losses/ties, and absolute successes over 16 cells per arm.
