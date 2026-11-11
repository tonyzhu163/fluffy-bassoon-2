# Release inspection and qualification status

2026-11-11, static inspection on macOS. No model computation or simulator
execution was performed locally. Source checkout:
`/tmp/libero-gen-bpp-20261111`, BPP
`58125e682acceefef3d60c071212dc483a4db1c9`.

The official Spatial Combination task configuration trains on
`libero_spatial_selected_combinations_inverse_view` and evaluates unseen
combinations in `libero_spatial_selected_combinations_view`. The static
inventory finds 183 and 10 BDDL files respectively, and 53 groups with
matching physical descriptions but different goals. These are BDDL counts,
not verified dataset or episode counts. The official documentation warns
that some generated tasks have no successful HDF5 demonstrations.

The source evidence is hashed by `audit_release.py` into `RELEASE-AUDIT.json`:

| Source | Relevant behavior |
| --- | --- |
| `config/libero_policy_dunetp.yaml:12` | Enables prompting; language and goal image ignored. |
| `config/task/libero_defaults.yaml:12` | Prompt length 768, at most one full demo, 20-action prompt chunks; observation keys contain no reward. |
| `env_runner/libero_image_runner.py:416` | Loads prompts from `LiberoReplayImageDataset`. |
| `env_runner/libero_image_runner.py:561` | Supplies prompt before rollout. |
| `env_runner/libero_image_runner.py:703` | Predicts actions from observation dictionary. |
| `env_runner/libero_image_runner.py:716` | Receives reward after stepping environment; success triggers done. |
| `policy/diffusion_unet_policy.py:244` | Normalizes and passes demonstration prompt to encoder. |
| `model/prompt/prompt_obs_encoder.py:842` | Stores/caches prompt features, not a reward-history interface. |

The source supports selecting a demonstration for the frozen motor. It does
not establish that a released policy can learn from appended failure or
reward tokens. The learned history reader in PLAN.md is an additional
component trained offline and then frozen.

## Baige preparation

The existing idle Jupyter terminal was reused; no new browser tab or GPU job
was created. At 07:17:20 UTC, project shared storage had 542126374912 free
bytes (~504.9 GiB), and an actual write/read probe passed. No `quota` command
was available, so a separate quota limit was not independently established.
`uv` was absent from PATH and three conventional locations; that is not an
exhaustive search. Existing cache entries were legacy LIBERO runtimes; a
compatible BPP runtime was not found or qualified.

An isolated source preparation was started at
`/mowenjie/fluffy-bassoon-2/cache/bpp-58125e6-20261111-r1`. The main public
repository clone completed and the exact commit assertion passed. The
subsequent LIBERO submodule clone did not produce the completion marker
within its 180-second timeout; the preparation process was absent at the
last check. Its exit code was not captured, so treat the submodule as
**unqualified**, not as successfully installed.

Remote evidence is in
`/mowenjie/fluffy-bassoon-2/outputs/libero-gen-history-pilot-20261111-r1/`
(`ENVIRONMENT-PREFLIGHT.json`, `prepare-source.sh`, `prepare-source.log`,
`write-probe.txt`). The remote values above were read from the console;
the raw remote files have not yet been copied back. Browser ownership was
returned to the parent for toy qualification after observing an idle prompt.

Next runnable stage is to finish source/dependency preparation, verify
checkpoint and prompt-data files, then run the bounded motor qualification.
No checkpoint has been loaded, no native episode completed, and no learned
history model has been trained. Additional GPU usage: **0 of 5 GPU-hours**.

## Minimal asset and launch preparation

Official Hub API metadata confirms the exact two prompt files exist.
`ASSET-MANIFEST.json` pins model revision
`56da72e7c27602b41608984d544ca05b638833f4` and dataset revision
`6cfd76524c02faf775263aaf8ba383d62ffd6db5`, with byte sizes and LFS SHA256s.
The checkpoint is 6,915,257,998 bytes; selected HDF5 files are 589,630,943 and
946,177,276 bytes. Total selected download is 8,451,066,217 bytes. Full data
is about 159 GB and is unnecessary for this qualification.

`prepare_runtime.sh` uses isolated uv 0.13.7 / Python 3.10.20 and the
official BPP CUDA-12.8 torch 2.8.0 option. `runtime-requirements.txt` retains
the relevant upstream pins, including eagerly imported Open3D, but excludes
robot-device and DrawAnything packages. Legacy gym 0.21 is installed with
upstream's pip 24 / setuptools 65.5 / wheel 0.38 toolchain. The documented
robomimic fork is pinned to `f4357ed548b656800636ee459c7a2f31f95b5a08`.
This dependency subset has only been statically inspected, not imported yet.
The uv pin was updated after checking the current official PyPI metadata;
the older 0.9.10 bootstrap predates the requested Python 3.10.20 release.

`qualify_motor.py` loads the checkpoint's own model configuration and strict
saved `model` state, freezes every parameter, and uses the original LIBERO
runner with an observation/action/reward recorder. Setting pretrained
initialization off avoids downloading redundant CLIP weights; strict loading
must still supply every saved weight. Both candidates receive a copied,
isolated common initialization bank because upstream init-file byte hashes
differ. Initial observation equality and unchanged parameter version
counters are required. Videos and native reward traces are retained.

`launch_motor.sh` requests no resources itself: it is for one approved
allocated GPU with a 60-minute scheduler cap, runs the repository GPU probe,
then bounds the first one-seed-per-candidate check to 55 process minutes.
The probe uses `--purpose evaluation` and the pinned project's `src` on
`PYTHONPATH`. Missing CUDA visibility is filled only after a single NVML
UUID exactly matches the scheduler's `NVIDIA_VISIBLE_DEVICES`; existing
visibility is preserved. The outer clock starts before the probe so probe
failures are included in the process receipt.
No launcher has run. Python compilation, shell syntax, and `git diff --check`
pass; those are static checks, not runtime qualification.

## Preparation update, 15:59 Shanghai

All three selected assets (8,451,066,217 bytes) finished downloading and
matched their manifest SHA256 digests. This was observed in remote
`asset-prefetch01.log`; raw download receipts are pending retrieval. The
checkpoint has not been loaded.

The private project Git store was repacked into an independent object
store; its connectivity check passed and its alternates file was removed
from use. Commit `27fe941012964f2ec922aae17f293ad0898032ed` was deployed
through a verified Git bundle into the new detached source directory
`/mowenjie/fluffy-bassoon-2/sources/libero-gen-preflight-27fe9410`.
The older source directory remains unchanged.

The upstream LIBERO submodule clone reached its 900-second bound before
completion. The old preparation process is absent and the source remains
unqualified. A separate CPU dependency installation started at 07:59:20
UTC using the dependency lines from the committed installer, isolated in
this pilot's environment (`dependency-prepare02.log`, PID 68530). An exact
upstream Git bundle fallback is being prepared. No runtime import check,
simulator episode, or GPU job has run; additional usage remains 0 / 5
GPU-hours.
