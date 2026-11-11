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
