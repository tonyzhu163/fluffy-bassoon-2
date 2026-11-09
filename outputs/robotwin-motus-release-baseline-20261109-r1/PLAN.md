# Released π0.5 baseline first

User scope, 2026-11-09: reproduce the released Motus π0.5 baseline before
anything else; then obtain PPO and GRPO baselines. Start with
`place_phone_stand`, as previously agreed. The paused custom controller run
and its failed competence gate do not establish this baseline.

## Reference and claim boundary

- [Motus v2, §5.2 and Table 14](https://arxiv.org/html/2512.13030v2): π0.5
  phone stand 49% clean / 53% randomized, 100 trials per setting. Across all
  50 tasks: 42.98% / 43.84%. Motus itself is a different model.
- [Released weights](https://huggingface.co/motus-robotics/pi0.5_robotwin2),
  revision `effc6e1a99bd4d0edf9447e8ba313def67649cfe`.
  Model SHA256 `de8e03a445f3e0af417a9d61a903d342fca5b6a5b4fdc6f20445e14ad25692eb`;
  norm SHA256 `1dce5ceb9a98eaa6ae587795e0e8977dc3ce046a7ba0a5ec07858ea896ed1adb`.
- Its metadata records global_step=40000, π0.5, H32, model action dimension32,
  token limit200, bfloat16, gemma_2b/gemma_300m, delta joints, adapt_to_pi=True.
  Metadata SHA256 `ef52d518167a652b74cc1c4d3a49fc18f758c286fc0ee8485d33a3caf3bd99ee`.
  Inspected with pickletools, without executing pickle imports.
- Motus's published inference adapter targets its larger architecture, not
  this OpenPI checkpoint. [Issue45](https://github.com/thu-ml/Motus/issues/45)
  asks for the missing exact evaluator/fork and remains unanswered at review.
  This is a documented reconstruction, not a maintainer-confirmed reproduction.
- The saved metadata additionally references an unreleased
  `add_state_from_actions` training transform. The model card instead describes
  a state input. Evaluation uses measured native joint state. Record this
  unresolved training provenance; do not invent the missing function.

## Execution

Use the RoboTwin source available at the paper's release date, commit
`fa8d97b0f33bb894eae16df5a8175490f5b977a7`, including the original native
`Base_Task.take_action`, without the RLinf sparse/chunk controller or our
waypoint fallback. Its [π0.5 adapter](https://github.com/RoboTwin-Platform/RoboTwin/blob/fa8d97b0f33bb894eae16df5a8175490f5b977a7/policy/pi05/deploy_policy.py)
maps head/left/right RGB to matching ALOHA cameras, CHW uint8, measured
14-dimensional joint state and unseen task instruction. It executes the
returned chunk (up to50 requested; the released checkpoint returns32), reading
observations after every target. Replan after32. Native phone limit400.

Reconstruct the card's inference data configuration in the installed
`rlinf-openpi==0.1.1` reference OpenPI package, using its direct
`create_trained_policy` / `Policy.infer`, not the RLinf policy subclass.
Use the checkpoint's own norm stats, quantile normalization as selected by
OpenPI for π0.5, native ALOHA conversions and delta inverse. K10 is the
OpenPI default, not a paper-confirmed π0.5 evaluator setting. Eager execution
avoids compilation setup; no custom sampler or policy-side noise injection.
Pin/hash the actual package source. Bind rendering and inference to the same
allocated GPU; retain scheduler visibility.

Match native evaluation seed0: candidate seeds begin100000; expert play_once
must plan and succeed before policy evaluation; count100 accepted episodes
per clean/randomized setting. Use native unseen language generation and
per-task limits. Record rejected candidates and exceptions. Policy rollout
errors abort the run rather than becoming silent exclusions.

First complete a 2+2 episode smoke, then100+100 independently under the same
frozen contract. Smoke qualifies execution only, not success rate. Retain
initial/final camera panels, all predicted action chunks and initial states,
per-episode seeds/instructions/results, controller/runtime hashes, and GPU
probe. Use BOS outputs and job-local NVMe. Do not tune against these results.

## Sequence after SFT

Only after this baseline is measured and its runtime qualified, compare the
RLinf SFT execution against it, resolve material execution differences, and
then launch separate PPO and GRPO baselines from the same released weights.
Use identical evaluation cases/settings and report per-task SFT/PPO/GRPO
results plus uncertainty and compute. No RL training is authorized to bypass
the SFT-first ordering. Broader six-task expansion follows phone stand and
must fit the existing cap; a phone result is not a50-task reproduction.

## Budget

Continue the previously approved100GPU-hour Epoch2 cap. Already closed:
918GPU-s=.255h. Reserve at most1GPU-hour for the first native smoke; inspect
its actual rate before reserving the full evaluation. No new100h reset.
Update the existing single LATEST-BUDGET ledger and our coordination row.
