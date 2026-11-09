# Released Motus π0.5: paired prompt/controller diagnostic

Completed 2026-11-09 on Baige Pro6K. All **40 policy episodes on ten paired
expert-feasible clean scenes** completed and passed independent Linux readback.

| Prompt | Native `take_action` | Continuous-chunk `TOPP` |
| --- | ---: | ---: |
| Raw unseen instruction | 1/10 (10%) | 1/10 (10%) |
| `DEFAULT_PROMPT` + instruction | 2/10 (20%) | 1/10 (10%) |

This small panel does not establish a prompt or controller performance gain.
With the prefix, continuous execution lost one native success and gained none;
raw-prompt controller outcomes were identical. Prefix versus raw had two paired
gains/one loss under native execution, and one gain/one loss under continuous
execution. All four descriptive exact McNemar tests have p=1.0. Wilson 95%
intervals are 1.8–40.4% for 1/10 and 5.7–51.0% for 2/10. These are diagnostic
baseline levels for one clean task, not a benchmark reproduction or a basis for
claiming competence recovery. No PPO, GRPO or other training was launched here.

## Protocol and controls

- Released `motus-robotics/pi0.5_robotwin2` checkpoint revision
  `effc6e1a99bd4d0edf9447e8ba313def67649cfe`; exact model/normalizer hashes and
  native source are in the [qualified manifest](REFERENCE-CPU-QUALIFICATION.json).
- Immutable evaluation source `250c15775bfd2245cd03aafa4d7ac0cafe0b354e`, isolated
  `motus-reference-openpi-20261109-r2` runtime, OIDN 2.3.3. Both arms use the same
  native scene/render configuration, H32/K10 policy and 400-waypoint cap.
- Native cuRobo expert screening accepted seeds
  `100000, 100001, 100002, 100005, 100006, 100008, 100010, 100012, 100014, 100015`.
  All six rejected candidates are retained in [analysis](ANALYSIS.json).
  Each accepted scene has one shared unseen instruction across all four arms,
  with common per-decision flow noise.
- All ten initial state vectors matched exactly across arms. Decision-0 chunks
  matched **byte for byte** between same-prompt controller arms. All 478 saved
  decisions/action arrays and 14,699 recorded delivered waypoints reconciled.
  Frozen-parameter checks passed; there were zero denoiser errors. Every one
  of the 484 continuous-arm plans used TOPP; no linear fallback was exercised.
- The separate two-pair qualification passed the same audit. Its eight episodes
  are operational evidence and are excluded from the ten-pair result above.
  [Qualification readback](ep2-01-READBACK.json).

## Execution timing and limits

Median recorded policy-episode wall times, including decision I/O but excluding
expert screening and model loading, were 35.64/10.94 s for raw native/continuous execution
and 36.66/12.24 s for prefixed native/continuous execution. These measure the
complete execution paths, not isolated TOPP or inference speed. The waypoint
budget is matched; physical integration time is part of the controller factor
and is not matched. Native physics-step counts were not recorded, so the
continuous-only counts are not a cross-controller physics-budget comparison.

Only `place_phone_stand/demo_clean` and ten paired scenes were evaluated. The
observed prefix advantage in the initial two-pair qualification did not turn
into a convincing improvement on this diagnostic panel. Broader baseline
qualification and scientific follow-up remain separate decisions.

## Closeout and evidence

| Allocation | Scheduler GPU-seconds, including startup/closeout | Worker GPU-seconds |
| --- | ---: | ---: |
| Two-pair qualification `job-drxzlm19htrf` | 322 | 284 |
| Ten-pair evaluation `job-wgabk7r9kn52` | 1,243 | 1,211 |
| Total | **1,565 (0.43472 GPU-hours)** | 1,495 |

Both scheduler allocations succeeded and are released. Final terminal time was
**2026-11-09 14:27:55 Shanghai**. The run used 0.43472 of its 2 GPU-hour cap;
there are no outstanding reservations or active jobs.

[Independent 40-episode audit](ep10-01-READBACK.json),
[raw result](ep10-01-RESULT.json), [paired reductions and intervals](ANALYSIS.json),
[scheduler closeout](ep10-01-SCHEDULER-CLOSEOUT.json),
[execution and budget](EXECUTION.json), [audit implementation](readback.py).
Full action arrays, initial/final images and logs remain at the immutable BOS
output path recorded in `EXECUTION.json`.
