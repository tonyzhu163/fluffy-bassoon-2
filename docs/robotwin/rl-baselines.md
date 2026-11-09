# RoboTwin benchmark protocol

## Shared task split

Use the [machine-readable split and reference settings](../../benchmarks/robotwin/configs/robotwin_vlrla.json). These tasks are held out from adaptation training, not necessarily from the base motor’s SFT.

| Source histories | Held-out transfer |
| --- | --- |
| `handover_mic` | `adjust_bottle` |
| `pick_diverse_bottles` | `move_stapler_pad` |
| `place_a2b_left` | `place_a2b_right` |
| `place_object_stand` | `place_dual_shoes` |
| `place_phone_stand` | `place_empty_cup` |
| `scan_object` | `stamp_seal` |

RLinf PPO provides a source-learning baseline and chronological data; GRPO is a separate online comparator. The [VLRLA protocol review](../literature/vlrla-protocol.md) describes the recurrent KV-bridge reference. Record whether a comparison uses released artifacts or a reconstruction. The task-specific `adjust_bottle` SFT release is a runtime check, not a common initializer for the six source tasks.

## Runtime setup

[Setup script](../../benchmarks/robotwin/setup.sh) takes seven arguments, in order: uv executable, base Python, fresh environment path, fresh output path, RLinf root, RoboTwin root, model directory. It checks the qualified RLinf `c605b9d5` / RoboTwin `dca9ec68` source pins and imports the twelve tasks into an isolated overlay. More recent run-specific pins require their own validation. [Configs and launchers](../../benchmarks/robotwin/) and [native runtime](../../src/fluffy_bassoon/robotwin/) use the ALOHA-AgileX embodiment.

On an allocated Linux CPU, the [controller parity test](../../tests/test_robotwin_native_execution.py) accepts `ROBOTWIN_REFERENCE_ROOT` (unmodified) and `ROBOTWIN_ACCOUNTING_ROOT` (matching patched source). For a rollout, independently verify renderer and inference device binding, three nonblank camera views, finite commands, and advancing native steps. Use matched scenes and motor sampling across paired policies. Record requested commands, delivered targets, physics steps and time; they can differ under different controllers.

## Motor and history contracts

The [Sidney release verification](../../outputs/robotwin-sft-initializer-selection-20261109-r1/RESULT.md) distinguishes its original broad SFT bundle from the RLinf conversion. Sidney predicts H50 absolute joints; Motus uses H32 delta joints and a distinct transform/controller. Keep each motor, normalization, horizon, sampler, scene bank and reward fixed within a comparison.

History contains executed actions, observable feedback and boundaries. A source checkpoint alone is not evidence of an improving policy. Evaluate reward-blind, episode-reset and full-history conditions on the learner’s own attempts. The [RoboTwin findings](../FINDINGS.md#robotwin) separate native baseline performance from learner qualification.
