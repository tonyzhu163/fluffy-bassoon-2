# Harness lane result

[Lane plan](HARNESS.md) · [Program plan](PLAN.md). Evidence here is **CPU fixture** plus one **archived-frame** packing check. No policy forward pass, no native reward, no Baige job. Nothing below is task success.

## Delivered

| File | Content |
| --- | --- |
| [harness.py](harness.py) | `Adapter` with the seven contract hooks, `collect_support`, `rollout(sim, policy, support, case, adapter)` over `ur5e_motor.run_episode` |
| [panel.py](panel.py) | Registered starts × discovered adapters with `raw` and `repeat`, reset and opening-proposal verification, per-start files, `summary.json` |
| [fixture.py](fixture.py) | Deterministic simulator, policy and packer that lanes can import. Synthetic rewards |
| [test_harness.py](test_harness.py) | 25 CPU checks |
| [run.sh](run.sh) | Pin check, adapter resolution, GPU probe, one RTX 4090, four-hour cap (`PANEL_SECONDS`) |

`uv run --no-sync pytest -q -W error outputs/frozen-vla-proposal-shift-20261124-r1` gives 25 passed on this VM.

## What the CPU fixture shows

- The default adapter sends the same commands, prompts, policy inputs and noise as the plain motor loop, with the same number of policy calls, for a full horizon, a horizon that is not a multiple of five and an early stop inside a window.
- Against the untreated first replan, an image change moves only the image digest and what follows from the proposal, a state change only the state digest, a prompt change only the prompt, extra noise only the noise digests, count and chosen index, and a command change only the consumed window. No hook moved another hook's record.
- Candidate noise digests, proposal digests, candidate count and chosen index match what the adapter drew and what the policy was called with. The candidate stream repeats for a start.
- Hooks received only the two camera images and `robot0_joint_pos`, `robot0_eef_pos`, `robot0_eef_quat_xyzw`, `robot0_gripper_qpos`. Writes into those copies reached neither the policy nor the simulator.
- On the archived Kinova3 layout 14 reset frame, the public view packs to the same policy input as the full observation through the real packer. The image resize is stubbed because `openpi_client` is cluster-only.
- The panel skips lanes without `adapter.py`, lists a lane whose import fails, records a hook fault per start and drops it from that adapter's denominator, and stops on an unmatched reset, opening proposal or incomplete reward tape.

## Choices beyond the contract

- `reset` receives a public reply: `observation`, `language_instruction`, `action_schema`, `arm`, `joint_count`, `max_steps`. Fingerprints, layout id and reward are withheld. `support` is the selector record plus the burst `commands` and four proprioceptive `frames`.
- The untreated opening proposal is always computed and stored as `untreated_opening_proposal_sha256`. It costs one extra policy call per episode when the adapter changes the first query. The panel matches branches on it.
- Equal queries inside one replan are answered by one policy call. This assumes the policy is deterministic for equal inputs and noise within a process, the same assumption the opening-proposal match tests.
- Locked starts 40–47 refuse to run without `--adapters`.
- The summary reports each adapter against `raw` and against `repeat` (delta and paired layout bootstrap interval for both), rescues, losses, controls, candidate count and the first control at which commands leave the untreated branch. It applies no pass threshold.

## Launch

```sh
bash outputs/frozen-vla-proposal-shift-20261124-r1/run.sh "$(git rev-parse HEAD)" native01 --starts development
```

Add `--adapters a,b` to restrict. `python panel.py <out> --list` resolves starts and adapters on CPU.

## Not done

- No native episode. `run.sh`, `load_frozen_pi05` and `UR5eSimulator` paths were not executed; `run.sh` passed `bash -n` only.
- No lane adapter existed when this was pushed, so discovery was tested on temporary lane directories.
- No resume. A timed-out job keeps completed starts and the summary so far; rerun the rest with `--starts` under a new attempt.
- Compute was not measured. Each candidate is one policy call per replan, so K-candidate adapters cost about K times an untreated episode's policy time.
- Passing `arm` and `joint_count` to `reset` goes beyond observations and commands. Remove them in `Episode.call` if the program wants the strict reading.
