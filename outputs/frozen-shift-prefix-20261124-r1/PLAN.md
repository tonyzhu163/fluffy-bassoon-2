# Can a small input prefix, fitted through frozen π₀.₅ on donor replays, rescue Kinova3 layouts it was not fitted on?

Lane of the [proposal-shift program](../frozen-vla-proposal-shift-20261124-r1/PLAN.md). Written November 24 before any result in this directory. Branches from the shared harness at `5fd6c426`.

## Question

IIWA command tapes replayed open loop on Kinova3 earned native reward on layouts 34, 35, 37 and 14. They are target-body observations paired with successful actions, obtained without a target expert. If the frozen policy proposes different behaviour because it observes Kinova3, a few thousand input-side scalars fitted on those pairs through the locked policy may move its proposals toward commands that work on this body. Does such a prefix raise native success on Kinova3 layouts outside the fit, without lowering Panda success?

## Prediction

The fitted prefix lowers the native flow loss on a held-out replay (Kinova3 31 under IIWA tape 35) below the unmodified policy on the same rows and noise. Native rescue on held-out failures 15, 17, 36 and 38 is the open part. The fit data are five replays of two tapes, so the likely failure is a prefix that encodes those two tapes and does not generalise.

## What counts against it

- Held-out replay loss with the prefix is not below the unmodified policy.
- No held-out failure is rescued beyond the untreated repeat's discordance, or losses on 16, 31, 39 match rescues.
- Panda 0–3 success with the prefix applied falls below raw.
- A random prefix of the same norm does as well as the fitted one (the effect is perturbation, not fitting).

No numeric gate is set. The comparisons are raw, repeat and the norm-matched random prefix on paired resets.

## Prior work checked

In this repository:
- [Co-training](../frozen-vla-cotraining-20261124-r1/model/README.md) differentiates `Pi0.compute_loss` on one RTX 4090 (OpenPI `215abfb`, direct per-example gradients, batch 32 by accumulation, 7.4 examples/s for three head gradients). It changes the stock output kernel. This lane changes no stock tensor and its gradient passes through the whole prefix stack, so memory and rate are unmeasured.
- [Donor](../frozen-transfer-donor-20261124-r1/RESULT.md) and [validation](../frozen-transfer-donor-validation-20261124-r1/RESULT.md) give the successful replays: Kinova3 34←IIWA 34, 34←35, 35←35, 37←35, 14←34. Also 31←35 and 41, 43, 46, 47, which are not fitted. Those replays ran on CPU with rendering stubbed, so **no target images are archived**; the fit job must replay again with cameras on. CPU and GPU physics hashes differed in that study, so a replay may not earn reward on the fit host.
- Phase-prompt and state lanes of this program change the text prompt and the state values. This lane changes embeddings, which neither can express.

Outside (cited from memory, not re-read in this session):
- Soft prompt and prefix tuning of frozen language models: Lester et al. 2021; Li and Liang 2021. Visual prompt tuning: Jia et al. 2022.
- Robot policies: TAIL (Liu et al. 2024) compares prefix tuning, adapters and LoRA on a frozen imitation policy for new tasks. X-VLA (Zheng et al. 2025) learns one soft prompt per embodiment jointly with the backbone. HPT (Wang et al. 2024) and CrossFormer (Doshi et al. 2024) use per-embodiment input stems trained with the trunk. OpenVLA and π₀ adapt to new robots by LoRA or full fine-tuning on teleoperated target demonstrations.

Not new: soft prompts, per-embodiment tokens, the flow objective. New here, as far as checked: fitting an embodiment prompt with every policy weight locked, supervised only by another arm's commands that were verified by open-loop reward on the target, and judged by native reward on held-out layouts with source retention.

## Protocol

Two input-side parameterisations on the PaliGemma prefix of π₀.₅ (width 2,048), both read by `Pi0.embed_prefix`:

| Arm | Change | Scalars | Initial value |
| --- | --- | ---: | --- |
| `prefix-shift` | one vector added to every valid prompt/state token embedding | 2,048 | zero, so the first step is the stock policy |
| `prefix-tokens` | 8 extra tokens appended after the prompt tokens | 16,384 | mean valid prompt embedding of the first fit row plus seeded jitter |
| `prefix-random` | `prefix-shift` vector replaced by a seeded random direction of the fitted norm | 0 fitted | control |

1. **Collect** (`collect.py`, GPU host, EGL): replay IIWA tapes 34 and 35 from the committed archive on Kinova3 34, 35, 37, 14 and 31 with cameras on, byte-checking each command. Keep a replay only if it earns native reward on this host. Rows are every control t with ten following commands: public observation at t, commands t…t+9. Reward selects episodes offline; it never reaches the adapter.
2. **Fit** (`fit.py` with `openpi_engine.py`): minimise the pinned `Pi0.compute_loss(train=True)` with respect to the prefix only. Per-example gradients, batch 32 by accumulation, Adam 0.001, seed 20261124, 300 updates per arm, final weights, no selection on rollouts. The optimiser holds only the prefix array. A digest of every stock leaf is taken before and after and must match.
3. **Held-out, offline**: replay 31←35 rows, prefix against zero prefix on the same noise, bootstrap over 10-control blocks. Offline forward pass only.
4. **Native panel** (`adapter.py` through the shared `panel.py`): development starts, then Panda 0–3, then locked 40–47 once if a development failure outside the fit is rescued. Layouts 34, 35, 37, 14 are reported as in-fit. The adapter tags the prompt; a wrapper on `Policy.infer` removes the tag and samples with the prefix. Untagged calls go to the stock method untouched.
5. **CPU here**: `test_prefix.py` runs the same fit loop and adapter on a tiny PyTorch stand-in transformer and checks that no base parameter changes. Tape statistics from the archive. JAX and OpenPI are absent on this VM, so `openpi_engine.py` and `collect.py` are not executed here.

Budget: one RTX 4090 Distributed Training job, under 3 GPU-hours for collection, two fits and the panel.
