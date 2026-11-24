# Embodiment prefix: result

**No native reward and no pi0.5 forward pass were run. Nothing here is task success.** [Plan](PLAN.md)

## What exists

| Evidence | What it shows |
| --- | --- |
| CPU fixture, 7 tests | On a tiny frozen PyTorch stand-in, `fit.py` lowers the flow loss for both arms while every `state_dict` leaf stays byte-identical and no base parameter holds a gradient. A base leaf moved on purpose raises `FrozenBaseChanged`. A zero shift reproduces the unmodified output. Through the shared harness fixture, the three adapters tag the prompt, the tag is removed before the policy sees it, commands change, and untagged calls equal raw. A missing fit is an error. |
| Archived command tapes, CPU ([tape-stats.json](tape-stats.json)) | Five replays give 171 replan-point rows from **two** distinct IIWA tapes. With layout and noise shared, the Kinova3 and IIWA opening five controls differ by 0.03–0.10 in L2 against a norm of 2.4–2.9; IIWA 34 against IIWA 35 differ by 0.79. |
| Offline pi0.5 forward pass | None. |
| Native reward | None. |

The harness directory's 25 tests still pass, and `panel.py --list` resolves `prefix-shift`, `prefix-tokens`, `prefix-random`.

## Parameters

| Arm | Trainable scalars at width 2,048 |
| --- | --- |
| `prefix-shift`: one vector added to every valid prompt/state token embedding | 2,048 |
| `prefix-tokens`: 8 tokens appended to the prefix | 16,384 |
| `prefix-random`: seeded direction with the fitted shift's norm | 0 fitted |

Base leaf count and digest are written to `fit.json` by the GPU job.

## Not run, and not verified against OpenPI 215abfb

`openpi_engine.py`, `collect.py`, `train.py` and `run.sh` compile or parse here but were never executed: this VM has no JAX, OpenPI, simulator or GPU. Specific unverified assumptions:

- `Policy._input_transform` accepts an `actions` key and normalises donor commands the way the checkpoint was trained. If `pi05_libero` applies a delta-action transform, the targets are wrong.
- `embed_prefix` places the prompt tokens last, and patching it on the class during a trace reaches `compute_loss` and `sample_actions`.
- `Policy.infer` calls `self._sample_actions(rng, observation, **kwargs)`.
- A 4090 holds one-row gradients through the full model for both arms.

`train.py` checks two of these on the host: an untagged call is byte-identical before and after fitting, and the zero-shift difference from stock is recorded.

## Differences from the plan

- Rows are taken at five-control replan points (171 rows), not at every control, because the simulator interface is only qualified for five-control chunks.
- Branched from the harness commit `5fd6c426`, which sits on top of the factory branch.

## What the archived tapes imply

A fit on two tapes can lower loss by memorising them. The held-out replay 31←35 shares tape 35 with the fit set, so it separates layouts, not tapes. The small opening gap between bodies means any effect must come from later replans. The random control and the Panda panel are what separate an embodiment effect from a generic perturbation.

## Launch

```
bash outputs/frozen-shift-prefix-20261124-r1/run.sh <40-hex pin> <attempt>
bash outputs/frozen-shift-prefix-20261124-r1/run.sh <pin> <attempt> <fit dir> --starts locked --adapters <one arm>
```

The first form collects, fits both arms, scores the held-out replay, then runs the development and Panda panels. The second runs the locked starts once, for an arm chosen from the first.
