# Native collector and scorer interface

Import the sibling `model.py`. All functions run on CPU with NumPy. A step counts native actions already issued. Make the fork after action 60 and before action 61.

| Function | Contract |
| --- | --- |
| `features(obs, raw_chunk, step)` | Return a flat float32 array. Pass the R2 public observation with `rgb`, `wrist` (256×256×3 uint8), `state` (8 float32), and `prompt` (string, ignored). Pass the full frozen proposal (5×7 float32 in [-1,1]) for this observation and the elapsed integer step. |
| `fit(pairs, out_dir)` | Fit from an iterable of training pair dictionaries (schema below), write `model.npz` and `fit.json`, and return a fitted policy. |
| `load(out_dir)` | Load the fitted policy from the directory containing `model.npz`. |
| `policy.score(obs, raw_chunk, step=60)` | Return one Python float at the fork. Enable correction for the entire suffix iff `score > model.GATE_THRESHOLD`. Score once, then latch. |
| `policy.apply(obs, raw_chunk, step, enabled)` | Return the 5×7 float32 chunk. Pass false before the fork. Pass true on every suffix replan for the always-correct arm. For the gated arm pass the latched boolean. Both use exactly this action function. OFF returns the original raw proposal. |

Each training pair dictionary has these keys. For a raw episode which terminates at or before step 60, use the early-terminal representation below.

```python
{
    "split": "train", "arm": "Panda", "layout_id": layout_id,
    "raw_success": bool(raw_native_success),
    "teacher_success": bool(teacher_native_success),
    "fork_x": features(shared_obs, shared_raw, 60),
    "suffix_x": xs,                  # (N, D) features on teacher trajectory
    "teacher_actions": actions,      # (N, 5, 7) issued native actions
    "valid_lengths": lengths,        # (N,) integers 1..5
}
```

The collector verifies that the fork matches across continuations, including its raw proposal. Record every actionable fork, including raw wins and ties. For a successful teacher, append one feature row at each teacher replan starting at 60, using the frozen motor on the current teacher observation. The target chunk contains the actual next teacher actions. Right-pad the last target with zeros after native termination. For an unsuccessful teacher, the three suffix fields may be empty sequences. For an episode that terminated in the common prefix, pass `fork_x=None`, identical terminal outcomes and three empty sequences. This records exposure without introducing a fabricated fork. Each arm/layout appears once.

`python model.py selftest` runs a CPU-only synthetic numerical check on Linux. The native collector owns branch evidence and calls the fitting function after training collection. The scorer loads the same saved fit for both learned conditions.

## Fixed fit

The 140 public features concatenate row-major 4×4 RGB cell means for `rgb` then `wrist` in [0,1], the eight unscaled state channels, the entire unscaled 5×7 raw chunk in time-major order and the elapsed control count. Ignore the prompt text. Training arm and layout IDs determine weights and exposure records only. `RUN-API.md` specifies the native manifest and loader. The fitter also accepts its optional `layout_seed`, `decision_reached` and `base_chunks` fields.

Fit weighted linear ridge with an unpenalized intercept and coefficient penalty **1.0** on the weighted-mean squared-error scale. Normalize each feature using the same training weights as its head. Floor standard deviations at 0.04 for RGB, 0.02 for state, 0.05 for raw actions and 5 elapsed controls. Fit seven outputs separately for each of the five chunk offsets, using only issued actions on successful teacher suffixes. Give each represented arm equal weight, then each eligible layout within arm equal weight, then each valid example within layout equal weight. For each offset, predict teacher action minus raw. At deployment bound the additive residual per axis to ±(0.50, 0.50, 0.50, 0.25, 0.25, 0.25, 2.00), then clip the corrected chunk to [-1,1]. The jaw cap permits switching between both endpoints.

Fit the scalar gate by the same weighted regression on the signed difference of native outcomes at all actionable training forks. Switch on only for a predicted advantage strictly above **0.10**. The threshold, spatial resolution, caps and ridge strength are fixed before fitting. All-on and all-off results are possible. A head with zero eligible targets returns zero. This gives a closed switch when there are no actionable forks, and raw actions at offsets with no successful teacher examples.

The fit report records training identities and input digests, eligible counts and weights by arm/layout, signed labels, gate scores and switch counts, in-sample imitation errors, and the fitted normalizers. The native evaluation measures whether the fixed learned correction realizes the advantage predicted from the privileged teacher.
