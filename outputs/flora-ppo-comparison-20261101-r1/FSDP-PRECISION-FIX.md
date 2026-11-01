# FSDP precision compatibility fix

Source preparation after actual R2 failure, before simulation or SGD. The
67-case a3 CPU result remains valid for the code it covered; composition did
not exercise the official entry's `validate_fsdp_cfg` precision requirement.

Only `configs/native_ppo/libero44_flora.yaml` and its focused tests change.
Keep the default BF16 parameter/reduce/buffer policy and all PPO/model/optimizer/
batch settings. Add the c605 named-wrapper schema with individual FP32
boundaries for `RMSNorm`, `Float32Conv2d`, `SigLIPViT` and `ValueHead`, plus the
four pi0.5 action/time projections. Individual wrappers retain automatic BF16
child boundaries, so the `img` parent owns its FP32 positional embedding while
the encoder/head keep their existing policies. No whole-model FP32 switch,
validation bypass, standalone weight conversion or production Python hook.

Primary c605 definitions:

- `rlinf/config.py::validate_fsdp_cfg`: requires named precision rules for
  low-precision OpenPI.
- `rlinf/utils/fsdp_config.py::_parse_wrapper_config` and
  `build_mixed_precision`: named wrappers/policies/rules and FP32 input casting.
- `rlinf/models/embodiment/openpi/pi0.py::to_bfloat16_for_selected_params`,
  `_no_split_modules`, `_no_split_names`: FP32 norm/vision islands and projections.
- `rlinf/models/embodiment/openpi/tasks/rl.py`: FP32 `ValueHead`.
- `rlinf/models/embodiment/openpi/modules/siglip.py`: FP32 stem/position embedding
  before the existing encoder cast. Fully qualified class selectors are resolved
  by `rlinf/hybrid_engines/fsdp/utils.py` against the actual module tree.

Allocated Linux CPU selector, **five affected cases** (four existing composition
cases now call the real validator, plus one regression reproducing refusal of
missing rules and checking the corrected policy):

```sh
"$PPO_PYTHON" -B -m pytest -q "$SOURCE_WORKTREE/tests/test_rlinf_flora_ppo.py" \
  -k 'compose_actual_official_config or real_fsdp_validator' \
  --junitxml="$FRESH_CPU_OUTPUT/fsdp-junit.xml"
```

The full selector contains68 cases; do not rerun the unchanged63 for this fix.
The earlier actual67/JUnit record is in `QUALIFICATION-CARRY.json`. The source
author performed AST/whitespace checks only; these five cases are prepared and
unexecuted here. Sole `finish_assembly` binds the residual CPU allocation inside
the existing360-worker-second phase (97 already charged,263 remaining), then
the diagnosed GPU successor within the same loading ticket. Preserve R2's
88 GPU-seconds and prior costs; no source-created resource grant or retry.

Strict direct GPU loading of the unchanged published weights/model body is
carried as the execution owner's separate result, not a simulator/PPO pass.
Active and ended a3 worktrees remain immutable; this change requires a fresh
source/profile identity and explicit migration.
