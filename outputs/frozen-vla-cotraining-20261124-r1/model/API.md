# Training and native evaluation interface

Add this directory and the sibling `data/` directory to `PYTHONPATH` in the qualified Baige OpenPI JAX environment. Use the native simulator's existing `ur5e_motor.to_openpi_observation(raw, prompt, canonicalize_quat=False)`. The checkpoint is stock `pi05_libero` in `/mowenjie/fluffy-bassoon/inputs/pi05-libero-openpi-20261027-r1`, OpenPI revision `215abfb217dbac7d5f1273282331b9b1866c0479`.

```python
from cotrain_policy import LockedPolicy
from source_data import PublicHistory

motor = LockedPolicy.load(fit_dir)   # one stock model shared by all conditions
history = PublicHistory()
history.reset()                     # reset on every trial, after excluded warmup
last_replan = history.vector().copy()
# At each replan:
current = history.vector().copy()
used = current                      # aligned
# zero control: used = np.zeros_like(current)
# lagged control: used = last_replan (one complete replan behind, initially zero)
actions = motor.infer(item, noise=noise, condition="joint", history=used)["actions"]
last_replan = current
# Append each executed control with history.append(issued_action, previous_raw, next_raw).
```

`item` is the already converted OpenPI dictionary, `noise` is the paired float32 [10,32] array, `history` is the float32 [52] output of the data API and `actions` is the stock-decoded [10,7] chunk. Valid condition strings are `original`, `policy_only`, `adapter_only`, and `joint`. Zero and lagged use the identical joint checkpoint. The caller owns the causal buffer and lag. Inference has no optimizer or mutable history. `motor.digests()` returns a JSON-compatible mapping of verified artifact digests and fixed normalization for before/after scoring.

Training consumes `SourceDataset(data_root, split="train" | "heldout")` exactly as specified in [the source contract](../data/API.md). The fit directory contains `heads.npz` plus `fit.json`, written last. Pass this directory as `fit_dir`. The loader verifies fit artifacts and base identity. Runtime instructions and receipts are in [the run notes](README.md).

`motor.parity(item, noise=noise)` compares the observed decoded actions of stock and the independently materialized no-op projection with the same native sampler and noise. It returns both digests and `max_abs`. The previous fused zero-factor sampler is measured separately in `fused_zero_sha256`, `fused_zero_max_abs` and `fused_zero_bitwise`. `motor.stock` is the ordinary OpenPI Policy. `motor.digests()` hashes every resident stock NNX parameter and every resident head array, the current on-disk heads, fit manifest and normalization. Call at job boundaries.
