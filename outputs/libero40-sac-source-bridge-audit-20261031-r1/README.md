Verified2026-10-31. Both fixed source bridge streams finished exit0, with `collection_qualified=true`, complete K4 cases, unchanged frozen motor bytes and final paired AD/DPT exports. Execution source: `cf49bfac65a98cc7c4cef7fb35df4a6a18e20c0b`. No full24h top-up was launched.

| Source / Baige job | New episodes | New K4 cases | New chunks | New SAC updates | New actions | GPU seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|Spatial00 /`job-zh3wvn64ebfd`|100|25|862|17,240|9,121|1,078|
|Object00 /`job-xli5uonwbiph`|84|21|1,193|23,860|12,447|1,134|
|Total|184|46|2,055|41,100|21,568|2,212|

Each stream remained within its original1,800-second worker ceiling. Every new episode succeeded. Each new raw chunk is actually `sac_online`; saved clocks advance by exactly20 updates per actual chunk, with finite scalar update metrics. Actual new settled-state hashes cover25 Spatial cases and21 Object cases; seeded-case counts are distinguished from nominal-state coverage.

Spatial inherited the terminal migration qualification exactly once and now holds220 episodes /55 cases /1,927 chunks /28,620 updates /20,465 cumulative actions. Object inherited its original48 episodes and now holds132 episodes /33 cases /1,987 chunks /29,940 updates /20,677 actions. Both stop because their remaining1,335 and1,123 actions cannot fund another worst-case1,640-action K4; they should be retained without a cold relaunch under the21,800-action caps.

Each actual allocation passed its GPU probe and39 focused Linux tests, with2 JAX-environment skips. Saved-data verification passed32 checks per stream: all184 new raw hashes and actual episode boundaries/stages/clocks; finite bounded executed noises; full recovery/checkpoint and export hashes; complete saved optimizer/replay/sampler/global/motor RNG states; final publication learning bytes equal latest recovery; unchanged motor; finite exported labels; AD labels equal actual executed noises; and full cumulative teacher-cost bindings.

Latest full recoveries: Spatial `895bf0ff083ef7839e820cd6fbf00d79b2d33d903ac8be784ec14893f487f45d`, Object `1b5ab572cb46d1a143b6fc84ab4f6054994a256ac6a5ef82dcbf47a55d17adb8`. Final source actor checkpoints: Spatial `b95e004854ae641910d82a71f5fabeb6846e77d9472513f8dfa364dd9d11729c`, Object `1f288d06e87bb502ee8855a77698b8ba2bc0439b9444950f7dcaa6ae5fb7183a`. Export paths/hashes, source ancestry and allocated restore receipts are retained in `receipt.json`.

The full saved final payloads were inspected without constructing a model. Initial real allocated restores passed; no additional independent final model restore was run for these production streams. `audit_saved.py` only reads terminal saved files/arrays and performs no rollout, model inference, training or tests.

The184/184 source successes provide runtime and data-integrity evidence with no success headroom. They do not establish improved teacher competence, sample efficiency, adaptation, transfer or a memory benefit. Final actors remain unvalidated source candidates; only one latest cumulative prefix per history should enter fitting.
