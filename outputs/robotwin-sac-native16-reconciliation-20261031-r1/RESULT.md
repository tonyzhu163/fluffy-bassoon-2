# SAC endpoint and the separate historical matched native16 bank

Follow-up to the immutable SAC100 audit, 2026-10-31. **A historical matched
native case-bank comparison is supported.** This is different from the 16
unpaired stage-native probes discussed in the earlier endpoint receipt.

Root commit `4899a9a63472620837816a3cc49585d9a3974cf9` records an existing native
bank with 6/8 bottle and 0/8 stapler successes. Its pairing proof matches all
16 native requested/resolved seeds, raw initial observations and public features
to the final full-PPO and separately trained reset cases. The SAC100 audit
matches both SAC conditions to those same full/reset cases, including raw
RGB/wrist/proprio/feature descriptors and language. This follow-up checks that
chain for all 16 scenes and both SAC conditions; all fields agree, including
bottle episode 5's requested `22265018` / resolved `22265019` fallback.

Nine existing native-proof input/result/receipt file hashes were checked and
matched. The committed root proof/note, prior endpoint receipts and saved SAC
contract are separately hash-bound in `receipt.json`. No raw corpus was rescanned.

Motor fingerprint, capture qualification, model/norm-stat identity,
RLinf/RoboTwin versions, H32 / 32×32 noise dimensions, 2,062 features, action
transform and `demo_clean` configuration agree. The qualified action-count
contract is the same native `valid_action_mask` convention. This supports a
**cross-run matched-initial-case reference**, with these limits:

- Native wrapper `2fd2096…` / backend SHA `eacc64b4…`, PPO executor `57016ba…`,
  and SAC executor `e825821…` are distinct. Whole-executor equivalence is not
  asserted.
- Execution dates and stochastic samples differ. Native uses its original
  unclipped standard Gaussian. These are not common random numbers or one
  shared motor process.
- The native results cover all eight episodes per task. Original support/query
  labels remain unchanged; reusing case outcomes creates no new one-shot or
  separate frozen-query claim.

| Policy on these same initial cases | Bottle /8 | Stapler /8 | Total /16 |
| --- | ---: | ---: | ---: |
| Historical native Gaussian | 6 | 0 | 6 |
| Final frozen SAC | 6 | 2 | 8 |
| Final online SAC | 5 | 0 | 5 |

Against native, frozen SAC has three case wins and one loss: bottle exchanges
one win/one loss; stapler adds two wins. Online SAC has zero wins and one loss.
These paired case outcomes remain a small cross-run development comparison,
not an independent replication or statistically established performance gain.

The source-stage native before/after probes use the separate `21161030` bank
and remain unpaired with the learned targets. Both statements can hold: those
probes are unpaired, while the older `22261030 + 997i` native16 bank is matched.
The earlier endpoint report is preserved; this note adds the previously
unincorporated historical reference rather than relabeling the stage probes.

The original native panel's 5,051 actions stay charged once to that completed
panel. This reconciliation adds zero robot actions, GPU time or SGD and creates
no resource reservation, raw-data scan, model/test execution, allocation or
source-collection gate.
