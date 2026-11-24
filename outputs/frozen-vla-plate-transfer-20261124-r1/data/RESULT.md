# Matched plate-pushing source data

All 40 native replays reached terminal reward 1. The 20 original source layouts were retained, with 16 for fitting and four for offline validation. All 20 have distinct movable-object placements and full scene fingerprints. Each Panda/Sawyer pair shares the generated sampler state, physical scene, BASE Goal 5 instruction and scripted teacher revision `17338c926b24a8eb0ceb985b1a0db90f745c15ab`. The selection came from the 253 matched generated layouts in the archived source bank before target testing. [Enrollment order](evidence/source-order.json) · [Native paired receipts](evidence/gpu01/enrollment.json) · [Attempt ledger](evidence/gpu01/attempts.jsonl)

| Source data | Layouts | Windows per layout | Panda windows | Sawyer windows |
| --- | ---: | ---: | ---: | ---: |
| Panda-only training | 16 | 12 | 192 | 0 |
| Mixed training | 16 | 12 | 96 | 96 |
| Fixed Panda diagnostic | 4 | 8 | 32 | 0 |
| Fixed Sawyer diagnostic | 4 | 8 | 0 | 32 |

The two training manifests align every row on layout and phase slot. All 96 common Panda examples are byte-identical across variants, including both views, both measured finger joints, tool pose, commands and causal history. Each mixed-source arm contributes four examples at each history length from zero to three, and 80 at length four. Each full action target contains ten commands. The two diagnostic banks are fixed for both fits. Each primary manifest also includes a 32-window held-out split matching its own training arm composition. [Independent readback](evidence/readback01/readback.json) · [Panda manifest](evidence/panda-manifest.json) · [Mixed manifest](evidence/mixed-manifest.json)

Panda replayed 2,099 controls and Sawyer replayed 2,090. Every episode executed the complete archived command stream and ended at native sparse reward 1, with zero earlier rewards. The per-step recordings include the terminal observation, both native RGB streams and both measured fingers. The collection matches source task, layout, window count, phase slots and teacher family. Different bodies can require different action sequences and produce different contacts and camera views. Target transfer is evaluated separately. [Native receipts](evidence/gpu01/enrollment.json) · [Fixed Panda diagnostic](evidence/diagnostic-panda-manifest.json) · [Fixed Sawyer diagnostic](evidence/diagnostic-sawyer-manifest.json)

## Sealed Baige data

The trainer roots are `panda/` and `mixed/` beneath `/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-plate-transfer-20261124-r1/data/dataset-v1/`. Fixed diagnostics use `diagnostic-panda/` and `diagnostic-sawyer/`. Each manifest binds its own HDF5 bytes. The canonical seal SHA-256 is `3e133b69d850318cb20d61549e2f96ce6a7da8548cf05d25af7401ec28047c78`. Full HDF5 replay tapes remain under the same PFS run's `data/gpu01/tapes/`. [Dataset seal](evidence/seal.json) · [Execution and timing](EXECUTION.md)
