# Qualification and Baige handoff

Baige CPU attempt `cpu01` exited zero at pin `22cc4bd4caeb9a07bb2d69c745a1bf6aff522061`. On cores 30/31, eight deterministic fixtures on layout 20 consumed 880 actual native controls. Each matched quartet had identical opening physics, observations, anchor, noise and opening proposal. The support, checkpoint files, released row banks, benchmark definitions and predecessor's verified real-motor dry run passed their identity checks. [Linux launch](evidence/cpu01/launch.txt) · [Asset and support identities](evidence/cpu01/origin.json) · [Qualification](evidence/cpu01/qualified.json)

| Task | Native horizon | Noise bank shape | Source first positive / cut | Switched modes reaching replan 21 | Identical source vs corrected 7D vectors | Largest absolute channel change |
|---|---:|---|---|---:|---:|---:|
| Drawer | 300 | 60 × 10 × 32 | 106 / 105 | 3/3 | 0/105 | 0.108302 |
| Moka | 520 | 104 × 10 × 32 | 107 / 105 | 3/3 | 0/105 | 0.025996 |

The Switch fixture used the raw input and raw proposal on replans 0–20. At each of the six handbacks, the actual motor received the tuned input for the current physical observation. Its state differed from the corresponding raw state. The independent reconstruction found 15 changed XYZ components across the five issued handback controls in each case. Source and Corrected changed all three translation coordinates on every source control, while rotations and grip were numerically identical. Clipping erased zero differences between changed and unchanged channels. A second read of the eight original episodes reproduced the complete qualification. [Original step records](evidence/cpu01/episodes.tar.gz) · [Linux digests](evidence/cpu01/BAIGE-SHA256SUMS) · [Readback](evidence/cpu01/readback.json) · [Exit status](evidence/cpu01/exit.txt)

The development fixture uses a deterministic policy with the frozen motor's input and noise shapes. The preceding pinned trial had already qualified real frozen handback inference on this runtime. This job checks every genuine π0.5 query and physical command at its actual handback, including those on the second task. [Predecessor handbacks](../frozen-transfer-demonstration-20261124-r1/QUALIFICATION.md)

## Scheduler command

Parent dispatches one exclusive RTX 4090 in Baige Distributed Training, native LIBERO image, shared project mount, 90-minute scheduler ceiling. The outer command caps execution at 5,280 seconds with ten seconds of kill grace.

```bash
timeout --kill-after=10s 5280s bash /mowenjie/fluffy-bassoon-2/setups/symmeta-perceiver-followups-20261123-r1/src-22cc4bd4caeb/outputs/frozen-transfer-approach-confirmation-20261124-r1/launch.sh gpu 22cc4bd4caeb9a07bb2d69c745a1bf6aff522061
```

Shared receipts go to `/mowenjie/fluffy-bassoon-2/outputs/frozen-transfer-approach-confirmation-20261124-r1/gpu01`. The launch requires the pinned CPU exit and readback, an unused output directory, a scheduler hostname, one verified 4090 and NVIDIA rendering. The full task-first paired result and second readback are written after the last episode. Each episode is saved before its audit.
