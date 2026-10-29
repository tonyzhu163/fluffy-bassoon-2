# Frozen VLA steering evaluation

Verified robot evidence only. Method: [LIBERO interface](../frozen-vla-libero.md).
Paper: [draft](../paper/DRAFT.md), [decisions](../paper/DECISIONS.md).

## Released π₀.₅ technical pilot

Frozen official OpenPI `pi05_libero`; native horizon10, execution chunk10,
ten flow steps. LIBERO-90 task57: cream cheese → tray. This task belongs to
the source-training bank. Four auxiliary validation initial states40–43,
two attempts each; official held-out LIBERO-10 tasks remain untouched.

| Method | Successes | Interpretation |
| --- | --- | --- |
| Native π₀.₅, independent full Gaussian initial noise | 8/8 | Working motor on these four scenes |
| π₀.₅, 32D tied Gaussian noise | 8/8 | Working tied-noise interface on the same scenes |
| Frozen final DSRL steering actor | 8/8 | No success improvement on these pilot scenes |

- Four independent scenes; repeated attempts are correlated. This small
  sample does not establish benchmark-wide performance or a memory benefit.
- Observed baseline success is already at its ceiling on these pilot cases.
  Useful steering/adaptation headroom remains to be measured elsewhere on
  predeclared training/development cases before large corpus collection.
- Canonical shared prefix features4104D: maximum reference delta0.0.
  Replaying identical native initial noise: maximum action delta0.0.
- Actual GPU motor/renderer binding verified;79 focused Linux tests passed.
- Exact source: `6ec4fff88390041fea6bb23a8fd6fb7e70a7125c`.
- Checkpoint receipt SHA256:
  `ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170`.
- [Raw episodes, source contract and parity receipts](../../outputs/libero-pi05-corpus-20261029-r1/checks/20261029-1714/).

The fresh DSRL learner completed84 training episodes, all84 successful,
12,022 primitive interactions and24,540 SAC updates. Thus its source histories
also have no episode-success contrast on this task. On the same eight
validation episodes, native/tied/frozen-DSRL executed1,128/1,229/1,509 primitive
steps respectively; this pilot does not show a steering efficiency gain.

The deterministic final teacher separately passed its declared competence
gate with7/8 successes on calibration states24–31, costing1,428 primitive
steps. Those are different scenes, so this7/8 is not paired with the8/8 table.
Exact saved-state and next-action recovery passed; the next GPU optimizer
update passed the fixed numerical tolerance rather than bitwise equality.
The complete source checkpoint remains on Baige.

[Final raw episode/progress/qualification receipts](../../outputs/libero-pi05-corpus-20261029-r1/teacher-t57-final/)
include the completed teacher stage. Student adaptation, held-out-task
transfer, Plus/Pro performance and an advantage over ordinary DSRL or trained
static/reset controls remain unestablished.

## Native prefix feature implementation

The original fused model-plus-pooling path differed from separately
materialized native hidden states: maximum absolute error0.0059466 on a π₀
diagnostic. Pooling the same materialized hidden tensor with NumPy versus JAX
differed by at most9.54e-7; repeated fused calls were deterministic.

Canonical feature-v2 materializes native hidden states, then pools in float32.
Its reference error was0.0 in the corrected π₀ diagnostic and in the π₀.₅
pilot above. The qualification tolerance remains2e-4. This verifies the
chosen numerical feature contract; it does not establish a learning result.

## Student CUDA runtime

The original OpenPI Torch2.7.1+cu126 wheel failed on the allocated RTX6000
architecture. An isolated read-only overlay of existing Torch2.7.1+cu128
packages completed two real history-policy forward/backward/optimizer updates.
JAX GPU convolution outputs before and after those updates matched exactly;
actual same-process GPU binding and a complete student checkpoint were saved.
All86 focused Linux tests passed. Shared environments were unchanged.

[Runtime receipts](../../outputs/libero-pi05-corpus-20261029-r1/student-cuda-r1/).
These verify execution compatibility, not useful robot learning.
