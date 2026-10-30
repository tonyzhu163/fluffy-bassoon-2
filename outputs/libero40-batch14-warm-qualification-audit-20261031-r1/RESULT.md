# Warm-start batch14 qualification: terminal audit

Baige `job-pqje6dlyz0of`, run
`libero40-batch14-warm-qualification-20261031-r1`, source
`376415e6915399436cb3fea214c50bd3e41f32d5`, finished **exit0**. The actual
allocation probe,50 focused CPU tests and2 typed/legacy motor-RNG tests passed.
The independent terminal CPU checkpoint/receipt audit passed **27 checks**.
It constructed no model and ran no forward pass, simulator, optimizer or test.
No original output was changed and no older raw trajectory bank was scanned.

The source batch completed14 K4 trials/56 episodes on14 interleaved source
goals,894 steering decisions,55 successes and1 failure. The only nontrivial
pattern was `SFSS` on `libero_10:00`, state0. Every episode was collected under
the inherited fixed behavior before the new batch's first update. This pattern
does not establish an adaptation or memory benefit; it has no matched control
and stochastic resampling can also change outcomes. No target performance was
measured. The model inherited actual all40 source exposure; this new phase
visited14 goals, not40.

The complete recovery retained inherited238 updates/476 SGD, every old
non-contract leaf (fingerprint
`c931fae66f3e05defdc8dafe8c441d47cd57fde8a462526141d8bedf27977d64`),
and the original952 completed source episodes. Migration and actual restoration
performed zero SGD. The new phase committed1 update/2 Adam steps; actual Adam
state clocks are **478**. New cursor14, remaining0, no pending trials/plan;
sampler state exactly matches the inherited sampler after the planned14 seed
draws. Python/NumPy/Torch/one-CUDA-generator RNG and typed threefry2x32 uint32[2]
motor RNG remain present. Initial sampler/global/motor state matched the parent
exactly. Exported candidate weights and manifest agree with the full recovery.

All four exact behavior replay errors were zero. Update metrics were finite:
KL0.00375277, clipping fraction0.0520134; this is numerical/runtime evidence,
not evidence that the policy improved. The frozen OpenPI motor's51 bfloat16
parameter leaves had identical before/after hash:
`c565dd3f0c4005be728cf7597698cba937c4011cd5bd813700a89b789f9d69e9`.

| Measured cost | Value |
|---|---:|
| Collection |691.911s|
| Exact replay |4.767s|
| Two-step PPO update |15.362s|
| Collection + replay + update |712.040s|
| Actual cold worker duration |849.842s|
| Other cold setup/shutdown work |137.802s|
| Charged GPU time, including60s allowance |**910s**|
| Training episode primitives |8,702|
| Training episode settling |560|
| Qualification settling |10|
| Total new resource actions |**9,272**|
| Prior source resource actions |195,964|
| Cumulative source resource charge |**205,236**|

The9,262 training episode actions were collected fresh;10 additional actions
were qualification settling. The195,964 prior charge also includes repairs,
source validation and qualifications, so205,236 is a **source resource charge,
not a training dataset size**. There were no unresolved action ceilings.
At this one batch's observed lengths, training throughput was13.008 physical
actions/s for collection/replay/update together, or10.189 resource actions/s
including the charged cold worker cost. This14-goal near-ceiling slice does not
certify throughput on the rest of the source suite. Peak sampled whole-device
memory was10,986MiB; Torch peak allocated/reserved1,507,669,504/1,654,652,928 bytes.

Immutable artifacts on Baige:

- Full recovery SHA:
  `c689a2ddb190510ba09452ad9c755446bf605351d22b41017c75187834e90ee0`.
- Candidate SHA:
  `e1a59d9942f138c95c40500e805f9f500ebcb71604bf74ded32de03e5e7166e2`.
- Index SHA:
  `45759da5c5fe1e1213c4a8e2e1ef6a977eb4e6609d256bf7534fc06ea5582a88`.
- Worker-clock SHA:
  `7cf07aac114c4999be6fe3d2ca4858cc42c25bf9e91ea426ed2b165d056f21c8`.

`receipt.json` stores the compact checks, exact artifact paths/hashes, counters,
RNG signatures, per-goal committed exposure and numerical metrics. The included
`read_checkpoint_audit.py` reproduces the read-only CPU inspection. Full model
execution/deployment admission still belongs on allocated resources.
