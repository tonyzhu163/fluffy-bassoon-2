# Final component integration details

These details refine HANDOFF.md against the actual now-written producer APIs.
No tests/models/simulator/GPU work was run in this macOS code-owner chat.

Dependency: collector `edbe6c5fba5180788c0283d66123021e0bdaa87f`, driver parent
`8248`. The focused test uses the REAL `BoundedTeacherRecovery` from that
collector. Qualify the combined source pin on allocated Linux.

The actual disk envelope is `source_coefficient_full_recovery_v1`; its `collector`
contains the inner `source_coefficient_teacher_v1` and learner. The outer file
retains Python/NumPy/Torch/CUDA RNGs, control and event. Prefix refs preserve
`retention=immutable`, `size_bytes`, and original canonical `root`/relative-path
references. Rolling slots and pending/partial optimizer states fail admission.

Control qualification is a metadata binding envelope:

```
format: frozen_vla_4d_control_qualification_binding_v1
control_interface: ORIGINAL file reference
driver_plan: ORIGINAL plan.json reference
driver_qualification: ORIGINAL gpu-qualification.json reference
completed_episode: ORIGINAL completed native episode summary reference
```

The code maps actual receipt fields: `status=passed`, `native_parity.passed`,
zero-edit booleans, nonzero KV gradient, zero motor gradients, equal frozen
before/after fingerprints, `complete_real_episode` and actual action count. It
also verifies plan scope/interface bindings and the actual interface's motor
FILE hash. It does not fabricate flags or request duplicate physical episodes.

The sole command-history projection/state owner remains the actual driver.
`DriverCoefficientStudent` calls `control.student_observation(native2062)`;
offline projection reads typed `student_previous_agent_command` and verifies
it against the last actually valid PRE-condition command. The student sees2076,
teacher sees2062. Both physical-reset zero and mask-index semantics are tested.

Pass `on_episode=student.finish_episode` to the SAME `run_complete_episode`.
On its zero-valid terminal path, the canonical receipt/mask/step sum are checked;
the unexecuted token is rolled back, actual proposal RNG consumption is retained,
valid feedback stays unchanged and the complete boundary closes. No synthetic
executed action/reward is created. Positive-valid or uncertain pending calls
remain blocked. A canonical-writer test covers this edge.

The actual physical readout entrypoint is a separate follow-on orchestration
commit under the same phase, using these classes and the existing driver; this
component pin by itself prepares data admission, fitting and controller APIs.
