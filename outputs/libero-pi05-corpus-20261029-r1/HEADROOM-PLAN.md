# π₀.₅ noise headroom pilot

Declared2026-10-29 Asia/Shanghai, before these arm outcomes. The completed
task57 teacher pilot had84/84 successful training episodes and8/8 native,
tied-Gaussian and frozen-DSRL validation episodes. Large teacher collection
is deferred; first measure whether the proposed control space supplies useful
variation and repeatable success differences.

Three independent Baige jobs, one exclusive RTX6000 each, at most30 minutes
per job;16CPU/96GiB/8GiB shared memory. Tasks fixed before arm collection:

| LIBERO-90 task | Frozen split | Role |
| --- | --- | --- |
| 57: cream cheese → tray | train | Existing successful pipeline reference |
| 1: close top drawer and put black bowl on top | train | A different, composed source task |
| 2: black bowl → top drawer | validation | Development-task transfer/headroom check |

- Initial states0 and1; scene seed0 or1000 respectively, identical across
  all fixed-arm repeats and native references for that scene.
- Official frozen OpenPIπ₀.₅ receipt`ec064cce…`; nativeH10/chunk10,
  ten flow steps; standard LIBERO87edbd1; horizon400.
- Eight fixed32D arms: zero plus seven seed0 Uniform[-1,1] vectors.
  One vector remains fixed for the entire episode and is broadcast overH10.
- Each arm executed twice per scene; two independently seeded full-Gaussian
  native episodes per scene. Total36 episodes/job; at most14,400 primitive
  actions plus370 benchmark settling actions including qualification reset.
- Save the plan, exact arm vectors and hashes before any outcomes. Preserve
  real two-camera/state/action/noise traces and completed-episode progress.
- Select arms using repeat0 only; evaluate that selection on repeat1.
  Charge all eight selection episodes. Record outcome and trajectory
  repeatability, per-scene arm-success sets and the global constant-arm control.
- Simple search/explore-then-commit curves reuse actual recorded episodes;
  count all attempts. Native comparison exists only up to two attempts.
- No final-test/excluded task, learned policy, teacher optimization or
  student training in this pilot. No synthetic hidden condition is added.
- A finite constant-arm negative result concerns these sampled controls;
  it cannot rule out observation/chunk-conditioned steering. Two scenes
  cannot establish observation-only predictor inadequacy or task-level
  confidence. Any useful pilot signal needs broader development replication.
- Admit a small corpus only after checking meaningful behavior/outcome
  variation. Scheduled corruption and a reward-dependent schedule remain
  candidate data recipes; none is assumed to establish ICRL by construction.

Run artifacts: fresh`libero-pi05-headroom-20261029-t01-r1`,
`…-t02-r1`, and`…-t57-r1` under the Baige project outputs/worktrees.
Sources and actual job IDs are recorded after deployment in coordination.
