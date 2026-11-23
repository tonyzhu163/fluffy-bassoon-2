# Measured FK and seed-41 input qualification

**264 of 288 cases have qualified inputs for both methods.** The full UR5e × π₀ panel is available. Each seven-joint arm has seven qualified tasks and one blocked task. The seed-41 reader is available on all 288 cases. All qualification ran on Baige CPU and consumed 0 GPU-hours.

| Panel | Paired cases | Matched tasks | FK yaw MAE | SymMeta yaw MAE on the same cases | Historical SymMeta native success, full panel |
|---|---:|---:|---:|---:|---:|
| IIWA × π₀.₅ | 84/96 | 7/8 | 2.34° | 6.07° | 76/96 |
| Kinova3 × π₀.₅ | 84/96 | 7/8 | 4.46° | 7.50° | 60/96 |
| UR5e × π₀ | 96/96 | 8/8 | 2.37° | 10.53° | 54/96 |

The inputs were evaluated on the original closed-loop passive supports. The yaw errors describe these fixed cases. Success counts come from archived GPU queries. FK query success remains unmeasured.

## Reset and kinematics

The simulator reconstructed layout 11 with its official per-task seed, the target arm's stock geometry, and the mounted PandaGripper. The compiled world grip site's ancestor chain comprised exactly the ordered measured arm hinges on all 24 arm/task combinations, with six on UR5e and seven on each of IIWA and Kinova3. The archived active and passive frame-zero motor states matched the reproduced state within the existing numerical reset check. All eight π₀ passive resets required the historical quaternion sign flip. Their active siblings used the original sign. All 24 whole-simulator fingerprint comparisons differed. Numerical reset correspondence is established on the 22 tasks that reproduced the nominal sibling's 15-control encoder and TCP trajectory within 2 × 10⁻⁶ rad and 10⁻⁶ m per component.

| Blocked task | Support seed | Maximum active joint residual | Maximum active TCP residual | Affected indices |
|---|---:|---:|---:|---|
| IIWA, libero_object 1 | 31820111 | 5.7742 × 10⁻⁵ rad | 9.019 × 10⁻⁶ m | 48–59 |
| Kinova3, libero_spatial 8 | 31830811 | 1.0252 × 10⁻⁵ rad | 2.408 × 10⁻⁶ m | 84–95 |

Both discrepancies repeated across fresh runs. Active and passive frame-zero state residuals on these 24 cases were zero. A frame-zero tool pose cannot resolve every joint of a redundant seven-joint arm. The tapes store joint differences, and the absolute archived reset joint vector is unavailable. Their rows carry `fk_available=false` and a null FK yaw. The source of the replay differences remains unresolved.

For the 264 available cases, FK used the reset joint measurement, the three archived encoder differences and the passive tape's clipped issued-command means. The loader assumes five executed controls for each valid chunk. The archives contain no separate primitive-count or action-order log for the final chunk. Planar alignment consumed only the original action exposures and computed world displacement. Labels entered during scoring. The largest qualified per-chunk discrepancy between the FK and logged public planar path lengths was 0.760 mm.

## Frozen-reader and harness identity

The reader used the original 24-slot arrays, valid mask, historical DIS preprocessing and seed-41 checkpoint. All 288 unrounded yaw, gain and grip triples exactly matched a separate CPU readback that invoked the historical evaluator's actual loader and predictor. All 288 gripper signs matched the archived GPU queries. The maximum absolute differences from the rounded GPU values were 0.099873° yaw and 0.00290721 gain. Historical rounding agreement is recorded independently. Use the exact exported gain and grip for both arms of a fresh paired query.

The evaluator launches pinned IIWA and UR5e to `490f25c5716aa067199c5b65e32caf6cd15421a4` and Kinova3 to `465f33dc657ceab341dbceaf6b98355b94810059`. All 24 historical query-shard SHA-256 values matched the published paired-statistics provenance. The policy-specific weight locations, OpenPI pin, query seed and noise schedule, proposal horizons, action convention and π₀ quaternion gate are recorded in [QUERY_CONFIG.json](QUERY_CONFIG.json).

## Deliverables

Each compressed panel contains all 96 indexed cases and eight reset receipts. Fields include both tape hashes, reset and query identifiers, historical success, numerical residuals and availability. Available cases also include the three input XY exposures and measured FK displacements. The export check reconstructed every available FK angle from those pairs.

- [IIWA cases](evidence/gpu09iiwa-cases.json.gz)
- [Kinova3 cases](evidence/gpu06kinova3-cases.json.gz)
- [UR5e π₀ cases](evidence/gpu08pi0fix-cases.json.gz)
- [Aggregate counts, execution pins and artifact hashes](evidence/summary.json)
- [Run and consumption instructions](INSTRUCTIONS.md)

Baige raw outputs are under `/mowenjie/fluffy-bassoon-2/outputs/symmeta-fk-crossarm-20261123-r1/`. The verified native CPU reader exports and UR5e result are in `cpu02` at `1f42520057e893c31e9f910abb3def6c0719aa5f`. The independent task-level availability records for both seven-joint arms are in `cpu03` at `2b21b0db3252a7879c747e7433a208892b105576`. The earlier `cpu01` run stopped on archived GPU rounding discrepancies. Its 288 CPU triples were identical on rerun. The `fk01` and `cpu02` logs preserve the repeated active replay failures.
