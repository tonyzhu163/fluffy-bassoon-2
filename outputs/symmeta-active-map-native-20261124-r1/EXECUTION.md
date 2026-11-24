# Deployment, timing and validity

The shared run root is `baige-dev:/mowenjie/fluffy-bassoon-2/outputs/symmeta-active-map-native-20261124-r1`. The parent owns Git and both scheduler submissions. The worker owns the CPU input preparation, instrumented rollout and readback in this directory. Source and target offline evaluation are a separate workstream. [Registration](PLAN.md) · [32 case assignments](cases.json)

| Shanghai time on November 24 | Receipt |
|---|---|
| 14:07 | Registered all cases, methods, signs, contrasts and stopping rule before target estimates or scores. All four official banks contain 50 rows. Parent received the physical-support release schema. |
| 14:14 | CPU qualification started on core 34 only, with CUDA disabled, at deployed commit ed4f3aace5a3c808d91679373639322f16e5ccba. Eight groups use source-independent fixture estimates, genuine official resets and five real simulator steps per method. The CPU job uses no motor weights. |

The production motor and simulator both use the historical multi-arm runtime a8a997e7d06bb131d3f9710204c58e106f0f0904, also used to acquire support. Motor loading uses the shared verified pi05_libero transport. The registered forward map enters the simulator-only worker. The nominal motor receives ordinary world proprio, both camera streams and the official prompt. The native observer reads back the actual underlying `ArmEnv._env.step` argument. Support tape checks reconstruct all causal issued controls from the hashed original layout-11 archive and check the received OSC action against the measured goal. They independently recompute the incumbent from issued controls, FK, the 20 Hz clock and the frozen source law.

### Throughput

The original inclusive baseline projects 31.6 minutes per robot for 128 episodes. A metadata-only timing read of the ongoing previous camera cohorts at 14:15 found 12.69 seconds per episode over the intervals between 37 complete UR5e quartets, and 14.50 over the intervals between 18 Kinova3 quintets, both excluding startup and the first group. These imply 27–31 minutes per robot before fresh startup, with different tasks and maps. Allow approximately 35–45 minutes per robot and cap the *official allocations* at 60 minutes each. Stop earlier after the fixed cohort or if the remaining allocation is exhausted. Queue time is separate.

### Parent handoff

Use two distinct official 1×RTX 4090 Distributed Training jobs on the same immutable panel. Each job runs one loaded policy and one simulator worker. The command takes absolute shared paths. The parent records the official ID, job start and remaining aggregate allocation. The optional along methods run after all eligible essential groups for each body.

~~~sh
bash "$study/launch.sh" prepare "$pin" prepare01 34 "$released_json" "$released_sha"
bash "$study/launch.sh" gpu "$pin" gpu01-UR5e UR5e "$run/prepare01/panel.json" "$panel_sha" 3540 "$run/cpu01/completed.json"
bash "$study/launch.sh" gpu "$pin" gpu01-IIWA IIWA "$run/prepare01/panel.json" "$panel_sha" 3540 "$run/cpu01/completed.json"
~~~

Use 60-minute scheduler limits and a payload deadline of at most 3,540 seconds per robot, including probe and startup. Any retry must use a fresh directory and the remaining time from that robot’s one-hour reservation, counting scheduler overhead on all prior allocations. The native launcher validates the allocated GPU, binds its physical EGL device, records the pinned runtime and generates atomic compressed per-episode receipts. The first genuine complete essential group qualifies each allocation. During collection, inspect counts and forensic pairing only. Recheck approximately 20 minutes after that first group. After both jobs terminate, run the reducer once on one CPU with their exact attempt directories.

~~~sh
taskset -c 34 "$py" -B "$study/analyze.py" "$panel" "$panel_sha" "$ur5e_out" "$iiwa_out" "$audit" --first 0
taskset -c 34 "$py" -B "$study/analyze.py" "$panel" "$panel_sha" "$ur5e_out" "$iiwa_out" "$audit"
~~~

Use separate first-group output directories for UR5e case 0 and IIWA case 4. Native endpoint data include the full per-case table, four task effects, all predeclared comparisons, saturation, achieved vs desired commands and rescues/losses. A failed or unverified source qualification leaves the architecture hypothesis unresolved. The physical panel receipt independently records excluded or corrupted support identities.

If the frozen inference job supplies the matched matrix rows in its original format, `pack_release.py` constructs the same version-1 release from the final `sealed.json`, `fresh-rows.json` and `fresh-files.json`, using the parent-supplied revision and SHA-256 for each file. It verifies the upstream exit, checkpoint, selected source test and original support joins. It copies no offline target metrics into the native panel. Run only after the parent releases those exact completed artifacts. A missing direct source qualification can instead be recorded in a matrix-only release for validated exploratory endpoints.

The manifest SHA-256 is `cb251c7b151713467536a90bf03890d1098cdbb1eaf33cc119723b1cce14bf44`. Both jobs also require the sealed panel SHA. Each completed retry records every proposal and actual received action. If one archive fails its checksum or pairing, retain its identity and failure reason, mark the architecture comparison unresolved, and report valid paired endpoints separately.
