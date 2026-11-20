# Panda and Sawyer source qualification on Baige

**The 5,200-update waypoint motor scored 1/50 Panda and 1/50 Sawyer.** The producer published the checkpoint, and the existing Distributed Training job completed all 100 registered episodes with those exact bytes. Of the five scored candidates, Huber regression ranked first at 34/50 Panda and 20/50 Sawyer. The registered development cutoff is 38/50 on each arm. No candidate qualified for the sealed 100-start-per-arm source confirmation. This readback used CPU only and launched zero jobs. [Native tape recount](evidence/motor/pfs-native.json) · [Source gates](../libero-paligemma-dualarm-20261120-r2/PLAN.md)

## Completed native development results

| Frozen motor | Update | Checkpoint SHA-256 prefix | Scoring job | Panda | Sawyer |
| --- | ---: | --- | --- | ---: | ---: |
| Original SFT | 1,600 | `0d8353ace1d1` | `job-iticjkite8we` | 25/50 | 6/50 |
| Huber regression | 6,400 | `77d18c386b2c` | `job-9zpwf7aapj3u` | **34/50** | **20/50** |
| Categorical | 6,400 | `824c695f68dc` | `job-3bffdcmp7clx` | 14/50 | 15/50 |
| Categorical with recovery suffixes | 2,400 additional | `071c280eb4b2` | `job-1w5htplffp89` | 17/50 | 15/50 |
| Waypoint-supervised | 5,200 | `a0bfa49fbb22` | `job-qhlx3223np0a` | 1/50 | 1/50 |

I hashed the checkpoint bytes and fit manifests, then checked all 500 NPZ tape hashes against the individual receipts and completed reports. Each arm had one result per draw on seeds 740000–740049. The ordered layout IDs, donor-state hashes and scene hashes matched across all five prepared panels. Each success came from a terminal native reward of one on Goal 5 BASE, with five issued controls per replan and a maximum of 300 controls. The older host-local tape format records the per-step outcome arrays and exact reset admission. The newer regression format also records the native step counter. Both formats reproduced their reported arm totals and successful layout IDs. [Reader](evidence/motor/read_source.py) · [Full checkpoint, panel and report digests](evidence/motor/pfs-native.json)

## Waypoint ownership and acceptance

`waypoint03` producer `job-6un3g38a249n` exited 0 and retained updates 1,300, 2,600 and 5,200. The final `plate_waypoint_v1` file was 2,378,769 bytes, SHA-256 `a0bfa49fbb2256d3a8a54f991dc6bdbdd04d4f47d4b13151038a056127036509`. Its completed candidate index binds it to fit manifest `de07ee0ef71c63d0bfbc691f8234e7647f63b92b1ac7ae0d90884429ffc1f0a4`. [Fit index and byte readback](evidence/motor/pfs-native.json)

The three receipts `hostway01prep`, `hostway01score` and `hostway01sum` all identify **one** worker, `job-qhlx3223np0a-master-0`, on one NVIDIA RPBZZZ6 in the Beijing Pro6K queue. Each stage exited 0 at evaluator pin `e2f794c41c9141a7aadaf67339da919336362f22`. The matching source report is `large-panel-hostway01/dev/scores/<full checkpoint SHA>/stride-5/report.json`, SHA-256 `ad3a08052ed7e410e9c98717c8495af05cc6fa4ce5c61382105e7511c0d3aa54`. The sole successful layout IDs were **1740037 on Panda** and **1740026 on Sawyer**. The aggregate job log independently gives 1/50 on each arm. This resolves the conditional evaluation against its existing owner. [Scheduler](evidence/motor/scheduler.json) · [Aggregate log and GPU probe](evidence/motor/source-context.json)

The PFS snapshot at 16:08 Shanghai on November 20 and scheduler check around 16:10 found seven large-panel roots, five complete development reports, zero accepted selections and zero sealed source manifests. There were also zero legacy source confirmation reports. The named scheduler campaign returned 36 terminal jobs. Earlier failed score attempts have no additional completed reports. Two partial retries in `large-panel-v3` each retain 24 Panda tapes and zero Sawyer tapes, for the original and categorical digests. Their later complete host-local scores appear above. The best motor was short by four Panda and eighteen Sawyer successes. [Panel and attempt inventory](evidence/motor/pfs-native.json) · [Scheduler snapshot](evidence/motor/scheduler.json)

## Other registered source work

| Generated collection | Admitted layouts | Accepted Panda tapes | Accepted Sawyer tapes | Layouts positive on both arms |
| --- | ---: | ---: | ---: | ---: |
| `gendemos01` | 96 | 66 | 94 | 65 |
| `gendemos02` | 128 | 93 | 128 | 93 |
| `gendemos03` | 128 | 98 | 123 | 95 |

All three collections completed. The readback verified all twelve worker index hashes against their campaign receipts. The recovery collector returned launcher exit 1, while its completed bank index retained 49 accepted suffixes from 120 handoffs. The completed 2,400-update continuation has its own scored row above. [Worker counts and hashes](evidence/motor/pfs-native.json) · [Recovery bank](evidence/motor/source-context.json)

**The first generated-data fit is available and unscored.** `broadfit01`, job `job-l5dd256gvivo`, exited 0 and retained updates 1,600, 3,200 and 6,400. Its manifest identifies the categorical 6,400 parent and includes 66 generated Panda and 94 generated Sawyer training trajectories from the first campaign. The final child is 2,891,409 bytes, SHA-256 `4720a1910cd6fe6c2592a75ea92f782955bac5ea77f3125b791c0ea215c740ad`. That digest occurs in none of the completed or partial source50 score directories. The scheduler showed the completed `broad96` fit and no matching active scorer. The six completed fit indexes account for 18 verified snapshot files. None is a wider fit using all three generated banks. Qualification of the first child remains unmeasured and is tracked by the separate generated-fit workstream. [Producer and score inventory](evidence/motor/pfs-native.json) · [Parent and bank provenance](evidence/motor/source-context.json) · [Separate follow-up](MOTOR-FOLLOWUP.md)

**Phase fit.** `job-cdoxszpm27hb` exited 0. Its 2,200-update checkpoint is 8,210,605 bytes and hashes to `51b14727e9bee101d3dad2f4b7064aff169f9f4d8c2121e3930ba39c1fa79949`, matching the offline check. That check had positive held trajectories on Panda layouts 24 and 27 and zero on Sawyer. There was no native four-layout or source50 report for this motor. The phase protocol documents the additional evaluator integration required for this architecture. [Phase receipts](evidence/motor/source-context.json) · [Phase protocol](../libero-paligemma-dualarm-20261120-r2/PHASE.md)

**Replanning stride.** `job-ybo867hh45g6` exited 0 on the original 1,600-update checkpoint. On released poses 32–35, stride five scored 2/4 Panda and 1/4 Sawyer. Strides one and two each scored 3/4 Panda and 0/4 Sawyer. Against stride five, stride one gained Panda poses 33 and 34 and lost 35. Stride two gained Panda pose 34 and retained both prior successes. Both shorter strides lost the Sawyer success on pose 35 and rescued none. These eight-start screens leave the registered stride-five source50 result at 25/50 and 6/50. [Stride report digest and paired wins/losses](evidence/motor/pfs-native.json)
