# Four-case result: settling occurred; repeated holds did not rescue04

2026-11-01, Asia/Shanghai. Saved scalar metadata only; no new physical run, numerical array/image analysis, test or resource allocation.

All four prespecified conditions failed at the natural400-action limit. The matched comparison is valid within its declared limits: all four native reset identities, three-chunk preludes and frozen anchors match; recorded same-grip targets match across doses; other13 parameters match; active nominal contrast is0.472361803 at both doses. The first hold's physical endpoint also matches exactly between the one- and five-hold cases of each grip condition. All52 physical audits bind the closed result/canonical chunk references and have continuous before/after physical boundaries. The full proposed/submitted/native-mask NPZ arrays were not decoded here.

**Answer to the mechanism question:** five holds establish a nearly stationary measured finger configuration under the full-close drive. They do not establish an adequate grasp. The result weakens a simple lack-of-settling explanation: the extra holds add physical exposure and little further finger motion, yet no condition succeeds. Contact identity/force, source initial object pose and source/native path dynamics remain unresolved. No residual-class impossibility or further dwell retry follows.

## Closure during each repeated hold

The common C3 endpoint has measured left finger coordinates0.043664921/0.044740230m, nominal0.472361803 and per-finger drive0.015979899m. These are named prismatic coordinates, not a calibrated geometric aperture or contact measurement.

| Hold endpoint | Current-command measured coordinates, m | Close-command measured coordinates, m | Captured left/right downsample positions |
| --- | ---: | ---: | ---: |
| 1 |0.039582305 /0.042444088 |0.035924509 /0.038519777 |8 /9 |
| 2 |0.036867034 /0.039620250 |0.035651904 /0.037592687 |8 /9 |
| 3 |0.035608567 /0.038318999 |0.035675451 /0.037569180 |8 /8 |
| 4 |0.035606589 /0.038291160 |0.035677817 /0.037563372 |8 /8 |
| 5 |0.035591833 /0.038274597 |0.035659090 /0.037551921 |11 /10 |

The one-hold cases end at the first row; the five-hold cases reproduce that first endpoint exactly and then continue. The close branch delivers nominal0 and drive−0.0099999998m throughout every hold. The current branch retains nominal0.472361803/drive0.015979899m. Thus the command contrast persists.

Compared with one close hold, five close holds add about0.27mm/0.97mm of measured closure. The close coordinates plateau after hold2: subsequent changes are only tens of micrometres. The phone also becomes nearly stationary on the captured boundaries: close-branch height after hold2 is0.742361963m and after hold5 is0.742360532m. At hold5, the two grip conditions' measured configurations largely converge: close joint7 is about0.07mm *more open* than current, while joint8 is about0.72mm more closed. A large nominal contrast is therefore not a correspondingly large endpoint-coordinate contrast after settling.

Positive coordinates against a negative closed-drive target may reflect object contact, other constraints or actuator tracking. They do not prove a missed contact, and zero coordinates are not a criterion for an adequate grasp either. The captured values justify “settled configuration under full-close command,” not “secure grasp achieved” or “closure remained inadequate because it was nonzero.”

## Native exposure and arm following

C3 has only7 left/right TOPP/downsample positions for32 submitted gripper parameters: under the pinned quotient/remainder rule, its first7 slots receive one interpolation step each and the final25 receive none. Each hold repeats a constant gripper target, so its first exposed slot already delivers that target; trailing zero-allocation slots do not remove the current-versus-close contrast.

The five holds have43 left and44 right planned/downsampled positions in total, versus8/9 for one hold. Each left hold has8 positions except the fifth with11; first four right counts are9/9/8/8, then10. These are captured planner/implementation counts, not a dense measured force/contact or elapsed-physics trace. The extra calls provide actual additional native exposure; the near plateau cannot be explained by accidentally omitting all additional calls.

Arm following remains offset from the captured drives. Immediately before lift, left joint2 measured/drive is2.413188696/2.447207689rad for one-close and2.414301395/2.447421551rad for five-close: about−0.0340 versus−0.0331rad. Joint4 is−1.047170877/−1.042235732rad versus−1.048935652/−1.041968703rad. Extra holds do not make the arm reach its drive targets exactly. The native TOPP calls also change small drive details and consume planner randomness, as anticipated in the recipe interpretation; this was never a pure elapsed-time intervention.

Both doses' subsequent lift chunk has741 raw left TOPP positions downsampled to500; the next transport chunk has352/352. All32 left gripper slots receive positive interpolation allocation in those chunks. Later opening differs too: one-hold cases have8 left positions and delivered nominal0.603015065, five-hold cases have10 and nominal0.753768861. That reinforces the native-recipe limit; the height decrease below already occurs before either opening phase.

## Initial lift and height decrease precede opening

Logical source chunks are identical, while physical ordinals differ: after one hold, lift/next transport/opening occur at C5/C6/C7; after five, C9/C10/C11.

| Case | Phone height before lift, m | After lift, m | After next transport, before opening, m | Final400-action height, m |
| --- | ---: | ---: | ---: | ---: |
| 1-current |0.742363334 |0.764889061 |0.741115034 |0.741115034 |
| 1-close |0.742227912 |0.784620106 |0.767390847 |0.766133904 |
| 5-current |0.741804957 |0.776292622 |0.743310690 |0.742189586 |
| 5-close |0.742360532 |0.773621023 |0.761382639 |0.771230578 |

All four fingers reach measured0/0 by the end of the lift chunk, while the nominal command remains closed. The phone moves laterally and rotates, but rises only transiently at the captured boundaries. In five-close it moves from(−0.083750,0.001593,0.742361)m before lift to(−0.137933,−0.052273,0.773621)m, then to(−0.219920,−0.055606,0.761383)m before opening. Full recorded poses/quaternions and all active-arm/finger measured/drive coordinates are retained in EVIDENCE.json.

The common pattern is a transient interaction during the initial lift/transport, followed by lower height before release. This is consistent with a weak/unstable grasp or contact/path mismatch, but it does not identify which. Some endpoints remain elevated or change height later; “all four dropped flat on the table” would overstate the evidence. No source-synchronized contact or end-effector-relative phone trace was captured.

## Consequences

The prediction that additional holds would produce substantially more pre-lift closure is only weakly supported: most measured closing occurs in the first hold and the configuration settles by roughly hold2. The proposed repeated-hold rescue fails under both matched grip commands. Further blind dwell increases are unsupported by these data.

The remaining factual gap concerns initial contact and lift dynamics: was the native object in the same source-relative starting pose, which surfaces contacted the fingers, and how did the object move relative to the hand during the first transport? Four equal *native* resets and earlier source seed/variant/camera checks do not establish author-source object worldpose or measured trajectory equality. Those quantities were not in the qualified source fields. Source drive-target equality is not a measured-path or contact guarantee.

No new physical experiment, contact framework, hold profile or fit is proposed/admitted by this report. If the parent proceeds with its separately reviewed all12 **command-imitation diagnostic**, retain the original failed04 and its status explicitly. That can assess command approximation while preserving failure evidence; it is not a positive12-teacher bank, competent-teacher result, source RL admission or adaptation claim. Current candidate04=null and all12/cache-fit/source/student gates remain false for this result.

## Provenance, cost and verification limits

Producer8632e421412388e8bb9c933b2b10cf5940dc7156, job-snrggr7e3xoo, result96242cb9ccaf9d7a9bb8efe6ef923cd4b5619151ccd7c099da54903a78939b95. Formal terminal receipt2da5b0bf18eb0c444b32e9cb6514970a5c5aeaa828b4552e5f3316999e7c5899. Packet manifestd9cddbf6b030883e27c6f92785b2fe97d94ce93fbd39620ce8ada1767f6285ea binds57 JSONs/1,884,749bytes: result,52 chunk audits,4 manifests. Every hash/byte count and result→audit→canonical-chunk reference was checked. Record-level target JSON, frozen anchors and comparison falsifiers were independently checked; numerical NPZ proposals/submissions/masks were not decoded. Native physical before/after boundaries are continuous within all four cases.

F consumed310 charged GPU-seconds and1600 new actions. Its terminal snapshot has3298 settled phase GPU-seconds/23922 actual=charged actions; the owner's later independent PARTIAL11 cache charge154 brought the phase to3452 without actions. That separate cache is outside this mechanism comparison and does not change the failed04 status. Old E336=103+233 and all earlier charges/recoveries remain retained. This diagnosis adds no resource charge.

Only standard-library JSON/hash reading and selection of existing scalar fields ran here: no NumPy/Torch/model/test imports, array/NPZ/image/tensor decoding, checkpoint/ledger construction, SSH, live observation, allocation or action. Metadata worktree rebased onto reviewed latest mainffa786176c62f0b2b90d801af397a8852dfe4d6c before this report; no active/ended source or output contract was changed. TERMINAL-CHECK.json retains the earlier receipt-only check separately from the complete saved-scalar interpretation.
