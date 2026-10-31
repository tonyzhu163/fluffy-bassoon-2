# E04: closure advanced, but the brief lead did not establish a successful lift

2026-11-01, Asia/Shanghai. This is a scalar diagnosis of the completed pair, not a new experiment or resource admission.

Both complete400-action episodes failed. The first three chunks, reset identities and other13 inserted-command parameters match; nominal contrast0.321608 is nondegenerate. The treatment did change the delivered nominal/drive target and moved the measured fingers further closed before lift. It did not settle them near zero during this short lead. Thus the result rejects the sufficiency of this one brief intervention, rather than establishing that closure timing cannot help or that the residual class is incapable.

| Boundary / stored quantity | hold_current | close_endpoint |
| --- | ---: | ---: |
| Before insertion: left measured finger coordinates, m |0.042632047 /0.044041846 |same |
| After insertion: nominal gripper |0.321608037 |0 |
| After insertion: per-finger drive, m |0.007688442 |−0.010000000 |
| After insertion: measured fingers, m |0.037493896 /0.040317968 |0.035169855 /0.037959058 |
| After insertion: phone height, m |0.742052078 |0.742296875 |
| After next source chunk (C5): measured fingers, m |0 /0 |0 /0 |
| After C5: phone height, m |0.795132220 |0.792401373 |
| After C6, before opening: phone height, m |0.754124224 |0.762832582 |
| Natural400-action endpoint: phone height, m |0.754124284 |0.756564379 |

The common pre-insertion phone height is0.742823005m. Treatment fingers finish about2.32mm /2.36mm more closed than control. This is a real measured response, not an absent intervention. Both remain far from their later zero-coordinate state before lift. Neither zero coordinates nor a nonzero separation alone proves contact, a stable grasp, or the object's location between the fingers.

The phone moves during C5: its endpoint position is(−0.164416,0.113178,0.795132)m for control and(−0.183910,0.143535,0.792401)m for treatment, with substantial recorded orientation changes. By the end of C6, its height has already fallen while both fingers remain at0/0 and the nominal command stays closed. Opening begins only at C7. The sparse endpoints support transient displacement followed by loss before release. They cannot determine whether the approach missed a stable contact, the phone was pushed rather than grasped, or an initially weak grasp slipped during transport. End-effector-relative phone geometry/contact/force histories and source measured trajectories remain unknown.

## The H32 cost command was a short physical exposure

The new observer captured exact existing TOPP/downsample counts; no additional planner or simulator calls were made. Both branches have identical counts through the relevant phases:

| Command | Left raw/downsampled positions | Right raw/downsampled positions | Derived left gripper allocation across32 submitted slots |
| --- | ---: | ---: | --- |
| C3, original closing phase |9 /9 |9 /9 |first9 slots:1 interpolation step each; final23:0 |
| C4, inserted hold |9 /9 |8 /8 |first9 slots:1 each; final23:0 |
| C5, original lift/transport chunk |741 /500 |10 /10 |first20 slots:16 each; final12:15 each |
| C6, next original transport chunk |352 /352 |9 /9 |all32 slots:11 each |
| C7, opening phase |8 /8 |9 /9 |first8 slots:1 each; final24:0 |

These allocations follow the pinned native quotient/remainder rule and captured counts; they are not dense observed waypoint/contact traces. C3's closing tail receives no left-arm interpolation steps, consistent with its delivered nominal stopping at0.321608 rather than the planned endpoint0. In C4, all treatment gripper targets are already0, so even its first exposed target delivers the closed command. The final23 zero-allocation slots do not erase the treatment contrast, but they also do not supply32 separate settling steps.

C4's recorded TOPP durations are0.0393333283s left and0.0353750903s right. These are planner outputs, not separately measured elapsed physics time. Captured counters are take_action_cnt96→128, while run_steps and FRAME_IDX both stay0; they do not independently measure a dense physical dwell. Treating one H32 cost hold as a long32-step settling intervention would overstate its dose.

## Expectations and the next bounded hypothesis

The prespecified prediction that closure should advance is partly met: nominal/drive and measured finger coordinates change in the intended direction. The stronger expectation that this lead alone rescues04 is not met. Since both branches lose height before opening, release timing alone remains an insufficient explanation. The original04 failure, both E failures and the successful11 cases are all retained; bank_candidate_04=null and all12/cache/fit/source/student admission remains false.

The next discriminating hypothesis is insufficient physical closure/settling before the lift, with approach/contact quality still an alternative. Coordinator steering after this factual diagnosis chose the next code-only bounded candidate: one fixed matched pair with FIVE total identical inserted H32 holds, using the same frozen post-C3 arm/other-gripper anchor in both branches and only the active gripper current-versus0 contrast. Five is a prospective dose, not a guarantee of adequate closure. This retains all140 original source frames:384 full planned slots, minimum358 slots to issue every original frame, within the unchanged natural400 limit. Keep the fixed two conditions/order, independently forked RNG, all prior failures, exact native executor and logged TOPP counts; do not use hidden measured qpos for an adaptive controller or retry-until-success rule.

Predeclare interpretation again: closure still not advanced/settled means the tested dose remains weak; more closure with no persistent lift points toward contact/approach or transport stability; sustained lift with a later failure shifts attention to placement. Neither negative result warrants a residual-class impossibility conclusion. No further phone GPU job, admission, old-clock reuse or sealed-capacity borrowing is authorized by this report. Parent retains any prospective cost/identity decision.

## Provenance and scope

E producer374ee8958f7cabcd54d4401057a604c6cc6ae51a; repaired resource helper6397b1dc005c08389e511db19a115c273580fbb6; job-i5dem442uskn. Result4cb73332320944a02de21493b0580f01e7b94abea70809d820831a3dd9633245. Terminal receipt11096cf683ddc069f6392a143289b422df47a51b1f64acbb02a638f453b0597a. Manifeste7e5bcb14e1e877a57eeedf7ef8759ccea4e520630b7624cf1ea432cdad5f146 binds29 JSON files/952223bytes: result,26 closed physical audits and2 manifests. All hashes/byte counts were checked; selected original scalar values and exact references are retained in EVIDENCE.json.

Earlier startup failure103GPU-seconds plus this execution233 gives336 of the600 ceiling; cumulative old phase2988GPU-seconds and22322 actual=charged actions, with pending0/uncertain0. These settled charges are retained; no new resource was consumed for this diagnosis.

The qualified c824 analyzer's existing physical-field schema and conventions were reused. Its whole-panel entry cannot directly ingest E because it hardcodes be247/three old cases and reconstructs reference commands by physical ordinal, which would be wrong after the insertion. No patch or fresh CPU unit is needed for these questions: the already-saved scalar JSONs suffice. No analyzer/model/test imports, arrays/NPZ/images/tensors, checkpoints/ledgers, SSH, live runtime observation or allocation ran here. Latest main17d5924ba6f66a3754355d23732c6d6f755e892a was reviewed for this metadata-only worktree; active and ended code/output identities were untouched.
