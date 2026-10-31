# RoboTwin trained continuation: cost and scientific review

Read-only static review of `scripts/diagnostics/robotwin_trained_pair_continuation.py` while its implementation is being revised. The reviewed root started at `b5a7737adbddc644c0b258f90ea88fbf23008605`; this document is not a final-code qualification. No tests, model computation, allocation, source-worker control or live ledger constructor were performed here.

The human-authorized objective remains **100 source rounds per original learner**, with no overall compute ceiling. The proposed future allocation is **one GPU, 32 CPUs, 192 GiB RAM, 16 GiB shared memory, at most 46,800 scheduler seconds (13 hours), with at most 43,200 driver seconds (12 hours)**. This is a metered operational envelope under that existing goal, not an added scientific target or a finite prepaid global pool. The initial 2,700-second subslice is **included**, never added to either envelope. A longer operational slice still needs an owner-issued reservation and actual-charge reconciliation, rather than another approval of the unchanged 100-round objective.

Neither the older 12k RoboTwin qualification hold, the settled 21,600-action qualification hold, nor LIBERO's separate compute authority contributes capacity. In particular, no sealed unused qualification capacity or LIBERO/DSRL time may be borrowed.

## Verified cost basis and dynamic planning

| Evidence | Source actions across both arms | End rounds PPO/reset | Timing basis |
| --- | ---: | ---: | --- |
| Trained R1 terminal evidence | 18,879 = 9,518 + 9,361 | 53 / 55 | 450 scheduler GPU-seconds; parent439.776 / pair368.380 seconds |
| Trained R2, owner reports done | 18,948 | 54 / 56 | parent379.55 / pair334.66 seconds; scheduler charge still pending |
| R3 bridge | Not treated as completed here | 55 / 57 **only if it passes** | Await its terminal source/cost receipts |

R1's local independent readback is `outputs/robotwin-trained-pair-20261031-r1/LOCAL-READBACK.json`. Its scheduler charge is owner-attested terminal evidence; parent/pair timings are not substitutes for scheduler GPU cost. R2's pending scheduler charge must remain unknown rather than being set to379.55 or334.66 seconds.

At post-R2 rounds54/56, the remaining source work is46 PPO rounds plus44 reset rounds: **90 genuine N6 rounds, at most972,000 new source actions**. The original final evaluation stage adds at most48,800 actions before subtracting exact completed units, for a conditional total maximum of1,020,800. If R3 passes to55/57, recompute45+43=88 source rounds, at most950,400 source actions; the full remaining final-stage maximum would then give999,200 actions. These are remaining-work bounds, not new ledger ceilings or live reservation receipts.

The simple conservative source planning basis is46 paired slots ×450 seconds = **20,700 seconds =5.75 hours**, using the first genuinely trained whole-allocation result. It is not a measured long-run total, a matched scalar speedup, or a guarantee that the final evaluation fits that rate. Later policy success, task seeds, native retries and remaining one-arm slots can change actual duration. The450-second basis already includes R1 cold loads/probe/audit/teardown; do not charge that same measured overhead twice. Add a separately stated margin for final scalar evaluation and different long-supervisor I/O/audit behavior. Do not estimate scalar K8 evaluation time by dividing its actions by the packed source rate.

Every real initial plan must use **new latest owner handoffs**, not hardcoded R1/R2/R3 manifest paths or cached round numbers. After every qualified child, use its audited output manifest, checkpoint, frozen-motor receipt and canonical head as the next input. Recompute remaining rounds and exact remaining evaluation units from those states.

## Original scientific and physical contract

The original parent SHA remains `1f80c08310dfa4543ec7a748ca6e2bf9aad589cd55978a402210abb0d83bdbc0`, with scientific lineage `db0184089fa88a5f96c153f9b80029bb91b77def` and qualified scientific driver `57016ba08eed79043fd67bf8b085bd6d42e6d766`. Bind the final continuation supervisor separately as execution provenance.

| Original hard cap | PPO | Reset PPO |
| --- | ---: | ---: |
| Source actions | 1,100,000 | 1,100,000 |
| All original source/evaluation phases | 1,226,400 | 1,168,800 |

Keep the two exact original canonical roots under `robotwin-native-wholetrial-20261030-r2/worker-ledgers/{ppo,ppo_reset}`. Do not clone a ledger, edit its header, import old costs again, or reduce the scientific source goal to a one-round probe. The upper source cost remains10,800 actions per new N6 round; success may make actual cost smaller. Exact latest physical charges and residuals still require owner-bound canonical evidence.

Each source child is one genuine update from its trained checkpoint: global batch six complete K4 trials, five epochs, minibatch16, LR1e-6, the existing KL guard0.1, native H32 and task horizons, fixed loss weighting, and the same frozen motor/norm/assets/capture. Require +6 trials, +1 update and +5 optimizer steps; preserve the complete Adam state and actual reset-memory convention. Preserve the task scheduler, private sampler and coordinator-authoritative global RNG; neither target evaluation nor child process reload may consume their source streams.

Scalar source validation and original target/native evaluation retain their10/25/50/100 schedule and exact unit cursors. Input-due evaluation runs before a lazy N6 pool. A source round reaching a milestone exits at the updated six-trial boundary before its newly due evaluation; the next fresh child completes that original evaluation. Earlier completed units remain suppressed. A partial evaluation continuation may proceed only after genuine newly completed, fully accounted units; a zero-progress wall stop is not an automatic retry signal.

The remaining full final-stage bounds are PPO31,600 (10,800 validation +19,200 target +1,600 native) and reset17,200 (10,800 validation +6,400 target). Subtract exact completed units from each phase. A fixed full-stage allowance is not a license to repeat an already completed unit, change target repetitions, or add target cases.

## Authority and progress acceptance

The hash-bound continuation authority needs schema `robotwin_trained_pair_continuation_authority_v1`, runtime/lineage/original-parent identities, and an exact reference to the fresh initial admission plan. Require `source_goal_rounds=100`, `original_total_compute_ceiling_seconds=null`, `qualification_capacity_reused=false`, `automatic_failed_retries=false`, hash-bound human authority and prior actual-charge/physical-accounting evidence. Resource and driver ceilings must be within the13h/12h maxima, with `initial_subslice_seconds=2700` and `initial_subslice_included=true`.

Do not reuse a spent initial admission output. Owner terminal release and mutually checked latest checkpoint/corpus/canonical-head evidence are still required before taking over a serial writer. An immutable historical manifest is not a live ownership lease. All potentially hanging filesystem/lock reads need an external bound before model work; a blocked admission halts without changing another writer.

For every future child, persist a hash-bound input request containing method, exact predecessor artifacts, output identity and runtime. Bind the completed audit back to that exact input and final checkpoint/manifest hashes before advancing the supervisor. The source audit must retain exact corpus/native-mask/ledger equality, task/seed multiplicities and final sampler identity, old-policy replay at2e-4/1e-5, unchanged frozen motor and durable coordinator ACK. Preserve source corpus ancestry across every fresh process.

For evaluation-only children, verify no source group/trial/action/update was added, source policy/Adam/private sampler/task schedule/global RNG are unchanged, new history is a prefix-preserving extension, and phase charges correspond only to newly completed original units. Retain an exact final coordinator SHA even when a completed arm has no next continuation manifest.

At round100 an arm may still owe original final evaluation. Run only that evaluation in fresh scalar children; never collect round101. Once its100-stage completion and all earlier required stages are attested, retire the arm and preserve its final artifact/head receipts. Spare memory after one arm retires is not permission to resurrect it, pad trials, enlarge the global batch, initialize a fresh controller or add cases. If both initial arms are already complete, reject/finish empty work before opening model/probe contexts.

## Static caveats and required final checks

The implementation agent is already addressing initial-round100 admission, exact remaining evaluation-unit bounds and durable child bindings. These fixes remain final acceptance conditions, not completed claims in this review. Source/evaluation failed or unknown outcomes must preserve charges and raw evidence and halt; no automatic failed retry or refund is admitted by the unchanged source-goal authority.

One minor schema edge is visible in the reviewed version: `scheduler_cap_seconds` allows2700, but `maximum_driver_seconds` requires at least2700 and at most scheduler minus180. Thus a2700 scheduler envelope cannot satisfy it; either align the lower bounds or document that continuation needs at least2880 seconds. This does not invalidate the proposed46,800/43,200 envelope.

The13h scheduler limit must contain the12h driver and all pre/post work and owned-session cleanup. Child wall limits and minimum start reserves are operational admission guards, not guarantees of native progress; partial/no-progress outcomes still need an owner audit. Account actual terminal scheduler seconds, including unsuccessful attempts, rather than summing or substituting overlapped child/parent timings.

Before any future execution, pin the final supervisor, review its Linux CPU evidence and the latest main, rebind dynamic latest owner inputs, and record the admitted operational slice plus prior charges. Afterward, reconcile final frozen/optimizer/RNG/corpus/ledger state, original evaluation completion, ownership cleanup and actual scheduler cost. This review performs no deployment and claims no long-lived native GIL/OIDN fix or guaranteed13h completion.
