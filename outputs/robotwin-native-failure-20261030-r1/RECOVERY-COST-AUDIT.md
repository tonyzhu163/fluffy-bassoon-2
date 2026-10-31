# Saved-state recovery and action-cost audit

Audited 2026-10-30 on the development host through read-only artifact inspection. No model construction, forwards, tests, robot actions, ledger changes, checkpoint edits, or GPU launches were performed. The stopped production source is `db0184089fa88a5f96c153f9b80029bb91b77def`.

## Outcome

All trained learner states are intact. Every coordinator and individual learner checkpoint agrees on parameters, optimizers, RNG, replay, sampler and pending rollouts. Their whole-trial cursors are incomplete because the sampler selected the next trial before native reset failed. Preserve the latest trained states and finish those selected trials with explicit episode-prefix recovery. A checkpoint rollback would discard useful training and is unnecessary.

All six latest and six saved stage-10/25 checkpoints passed full ZIP CRC checks. All parent/prior and child journal hashes verified, and replayed journal counters exactly match snapshots. Source committed-action totals reconcile to raw episode manifests; the only extra source exposure is two complete terminal first episodes, 239 SAC actions and 223 trained-reset actions.

## Exact latest cursors

| Learner | Complete K4 trials / iterations | Committed source actions | Learner primitive actions | Updates / optimizer steps | Pending full PPO trials | Selected trial |
| --- | --- | --- | --- | --- | --- | --- |
| ppo | 229 / 38 | 366,408 | 366,408 | 38 / 190 | 1 | 230 |
| sac | 236 / 39 | 394,231 | 394,470 | 244,820 / 734,460 | 0 | 237 |
| ppo_reset | 236 / 39 | 376,136 | 376,136 | 39 / 195 | 2 | 237 |

PPO retains trial 229: 1,996 actions in 64 decisions. Trained-reset PPO retains trials 235 and 236: 1,600/52 and 1,367/45 actions/decisions. Its seven-decision episode prefix is not yet in pending rollouts. SAC has 12,741 replay rows and replay position 12,741; their primitive counts sum to 394,470, matching its physical source charge. Its last eight replay rows exactly match the raw prefix observation, noise action, reward, next observation, termination, truncation and primitive counts at indices 12,733–12,740. SAC pending_chunks is zero. All 43 saved actor/critic/temperature optimizer state entries are at step 244,820; PPO and reset each retain 63 optimizer states at 190 and 195 respectively.

Global Python/NumPy/Torch/CUDA RNG fields are saved in coordinators. PPO learner RNG digests match their coordinator global RNG. SAC additionally retains private NumPy, interaction Torch and training Torch RNG states, plus actor, critic, target critic, temperature and all optimizer states. Component digests are in the JSON evidence.

## Lossless selected-trial recovery

- **ppo:** selected `place_phone_stand`, seeds `[774583299, 1050143885, 1061812452, 1816778261]`, iteration 39, trial 2.
- **sac:** selected `handover_mic`, seeds `[362798951, 121390495, 1150873424, 1191817971]`, iteration 40, trial 3.
- **ppo_reset:** selected `handover_mic`, seeds `[362798951, 121390495, 1150873424, 1191817971]`, iteration 40, trial 3.

Persistent PPO selected trial 230 but failed before recording a reset or action. Keep its existing pending trial 229 and execute selected trial 230 from episode 1 without sampling another task.

SAC and trained-reset selected trial 237. Each completed a successful terminal episode 1 with seed 362798951. Resume only episodes 2–4 using the original remaining seeds. SAC already observed all eight prefix decisions and performed 160 gradient updates on them, including 20 after the last acting checkpoint; preserve these saved replay/optimizer/RNG states and do not re-observe or re-update the prefix. Its last acting policy was update 244,800; the saved learner is update 244,820.

For trained-reset, rebuild the seven prefix PPO records from raw policy_input, pre_squash_actions, old_log_prob, old_value, recurrent_reset_mask, exact rewards and boundary flags, then concatenate the three newly collected episodes before admitting one complete K4 trial. Preserve original episode indices 1/2/3 for the suffix; the episode clock is still present when recurrence resets. Prefix trial_ends remain false; its final attempt_end is true. Scalar reset metadata is stored as a one-element array and must become a scalar before stacking. Update source/task counters exactly once by prefix + suffix after completing the trial, retaining the previous pending trials.

The raw recorder finalizes one `chunks/NNNNNN.json` record in place; it does not save separate immutable intent and transition JSON files. Bind the final complete JSON hash and every referenced NPZ/frame hash. A strict recovery manifest should bind the full coordinator, corpus contract, progress and episode logs, reset/episode manifest, selected task/seeds/trial/sampler cursor, prefix records and learner/pending-rollout counters. It must reject gaps, extra selected-trial episodes, unresolved intents, wrong behavior checkpoint identities and a nonterminal or budget-cut prefix.

| Prefix | Episode directory | Actions / decisions | Episode manifest SHA256 |
| --- | --- | --- | --- |
| sac | `episode-6306665914794a1083a68e81a992c237` | 239 / 8 | `06c92d6684656a84a5d2d0f85500981e772afb5a0d608d8d7151e033438effa1` |
| ppo_reset | `episode-723281a842b94affbb0adc576c397f44` | 223 / 7 | `d48ab06cd8132e8c41f3dedbc1ad738be206321a22a6af020e1ef57d33d801e4` |

All referenced prefix chunk JSON, decision/native arrays, and before/after frame files passed SHA256 checks and NPZ ZIP CRC checks. Both prefixes have pending_decision=null, terminal reward 1, success=true, truncated=false, budget_truncated=false, and requested_seed=actual_seed. The failed next reset issued zero policy control commands; internal physics reset steps remain explicitly uninstrumented.

The SAC context.behavior label is stale (`bounded uniform`). Its per-decision source_warmup=false, optimizer_updates, actual actor checkpoint identities and saved RNG are authoritative; do not infer warmup from that label.

## Checkpoint identities

Paths are under `/mowenjie/fluffy-bassoon-2/outputs/robotwin-wholetrial-{method}-20261030-r2/experiment/`.

| Learner | File | Bytes | SHA256 |
| --- | --- | --- | --- |
| ppo | `coordinator-recovery.pt` | 17,585,286 | `759533498db028a014e0908b9eaad0f7b193d09eb8df97b61fe50765be3f31f5` |
| ppo | `ppo-recovery.pt` | 17,537,016 | `2ddde084e5f8b4b1407e938e7a95558a3f39301c907ab3b175089ba221a8fd63` |
| sac | `coordinator-recovery.pt` | 422,155,985 | `33913fb781c618fec9152401ebe784ddf804cd638a5ad1fca68fa6db6cc2826f` |
| sac | `sac-recovery.pt` | 422,114,939 | `c775be56163f91f3c928f477800b6a0b3a31e7163fece096e6003b677635135e` |
| ppo_reset | `coordinator-recovery.pt` | 18,118,132 | `0ee20cca23a973b95659b3b476bcbc149384a7eeb7f51611a4542d7a95695960` |
| ppo_reset | `ppo_reset-recovery.pt` | 18,085,996 | `aa0b6f9b58eda6debe13235fc2e392372cbf8ec1a3aac01dfe720de2ec1d2966` |

The six latest files total 915,597,354 bytes. Original checkpoints should remain immutable; any copied recovery must retain the same hashes. Stage-10/25 hashes and CRC receipts are also recorded in the full JSON.

## Parent and child accounting

All admissions bind parent manifest SHA256 `1f80c08310dfa4543ec7a748ca6e2bf9aad589cd55978a402210abb0d83bdbc0`; worker identities and action caps match their child ledger headers. Each child has zero imported prior receipts, so prior costs are not duplicated across arms. All three journals register original run_id `experiment` at db01840.

| Ledger | Charged actions | Actual actions | Pending | Uncertain |
| --- | --- | --- | --- |
| prior | 202,394 | 190,394 | 12,000 | 0 |
| ppo | 418,654 | 418,654 | 0 | 0 |
| sac | 438,586 | 438,586 | 0 | 0 |
| ppo_reset | 404,847 | 404,847 | 0 | 0 |

Combined charged actions are **1,464,481**: 1,262,087 new-child actions plus 202,394 prior-campaign actions. Combined actual actions are 1,452,481. The prior campaign’s 12,000 pending qualification reservation remains charged and must not be refunded or confused with a new qualification reservation. All production-child pending/uncertain charges are zero.

The fixed child caps total 3,589,600, leaving 2,327,513 within those child contracts. The 4,589,600 parent ceiling leaves 3,125,119 numerically, but its unused prior allocation cannot silently expand the child caps. Separate newly owned qualification reservations of 21,600 / 2,700 were reported by the coordinator and are outside this audit; their foreign ledgers were not opened.

| Child | Ledger/corpus actions | Complete decisions | Committed source | Source prefix |
| --- | --- | --- | --- | --- |
| ppo | 418,654 | 13,555 | 366,408 | 0 |
| sac | 438,586 | 14,171 | 394,231 | 239 |
| ppo_reset | 404,847 | 13,105 | 376,136 | 223 |

There are 40,831 complete raw decision records across these three outputs. Recorder statistics report 15,824,122,014 frame-blob bytes and 838,513,204 decision-array bytes (16,662,635,218 bytes combined), excluding JSON, policy archives and training checkpoints. This is recorded raw data rather than only feature hashes. The audit checked all episode manifest counts and fully hashed the uncommitted tail; it did not rehash every historical RGB frame.

## Required execution-lineage handling

A new executor pin requires an explicit hash-bound migration that retains the original scientific checkpoint/ledger contract, records the actual new execution source separately, and uses a fresh run identity. The existing driver binds source_commit into both resume equality and the immutable ledger header; each child also already registered run_id=experiment. Reusing a new `.../experiment` leaf at another pin either rejects the contract/run identity or encounters old decision IDs starting at zero. Passing db01840 as a false new execution pin is not a valid workaround. Parent-owned continuation work is implementing explicit lineage and unique execution identities.

Process rollovers at durable whole-trial boundaries can limit simulator lifetime without changing the algorithm, action ceiling or frozen motor. The native GIL/binding root cause remains unproven. Fresh constructor/continuation qualification belongs on the separately allocated Linux job.

## Measured continuation cost

| Learner | Actions/s during complete source trials | Mean K4 trial seconds | Remaining K4 trials | Projected collection GPU-hours |
| --- | --- | --- | --- | --- |
| ppo | 28.61 | 55.92 | 371 | 5.76 |
| sac | 24.93 | 67.00 | 364 | 6.77 |
| ppo_reset | 28.48 | 55.96 | 364 | 5.66 |

The old-driver projection is 18.20 additional GPU-hours during source trials. SAC online optimization is inside its trial timings; PPO outer updates occur outside its collection timings. PPO outer optimization, stage-50/100 frozen evaluations, startup and process rollover overhead are additional, so 18.20 hours is a lower bound rather than a full-run quote. Any packed implementation needs its own measured throughput. These projections retain already trained states and do not include training them again. All three saved frozen-motor receipts report the same unchanged fingerprint: `721a7ed12c70d36e08882b02d8afee709286f6f27e72482570d7ac7451c0ae41`.

## Evidence files

The independent sequential and parallel artifact-audit reports are exactly equal as parsed JSON. The sequential implementation was slower because it read journal files serially. Both readers completed and exited normally.

- [recovery-cost-audit-parallel.json](recovery-cost-audit-parallel.json) — SHA256 `a8da2115d5f71040ef5ef2f20ebc650c76019dce4581abe59c15644674ce8e05`.
- [checkpoint-quick-audit.jsonl](checkpoint-quick-audit.jsonl) — SHA256 `94a8d71bb5812d5550d557ffe12a433cf999b171bacaa2d93502365c3847939f`.
- [prefix-evidence.json](prefix-evidence.json) — SHA256 `c103cad8c560fa02c7dd85efea07d31aae6913fe6199675f10636cd965f7e175`.
- [throughput-cost-evidence.json](throughput-cost-evidence.json) — SHA256 `51a37d315ead980d24838359d2cae31c45602bb6b911ce0dd0fe72c8e1ad80dd`.

- [replay-prefix-consistency.json](replay-prefix-consistency.json) — SHA256 `707f399e8f79c2f221b30794402b5a349cba439d8d58f9efacb927b46ede31d1`.

## Terminal scheduler costs and qualified continuation

Refreshed Baige terminal UI records show original workers ran PPO15,250GPU-seconds (4h14m10s), SAC17,710 (4h55m10s), and reset14,929 (4h8m49s), totaling47,889GPU-seconds. They ended20:24:32/21:07:50/20:21:30Shanghai respectively. Known earlierGPU9,702seconds is separate; these sum57,591seconds before separately metered optimization, offline fitting and new continuations. Saved zero-action failures still consume scheduler time.

Lossless continuation executor e825821 passed allocated CPU job-8jsjgsdc6cth:120tests plus all three real hash-bound manifests with unchanged learners. Source resumes are explicit migrations retaining DB scientific lineage, using supervisor c715e7b and fresh native Python processes at verified round boundaries. [Continuation launch evidence](../robotwin-native-continuation-20261030-r1/LAUNCH-REVIEW.md) records the unchanged source quotas and accounting.
