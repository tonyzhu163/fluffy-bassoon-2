# Paired trained continuation: scientific and budget review

Static review from clean detached documentation base `cd8ca01284ba2a958b65cd5a16272eb57a6a29aa`. Reviewed driver/admission executable `57016ba08eed79043fd67bf8b085bd6d42e6d766`, its preparation handoff, lifecycle review and original campaign recipe. This reviews proposed launcher requirements; it does not qualify an unwritten launcher, reserve resources, stop serial writers or authorize a live handoff. No tests/model computation or remote ledger constructor were run for this review.

The original parent manifest must hash to **`1f80c08310dfa4543ec7a748ca6e2bf9aad589cd55978a402210abb0d83bdbc0`**. Use its exact `workers[method].ledger_root`, `campaign_id`, `maximum_actions` and `native_episodes`; reject a matching copy of the journal at another root. Its scientific lineage is `db0184089fa88a5f96c153f9b80029bb91b77def`. The final launcher/executor commit is separate execution provenance and must be pinned in newly issued owner handoff manifests.

## Original caps and conditional cost

| Original allowance | Persistent PPO | Trained reset PPO |
| --- | ---: | ---: |
| Source hard safety cap | 1,100,000 | 1,100,000 |
| Four source-validation stages | 43,200 | 43,200 |
| Four target-evaluation stages | 76,800 | 25,600 |
| Four native-reference stages | 6,400 | 0 |
| Canonical worker total hard cap | **1,226,400** | **1,168,800** |
| One new six-trial K4 source round maximum | 10,800 | 10,800 |
| One full original input-due evaluation stage maximum | 31,600 | 17,200 |
| Input-due stage plus following one-round maximum | 42,400 | 28,000 |

The two original worker total caps sum to 2,395,200; this is not a new combined ledger or a new reservation. Keep their separate canonical ledgers/locks and unchanged headers. These evaluation allowances follow the existing fixed recipe; do not add new phase-cap fields to an old header. Prior campaign costs remain charged once by the original parent accounting, never imported again into either worker.

The proposal adds at most **21,600 source actions** toward the existing 100-round target, using the two genuinely trained recoveries. If original evaluation is due at input, the full worst-case additional evaluation allowance is **48,800**, giving **70,400** maximum policy actions for the pair. Use actual completed-unit cursors to subtract already completed work; do not reserve or execute the maximum again blindly.

One GPU, 32 CPUs, 192 GiB RAM and 16 GiB shared memory, with **at most 2,700 cumulative scheduler GPU-seconds**, is a proposed operational envelope only. It is eligible only within the owner's verified original residual compute and physical allowances. Startup, admission, probes, model loads, evaluation, updates, checkpointing and cleanup all consume that envelope. It is not a new resource budget.

The prior fresh packed allocation used 453 scheduler seconds / 21,310 actions. **600 seconds for both trained arms together is a planning estimate when no input evaluation is due, not measured trained speed or a driver wall limit.** The old 12k qualification hold and the settled 21,600-action/2,700-second qualification hold remain unavailable; sealed unused capacity cannot fund this launcher. The 1,339 LIBERO seconds reassigned to required DSRL are also unavailable.

## Required owner handoff before launch

1. The source owner alone selects new latest complete-update handoffs and relinquishes both serial writers at a stable boundary. Historical PPO round40 is not current ownership or residual-budget evidence. Keep current serial writers running until this controlled handoff; no launcher should stop them or opportunistically race their locks.
2. Bind the exact original parent SHA above; new latest coordinator and continuation-manifest paths/SHA; all raw-corpus/policy-archive ancestry; progress/episode artifacts; task, completed-unit and stage cursors; and the original canonical journal head/snapshot. No selected trial, retained prefix, pending rollout, pending/uncertain/unresolved physical decision, or unfinished native group is eligible for N6. Partial source trials finish on the owner's scalar path first.
3. Verify trained learner state: completed trials = six times completed iterations; learner updates = iterations; Adam/optimizer steps = five times iterations; full policy/Adam moments, LR `1e-6`, task schedule, private sampler and coordinator-authoritative Python/NumPy/Torch CPU/CUDA RNG retained. Check the actual saved `reset_each_episode` flag: false for PPO, true for reset PPO. The common contract's PPO field alone does not prove the reset learner's flag.
4. Both arms must be below100 updates to claim one new round each. Never collect update101. Bind the unchanged100-round source goal, six K4 trials per update, five epochs, minibatch16, joint-action weighting, KL guard0.1, native horizons, explicit noise, motor/norm/assets/capture fingerprints and original target repetition count1.
5. Recheck checkpoint hashes and canonical journal heads immediately before admission, before CUDA, and on exclusive ledger open. Admission needs an external time bound: a prior nonblocking ownership probe stalled in filesystem RPC, so reliable live lock rejection is not established. A stalled/held/changed ledger fails closed; no copied ledger, header rewrite, new receipt import or automatic retry.
6. Record owner-confirmed residual source/total/evaluation allowances and residual compute, excluding closed qualification holds. The source-only plan needs up to10,800 remaining source actions per arm. Input-due evaluation additionally needs its original remaining allowance and time; if unavailable, reject eligibility rather than suppress evaluation.

## Driver options for each arm

Run two separately supervised driver processes, one `--methods ppo` and one `--methods ppo_reset`, on the same attested GPU. Keep unique outputs, process sessions and execution identities, with the original separate ledgers. The fresh-controller `robotwin_packed_ppo_probe.py` command is **not** a trained launcher: its new ledgers, one-round scientific goal and disabled evaluations are ineligible here.

Each arm needs the following option set, with owner-bound variables supplied; this is an argument template, not an executed launch:

```text
--source-commit FINAL_EXECUTION_PIN
--lineage-source-commit db0184089fa88a5f96c153f9b80029bb91b77def
--resume OWNER_LATEST_COORDINATOR
--resume-trial-manifest OWNER_LATEST_CONTINUATION_MANIFEST
--resume-trial-manifest-sha256 OWNER_LATEST_MANIFEST_SHA256
--lineage-parent-manifest ORIGINAL_PARENT_MANIFEST
--lineage-parent-manifest-sha256 1f80c08310dfa4543ec7a748ca6e2bf9aad589cd55978a402210abb0d83bdbc0
--methods METHOD --output FRESH_ARM_OUTPUT
--campaign-root ORIGINAL_WORKER_LEDGER_ROOT --campaign-id ORIGINAL_WORKER_CAMPAIGN_ID
--campaign-max-actions ORIGINAL_WORKER_TOTAL_CAP --source-steps 1100000
--source-iterations 100 --source-eval-iterations 10 25 50 100
--collector-lanes 6 --collector-group-seconds 600
--max-source-trials-per-process 6 --recycle-at-round-boundary
--benchmark-episode-limits --ppo-trials-per-update 6
--train-episodes 4 --eval-episodes 8 --source-validation-episodes 4 --eval-trials 1
--eval-tasks adjust_bottle move_stapler_pad --native-episodes ORIGINAL_WORKER_NATIVE_EPISODES
--native-horizon 32 --episode-limit 600 --task-config-name demo_clean --action-limit 1.5
--ppo-epochs 5 --ppo-minibatch-trials 16 --learning-rate 1e-6 --ppo-max-update-kl .1
--seed 20261030 --record-corpus --final-observation-only
--model-path ORIGINAL_MODEL_PATH --norm-stats ORIGINAL_NORM_PATH
--robotwin-root QUALIFIED_ROBOTWIN_ROOT --rlinf-root QUALIFIED_RLINF_ROOT
--capture-qualification ORIGINAL_QUALIFICATION_RECEIPT
--wall-seconds ARM_DRIVER_SECONDS --boundary-reserve-seconds 300
```

`ORIGINAL_WORKER_TOTAL_CAP` is 1,226,400 / 1,168,800; `ORIGINAL_WORKER_NATIVE_EPISODES` is 1 / 0 for PPO/reset. Preserve the six original source tasks/order and common original SAC-config fields even though this launcher does not run SAC. Resolve all artifact/canonical paths before runtime changes its working directory. Pin and review latest main plus final execution source at deployment, keeping existing run worktrees immutable.

Do not pass `--stage-source-validation-only`, zero source-validation episodes, changed `--eval-trials`, source-iterations 1, source-steps 10800, fresh campaign roots, or probe-created controllers. Default LR is 3e-4 and campaign batch defaults differ, so do not rely on defaults. A manifest issued for a different executor pin is ineligible.

The 600-second group admission bound plus 300-second checkpoint reserve requires **more than 900 seconds remaining** at first source admission. Setting `--wall-seconds 600` rejects collection before RPC. Derive both arm deadlines from the shared outer residual deadline with launcher cleanup reserve; never grant each arm a separate 45-minute compute budget. A watchdog and ownership-checked TERM→KILL cleanup must include descendants even after a native parent exits. No automatic retry or extra allocation follows a timeout or native fault.

## Evaluation order, lifecycle and postconditions

An input-due original 10/25/50/100 evaluation must run normally with preserved `completed_units`/stage cursors before further source collection. Vector pools must remain lazy during it. Completed old stages and units are reused without new commands. After evaluation, admission closes the idle scalar scene and starts the N6 pool while retaining the identical frozen motor. A newly reached milestone after this one source round is deferred intact by the lifetime-six guard before `evaluate_stage`; it is completed by the next fresh process. A source-only timing screen may not bypass a stage already due at input.

Each source child has four data-reset RPCs, not four guaranteed setup attempts: the existing constructor plus four reset retry guards allow up to 160 setup attempts per child. Input evaluation retains up to 77 / 41 nominal scalar resets including preflight for PPO/reset. These process limits preserve accounting and reduce aging; they do not prove the native GIL/rendering issue is fixed.

For a claimed successful new trained round, require per arm exactly +6 complete source trials, +1 PPO update and +5 SGD steps, zero pending/selected data and at most 10,800 new source actions. Sum every physical/corpus/native-mask count, retaining any failed or unretained charge separately. Check old-policy replay at the original 2e-4 log-probability / 1e-5 value tolerances, the unchanged KL guard outcome, unchanged frozen motor and exact Adam/LR/config identities. An arm that stops before source dispatch has not qualified its new round.

Require durable group `checkpointed` status and coordinator ACK, absent ordinary-resume fences only after that ACK, zero unresolved/uncertain actions, and original full optimizer/sampler/RNG/task/evaluation recovery in the new hash-bound manifest. Preserve old and new corpus ancestry. Confirm both owned process sessions and simulator workers are gone, record GPU memory/renderer evidence and actual scheduler cost, and settle only actual costs against the original allowances. A nonzero cleanup exit is acceptable only under the original owner's separately audited complete-boundary rule, never as permission to repeat unknown physics. The next slice needs a new latest owner handoff.

Static checks found no additional scientific-cap change required in 57016ba for this proposal. The paired supervisor itself still needs exact-pin Linux validation in the owner's allocation and its own ownership/deadline review before execution. Fresh packed evidence does not establish trained throughput, long process stability or a fix for native GIL/OIDN failures. Current live readiness and residual budgets remain unverified here.
