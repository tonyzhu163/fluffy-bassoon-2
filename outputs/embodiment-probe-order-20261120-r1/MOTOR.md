# Which source motor warrants an oracle test?

**No verified PaliGemma checkpoint qualifies.** The strongest frozen motor scored 34/50 Panda and 20/50 Sawyer on the nominal LIBERO-Pro2.0 BASE Goal 5 plate push. It needs four more Panda and eighteen more Sawyer successes to reach the registered development gate. The cheapest next assignment is a read-only reconciliation of the existing R2 fits and development reports on Baige, at **0 GPU-hours**. The 384-tape probe-order experiment measures command response on familiar arms. Its query-motion scores cannot count as native task successes. [Native score receipts](../libero-paligemma-dualarm-20261120-r2/evidence/source50/native.json) · [Source gates](../libero-paligemma-dualarm-20261120-r2/PLAN.md) · [Frozen design](../embodiment-tape-observability-20261120-r1/RESULT.md)

## Completed source evidence

| Frozen checkpoint | Panda | Sawyer |
| --- | ---: | ---: |
| Original SFT, update 1,600 | 25/50 | 6/50 |
| Huber regression, update 6,400 | 34/50 | 20/50 |
| Categorical head, update 6,400 | 14/50 | 15/50 |

All three issued five controls per replan, up to 300 controls, on draws 740000–740049. At audit base `4bd494d1` the compact three-model readback agreed with the retained JSON, including report and checkpoint digests. Counting distinct successful layout IDs across the three rows gives **45/50 Panda and 34/50 Sawyer**. Even choosing the successful recorded motor separately for each start with hindsight leaves 16 Sawyer starts unsolved. This calculation uses the common seed/layout IDs across separately hashed panel manifests. It bounds selection among these three observed raw outcomes only. A bounded action correction would change the trajectory and needs its own paired experiment. [Per-layout IDs and manifest hashes](../libero-paligemma-dualarm-20261120-r2/evidence/source50/native.json)

The earlier Panda-only pilot had offline validation action MAE 0.03492 and zero native rewards on two fresh Panda starts. A Sawyer contact replay found the palm against the plate and table at control 82, then no sampled plate contact from 90 through 297. A changed teacher route later succeeded on layout 0 at control 204. In the separate recovery collection, teachers finished 49/120 handoffs (17 Panda, 32 Sawyer), with 65 teacher failures and six replay mismatches. These receipts establish that some source states are recoverable with new teacher actions. They do not tell us why each frozen-motor episode failed or whether an inverse applied to its proposals would rescue it. [Panda pilot](../libero-paligemma-dualarm-20261120-r2/SERVO.md) · [Sawyer replay and recovery](../libero-paligemma-dualarm-20261120-r2/EXECUTION.md)

## What is already owned

The coordination row lists two source results. The R2 closeout at `d2f102e2` (Nov 20, 13:59 Shanghai) adds the completed categorical result. Its earlier 1,175/1,600 training-progress line and first scoring-job IDs are historical. The later receipts identify the successful scoring panels. Current scheduler states for the follow-ups below are unverified. [Coordination](../coordination.md) · [Execution ledger](../libero-paligemma-dualarm-20261120-r2/EXECUTION.md)

| Work | Last supported status |
| --- | --- |
| Recovery continuation and waypoint supervision | Recovery collection finished with 3,653 eligible teacher controls. R2 reports a categorical continuation for 2,400 additional updates. Waypoint fit `waypoint03` has job `job-6un3g38a249n` and a verified fixed Panda validation replay. The native 50-layout scores for both follow-ups are registered. Neither final child checkpoint nor either score has a local terminal receipt. |
| Replanning stride | R2 lists `job-ybo867hh45g6` on the original 1,600-update motor. Its stride-five pilot was 2/4 Panda and 1/4 Sawyer on released layouts 32–35. Retrieve the actual stride-one/two report and paired wins/losses. The ledger gives `stride01/dev/report.json`; the example uses `stride01/panel/report.json`. |
| Generated demonstrations and SFT | R2 lists `gendemos01/02/03` jobs `job-isezj1jp5rg6`, `job-bt2z25374394`, and `job-1da0x4xol75p`. The 96/128/128 figures are requested layout counts. Terminal bank yields, bank-trained checkpoints and native scores need readback. The first categorical continuation with generated data and the wider fit are registered plans. |
| Further progression | The phase-conditioned route has code and a four-layout screen design. Its documentation reports static verification. A phase-model source50 integration, a selected passing development motor, and sealed source confirmation are unverified. |

The live queued/running distinction for these follow-ups is unknown. Check the existing IDs and output directories before requesting another allocation. [Stride protocol](../libero-paligemma-dualarm-20261120-r2/STRIDE.md) · [Generated collection contracts](../libero-paligemma-dualarm-20261120-r2/GENERATED-INTEGRATION.md) · [Phase screen](../libero-paligemma-dualarm-20261120-r2/PHASE.md)

## Dispatch decision

**First dispatch one CPU reader on the existing project volume.** Under `/mowenjie/fluffy-bassoon-2/outputs/libero-paligemma-dualarm-20261120-r2`, reconcile the launch and exit receipts, each fit's `manifest.json` and `candidates.json`, each generated `banks/campaign.json` with its four worker indices, the stride report, and existing `dev/scores/*/stride-*/report.json` files. Match each registered child update to its actual checkpoint digest, report hash, arm counts and distinct layout IDs. Return one row per candidate, including failed, partial and missing rows. Check existing scorer ownership by checkpoint digest, phase and stride. If all 100 tapes already exist and only the report is missing, the R2 owner can run `large_panel.py aggregate --phase dev --root "$panel" --fit "$fit" --step 5200 --stride 5` for waypoint on the matching pinned Linux runtime. Aggregation requires zero new rollouts. [Tape readback](../libero-paligemma-dualarm-20261120-r2/large_panel_protocol.py)

**Conditional evaluation only.** If that readback finds the registered final waypoint checkpoint complete and its 50-layout evaluation both missing and unclaimed, score that existing checkpoint once. It directly tests whether training on plate position and future wrist motion improved raw source success, without another fit or collection. Exact candidate is `waypoint03/fit/candidates/step-005200.pt`, architecture `plate_waypoint_v1`. Use the digest from its completed index and verify the bytes before allocation. That digest was unavailable locally. One Baige Distributed Training RTX PRO 6000D, 12 CPUs, 96 GiB host memory, 8 simulator workers, stride 5, 100 episodes, proposed cap **95 minutes (1.58 GPUh)**. The R2 owner can execute the following in the verified source50 checkout, after confirming a fresh attempt name. The launcher and evaluator below are already used by R2. [Waypoint producer](../libero-paligemma-dualarm-20261120-r2/waypoint_run.py) · [Native evaluator](../libero-paligemma-dualarm-20261120-r2/large_panel_rollout.py) · [Launcher](../libero-paligemma-dualarm-20261120-r2/run.sh)

```bash
set -eu
run=outputs/libero-paligemma-dualarm-20261120-r2
campaign=/mowenjie/fluffy-bassoon-2/$run
panel=$campaign/large-panel-v3
fit=$campaign/waypoint03/fit
pin=$(jq -er '.revision' "$panel/dev/manifest.json")
test "$(git rev-parse HEAD)" = "$pin"
test "$(sha256sum "$panel/dev/manifest.json" | cut -d' ' -f1)" = 663077ec7aad3621173d725d2cf1550e3859a7ca46a0de50727e25aec8c6bfc0
fit_id=$(sha256sum "$fit/manifest.json" | cut -d' ' -f1)
digest=$(jq -er --arg id "$fit_id" 'select(.status=="complete" and .fit_id==$id)
  | .snapshots[] | select(.step==5200 and .file=="candidates/step-005200.pt") | .sha256' "$fit/candidates.json")
test "$(sha256sum "$fit/candidates/step-005200.pt" | cut -d' ' -f1)" = "$digest"
bash "$run/run.sh" "$pin" wpdev50audit01 vlm large_panel_campaign.py \
  --fit "$fit" --step 5200 --root "$panel" --workers 8
```

**Decision-changing expectation:** the weaker arm reaches **38/50** at the fixed waypoint update. Anything lower keeps this candidate out of sealed confirmation. A pass merits the registered 100 fresh source layouts per arm. The existing `accept` subcommand ranks only the original three candidates, so the R2 owner must bind any later winner to the registered follow-up comparison before confirmation. A motor reaching **75/100 on each arm** can justify a source-only test of the true inverse through the same clipped action interface, counting shifted failures rescued and nominal successes broken on paired starts. No checkpoint in the reviewed evidence has reached that point. This audit used **0 GPU-hours**. [Selection implementation](../libero-paligemma-dualarm-20261120-r2/large_panel_protocol.py) · [Native oracle criterion](../embodiment-tape-observability-20261120-r1/REVIEW.md)
