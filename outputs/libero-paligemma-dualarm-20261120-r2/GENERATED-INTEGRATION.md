# Expanded source SFT and transfer readback

The generated training reservation is **[300000, 400000)**. Reserve **[740000, 750000)** for source development and **[840000, 850000)** for sealed evaluation. All intervals are half-open. Released source training uses poses 0–27. Released poses 28–31 supply offline validation. Generated training group IDs are 1,000,000 plus the native sampler seed. The bank reader checks the entire assigned seed budget, including rejected draws. Earlier sampler-smoke examples using seeds below 300000 are ineligible for SFT.

## Four collectors on one GPU

From the deployed repository root, use the simulator runtime and a fresh attempt name. Each child inherits the same scheduler device, runs Panda and Sawyer, and writes a separate bank and log. The launcher saves all assignments in `banks/campaign.json` before starting the first child.

```bash
set -eu
run=outputs/libero-paligemma-dualarm-20261120-r2
campaign=/mowenjie/fluffy-bassoon-2/$run
bash "$run/run.sh" "${PIN:?}" gencol01 sim gen_campaign.py \
  --out banks --seed-start 300000 --layouts 128 --max-seeds 160 \
  --candidates 8 --max-steps 300 --bodies Panda Sawyer
```

The four banks each request 32 admitted draws. Their 40-seed budgets begin at 300000, 300040, 300080 and 300120. Success counts are measured separately for each arm. Campaign exit 2 records a terminal shortfall. Successful tapes from fixed assignments remain eligible. A worker exception gives the campaign failed status. The launcher never retries or redistributes a worker's budget.

## Exact continuation from the 1,600-update fit

This example fixes the parent at `sft01/fit/candidates/step-001600.pt` with SHA-256 `0d8353ace1d15b7c6fa6a10b2dd3022b8637f6d71bac5748cdd72b69924dccdd`, as registered in the run plan. It prepends **the ordered bank paths in that fit's own manifest**, verifies their index hashes, and checks every inherited episode's group, split, tape hash and trajectory hash against the checkpoint. This preserves the first fit's original offline selection, which included searched positives. The append order is the completed `recover01/bank`, then generated banks 0–3 from the terminal campaign receipt. Include all native camera successes under that selection. Each generated bank's raw index bytes, exact command drafts and layout files are retained. The recovery snapshot retains the original index and branch receipts.

```bash
set -eu
run=outputs/libero-paligemma-dualarm-20261120-r2
campaign=/mowenjie/fluffy-bassoon-2/$run
parent="$campaign/sft01/fit"
parent_sha=0d8353ace1d15b7c6fa6a10b2dd3022b8637f6d71bac5748cdd72b69924dccdd
bash "$run/run.sh" "${PIN:?}" genfit01 vlm fit_campaign.py \
  --base-fit "$parent" --base-step 1600 --base-sha256 "$parent_sha" \
  --banks "$campaign/recover01/bank" \
  --generated-campaign "$campaign/gencol01/banks/campaign.json" \
  --architecture chunk_huber_v1 --steps 1600 --microbatch 4 --shortlist 3 --train-only
```

The child restores the parent's LoRA, action head and state normalization. It starts a fresh AdamW optimizer and sampler for 1,600 additional updates. Candidate steps 400, 800 and 1600 refer to child updates. The child manifest identifies its parent candidate and records the ordered source bank roots and index hashes. Ordinary `--resume` restores optimizer and RNG state for an interrupted fit with the identical contract. Omitting both `--base-step` and `--base-sha256` starts a new fit from MIX and estimates normalization on the expanded training partition.

## Source-only transfer fit

The following example fits a source reader for the fixed final child candidate with the exact SFT bank selections. The command checks the candidate index against the child manifest and checkpoint bytes. Promotion and target evaluation use the separately registered source panel gates. A new candidate needs its own registered comparison before any sealed evaluation.

```bash
set -eu
run=outputs/libero-paligemma-dualarm-20261120-r2
campaign=/mowenjie/fluffy-bassoon-2/$run
fit="$campaign/genfit01/fit"
fit_id=$(sha256sum "$fit/manifest.json" | cut -d' ' -f1)
step=1600
candidate=$(jq -er --arg id "$fit_id" --argjson n "$step" \
  'select(.status=="complete" and .fit_id==$id) | .snapshots[] | select(.step==$n) | .file' "$fit/candidates.json")
candidate_sha=$(jq -er --argjson n "$step" '.snapshots[] | select(.step==$n) | .sha256' "$fit/candidates.json")
test "$(sha256sum "$fit/$candidate" | cut -d' ' -f1)" = "$candidate_sha"
mapfile -t banks < <(jq -er '.demos[]' "$fit/manifest.json")
bash "$run/run.sh" "${PIN:?}" genjoint01 vlm large_transfer.py fit \
  --checkpoint "$fit/$candidate" --demos "${banks[@]}" --out joint
```

Both source fitting paths use `motor_bank.load_bank`. Generated ingestion checks the collector's actual layout file, accepted camera attempt, NPZ origin and cross-bank layout groups using `generated_split`. It compares issued actions to the successful headless draft, verifies file hashes, then applies the shared frame, quaternion, action and native reward checks. The transfer receipt records each resolved index path with its digest and every native tape and frozen proposal cache with its digest. `transfer_fit.verify_expert_data` reopens those indices, rebuilds the source membership, checks labels and compares reserved seeds and physical fingerprints before target use.

The recovery producer already writes standard motor tapes. Its `steps` counts teacher actions after handoff, its native reset uses the released source pose, and its NPZ contains those suffix controls. The reader admits this schema through the released branch and checks the handoff and remaining horizon. Generated groups participate in training sampling. Released offline groups alone supply validation anchors and validation error.

Mac verification covered compilation, collector and loader interfaces, and the historical parent writer at `e22c1d88`. The Linux collector banks, recovery yield, candidate bytes and native training results require readback on the allocated runtime.
