# Initial continuation manifests: 90a234d

Independent static review found no blocker for the three audited production states. The reviewed parser is exactly the file in commit `90a234de059eb6d8dc681d668b03839739236e88`; its file SHA256 is `98fbdbad1c32cf6c9c23682e7de302d4a451faf8f37aaa5193f2903549ef0122`. No implementation files were changed.

This review and manifest generation used source reading, pure JSON operations and read-only backup-manifest inspection. The parser was not executed, and no models, forwards, tests, simulator calls or ledger constructors were run on macOS or development SSH. Allocated CPU qualification remains required.

## Manifest identities

| Method | Local manifest | SHA256 | Selected trial | Prefix episodes / actions / decisions |
| --- | --- | --- | --- | --- |
| ppo | [ppo-initial-90a234d.json](ppo-initial-90a234d.json) | `4b9249835ef388fcef362cdc0a94daa0b52cf61cc2327599a7136553a2c760e0` | 230 | 0 / 0 / 0 |
| sac | [sac-initial-90a234d.json](sac-initial-90a234d.json) | `bc7b99e1970a32b96b70aa9a80295e4089cad6404c20da0a7413a2f6d7730747` | 237 | 1 / 239 / 8 |
| ppo_reset | [ppo_reset-initial-90a234d.json](ppo_reset-initial-90a234d.json) | `87f09d05665fc3c315f57dc25fbf53db13d925e7893d055b5559a8bec697b669` | 237 | 1 / 223 / 7 |

Each manifest declares actual execution pin 90a234d and scientific lineage `db0184089fa88a5f96c153f9b80029bb91b77def`. All checkpoint, ledger snapshot, progress, episode-log and corpus-contract bindings point into `/mowenjie/fluffy-bassoon-2/exports/robotwin-native-recovery-20261030-r1/{method}/`. Original closed raw-corpus episode/frame refs stay under the immutable r2 `experiment/corpus` roots. Backup identities were compared against the prior independent audit JSON; every copied artifact identity matches.

## Actual-data compatibility reviewed

- Parser schedule and pending-rollout digests use the same typed algorithm as the independent audit. Expected counters, schedule digests and pending digests were copied exactly from `checkpoint-quick-audit.jsonl`; checkpoint file SHA256 binds all remaining learner/RNG/optimizer/replay content.
- The parser reads the actual single finalized chunk JSON through each episode manifest and follows corpus-relative NPZ/frame paths. Its JSON keys and metadata-array lookups match the audited recorder schema.
- Progress uses `stage=source_trial_start`. PPO binds the selected place_phone_stand trial 230, iteration39/trial2, with no raw prefix. It retains completed trial229 and its pending PPO rollout.
- SAC binds handover_mic trial237, iteration40/trial3, episode-row index1056. Its terminal eight-decision episode totals239 actions. Last behavior optimizer_updates244800 plus UTD20 equals saved244820; history_index12740 plus1 equals saved12741; saved primitive394470 equals committed394231 plus239. Thus the parser retains the last20 completed SGD steps and requires no prefix replay/observation/update.
- Trained-reset binds the same selected task/seeds at trial237, episode-row index1024. The seven-decision prefix totals223 actions; saved learner primitive376136 still equals committed source376136. Its raw episode_index0/history_index0–6, policy_update39, reset feedback, H600 clock and action_limit1.5 are compatible with `_reset_prefix_rollout`.
- Scalar reconstruction of log_probs, values and reset masks produces saved rollout shapes `[T,1]`; observation/action shapes are `[T,1,3090]` and `[T,1,1024]`. All saved rollout fields match the two existing pending reset rollouts, which remain separately bound by digests.
- Both nonempty prefixes are naturally terminal successful episodes with reward1, no budget cuts or pending decisions. Original seeds are `[362798951,121390495,1150873424,1191817971]`; the three suffix episodes must retain original episode indices1–3. Only original episode4 ends the assembled trial.
- Corpus roots and prefix paths resolve to the same absolute `/mowenjie/...` locations on the inspected host; there is no symlink-normalization mismatch for the parser’s containment checks.
- Backup ledger contracts retain DB lineage and have zero pending/uncertain child actions. Runtime GPU PCI receipt differences do not enter this initial parser comparison, which compares the saved checkpoint and saved ledger contract directly.

The owner’s allocated qualification should call `load_trial_continuation` on all three immutable backup coordinators with these manifests and SHA files, then confirm PPO empty prefix, SAC239/8, reset223/7, reset reconstructed shapes and preservation of pending data. This note makes no runtime qualification claim.
