# Generated Goal 5 BASE source qualification

Development uses 50 distinct layouts paired across Panda and Sawyer. Update 1600 from the existing `sft01/fit` is the first candidate. Each arm contributes 50 fresh closed-loop episodes. The separate sealed panel has 100 new layouts and 100 episodes per arm. The full manifests and outcome tapes live on project NVMe.

## Development CLI

From the repository root on allocated Baige, set `PIN` to the deployed revision. Each invocation of the existing launcher needs a fresh attempt name. All stages share the same persistent `panel` directory.

```bash
run=outputs/libero-paligemma-dualarm-20261120-r2
campaign=/mowenjie/fluffy-bassoon-2/$run
fit=$campaign/sft01/fit
panel=$campaign/large-panel-v1
bash "$run/run.sh" "$PIN" lpdevprep01 sim large_panel.py prepare --root "$panel" --phase dev --fit "$fit"
bash "$run/run.sh" "$PIN" lpdevscore01 vlm large_panel.py source --root "$panel" --phase dev --fit "$fit" --step 1600
bash "$run/run.sh" "$PIN" lpdevsum01 sim large_panel.py aggregate --root "$panel" --phase dev --fit "$fit" --step 1600
```

The prepare command publishes `dev/manifest.json` and `dev/rows/*.npy` before scoring. Source writes one native tape and digest per layout and arm beneath `dev/scores/<checkpoint-sha>/stride-5/`. Aggregate writes `report.json` there only after all 100 episodes validate. The default rollout uses eight independent native processes, one batched frozen motor, and five issued controls per replan. One arm or a zero-based layout shard can be selected with `--arm Panda` or `--shard-index I --shard-count N`.
