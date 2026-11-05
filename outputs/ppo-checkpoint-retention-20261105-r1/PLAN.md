# Closed PPO checkpoint retention

User request on 2026-11-05: evaluate shared storage and archive old PPO runs except their first, best and last checkpoints.

Baige PFS initially reported 1.9 TiB used and 185 GiB free on its 2,048 GiB volume at 19:16 Shanghai. The Mac checkout is only 456 MiB. A fresh read-only PFS census is running; concurrent storage work and live training make filesystem free-space changes different from this task's exact reclaimed allocation.

The first archival batch covers completed Stage 1 PPO/GRPO, the completed E/C/D OpenDrawer ablations, and the completed shorter-chunk benchmark. Preserve code, configuration, logs and evaluation summaries. Move closed raw rollout, environment, sampling and video directories, and remaining nonselected checkpoint payloads, into BOS archives with exact GNU tar source comparison, exact member coverage and complete BOS SHA256 readback before reclaiming PFS originals. Every move has an inventory and explicit restore command.

Preserve the existing initial, first trained and final checkpoints. Best means highest recorded DEV success, with earliest-step tie breaking: Stage 1 PPO 20, GRPO 30; E 40, C 20, D 10. Existing archives containing these checkpoints remain intact. Preserve complete final recovery state and the shared step0 initializer; compare protected checkpoint inventories before and after this batch. The 10M sources already have their older full checkpoints and corpora in verified BOS archives and keep their final130 recoveries on PFS.

Leave active Source5 and meta-learning trees, held experiments, shared inputs/environments and the explicitly referenced Flow-SDE final18 source untouched. Local active-dependency records cite the Stage 1 final40 models and ablation shared step0 initializer, which are protected.

The archive implementation is the existing pinned helper in `sources/stage1-grpo-013c2f10/outputs/storage-audit-baige-20261104-r1/archive.py`, SHA256 `ce5df6e4d313fbdc6948a897ee1311c143e8efef00fdcd86df4da82e4888dbb5`. Its server hash matched before launch. Run-specific selection and worker code, full inventories, archive receipts and census evidence live on Baige at `/mowenjie/fluffy-bassoon-2/outputs/ppo-checkpoint-retention-20261105-r1/`. BOS destination: `/mnt/bos/adgfyl/strong-plant-archives/ppo-checkpoint-retention-20261105-r1/`. Staging uses the development machine's `/tmp/ppo-checkpoint-retention-20261105-r1/`, which had 2.0 TiB free. No GPU allocation or model computation is involved.
