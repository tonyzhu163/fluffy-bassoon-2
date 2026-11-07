# Outputs cold archival — verified complete

**Archived all 11 approved selections and reclaimed 207.700 GiB (223,016,656,896 allocated bytes) from PFS.** All 5,570 released bulk files have verified cold copies; 8,033 original regular files remain hot. The independent final reconciliation passed at **2026-11-07 14:53:57 Shanghai**. No selection failed; no GPU was allocated or workload stopped.

PFS free space was **71.77 GiB** at archival start and **278.05 GiB** at final reconciliation. This change includes concurrent project writes; the exact archive reclaim above sums physical allocation released, counting hardlinks once. The approved gross directory estimate was 207.79 GiB; compact/numerical evidence was retained on PFS.

## What moved

| Original selection under `outputs/` | Reclaimed GiB | Cold archive GiB | Bulk files | Hot regular files | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| `robocasa-opendrawer-ppo-ablation-3m-20261104-r1/C/attempt-01` | 17.963 | 13.370 | 65 | 220 | [Receipt](outputs__robocasa-opendrawer-ppo-ablation-3m-20261104-r1__C__attempt-01.json) |
| `robocasa-opendrawer-ppo-ablation-3m-20261104-r1/D/attempt-01` | 17.994 | 13.412 | 65 | 220 | [Receipt](outputs__robocasa-opendrawer-ppo-ablation-3m-20261104-r1__D__attempt-01.json) |
| `robocasa-opendrawer-ppo-ablation-3m-20261104-r1/E/attempt-02` | 17.022 | 12.940 | 47 | 221 | [Receipt](outputs__robocasa-opendrawer-ppo-ablation-3m-20261104-r1__E__attempt-02.json) |
| `robocasa-opendrawer-ppo-ablation-3m-20261104-r1/endpoint512` | 10.574 | 8.583 | 5,069 | 6,000 | [Receipt](outputs__robocasa-opendrawer-ppo-ablation-3m-20261104-r1__endpoint512.json) |
| `robocasa-opendrawer-ppo-ablation-3m-20261104-r1/eval-input-original40` | 6.472 | 5.131 | 1 | 1 | [Receipt](outputs__robocasa-opendrawer-ppo-ablation-3m-20261104-r1__eval-input-original40.json) |
| `robocasa-opendrawer-ppo-ablation-3m-20261104-r1/input-suffix-r1` | 14.996 | 9.876 | 21 | 4 | [Receipt](outputs__robocasa-opendrawer-ppo-ablation-3m-20261104-r1__input-suffix-r1.json) |
| `robocasa-ppo-short-chunk-benchmark-20261104-r1/attempt-01/OpenDrawer` | 34.621 | 23.887 | 42 | 4 | [Receipt](outputs__robocasa-ppo-short-chunk-benchmark-20261104-r1__attempt-01__OpenDrawer.json) |
| `robocasa-ppo40-matched-dev64-20261104-r1/staging-r1` | 25.888 | 20.330 | 4 | 2 | [Receipt](outputs__robocasa-ppo40-matched-dev64-20261104-r1__staging-r1.json) |
| `robotwin-sidney-matched-baselines-20261106-r1/grad02/train` | 9.291 | 7.642 | 8 | 88 | [Receipt](outputs__robotwin-sidney-matched-baselines-20261106-r1__grad02__train.json) |
| `robotwin-sidney-matched-baselines-20261106-r1/qual02/train` | 21.738 | 15.691 | 116 | 594 | [Receipt](outputs__robotwin-sidney-matched-baselines-20261106-r1__qual02__train.json) |
| `robotwin-sidney-matched-baselines-20261106-r1/qual03/train` | 31.141 | 23.503 | 132 | 679 | [Receipt](outputs__robotwin-sidney-matched-baselines-20261106-r1__qual03__train.json) |

Cold archive bodies total **154.366 GiB** (165,749,031,637 bytes) under:

`/mnt/bos/adgfyl/strong-plant-archives/outputs-retention-20261107-r1/`

Each object has a complete source inventory, SHA256 sidecar and receipt. Released-file lists, immutable plan/source, qualification, restore instructions and final verification also live beside the cold objects. Original-path adjacent pointers and `COLD-ARCHIVE-outputs-retention-20261107-r1.json` inside every source selection resolve to the exact archive and digest.

## Verification and retained evidence

Before each release: full source inventory/hash; GNU tar comparison; independent full BOS SHA256 readback; extraction of the complete archive into fresh staging; exact member coverage, bytes/SHA256, mode/owner/mtime, symlink targets and hardlink grouping; unchanged source metadata and current scheduler/reader checks. Every eligible file was rehashed and checked for visible readers again immediately before unlink. Only classified bulk regular files were removed. Original directories and symlinks remain.

Final reconciliation rehashed every retained original regular file, reconciled every receipt/inventory/pointer and released-file list, confirmed every released path absent, and matched **eight published actor identities** from E/C/D/original40 and DEV64 manifests against the verified archive inventories. See [actor/consumer restore mapping](RESTORE-REFERENCES.json). Existing older cold-storage pointers remain dependencies on their original archives.

Compact code, configs, logs, results, images, boundary markers and numerical-reference directories remain hot. The 38.77 GiB deferred predecessor state, matched PPO/GRPO production arms, pilot train07, held critic/joint-steering recoveries, canonical teachers/corpora, source5, shared prefix initializer, active Mikasa/AnyViewPhys material, caches and environments were excluded. All protected/deferred output roots were present at final reconciliation.

Two CPU workers used separate local `/tmp` staging within the 160 GiB cap. All run staging was removed and the archive worker exited. Source **`c8385974c7e3a36196f4456a24447c95214e704e`**, archive implementation SHA256 **`3d58eaa4a18a82e719bf1ce28c8527baa6b43b1e9c2811402b449d6d61acffe4`**. The final verifier was deployed from Git after the worker exited; see [final deployment](FINAL-DEPLOYMENT.json). No active run worktree was edited.

## Restore and evidence

Follow [RESTORE.md](RESTORE.md): verify the full archive and inventory, extract into fresh staging, verify all members, then restore only missing payloads without overwriting changed hot files. Stage any GPU reader's inputs back to PFS before launching it.

- [Execution result](RESULT.json) · [Final verification](FINAL-VERIFICATION.json) · [State](STATE.json).
- [Linux fixture qualification](QUALIFICATION.json) · [Immutable plan](PLAN.json) · [Runtime source](archive_outputs.py) · [Verifier source](verify_outputs.py).
- [Evidence manifest](EVIDENCE-MANIFEST.json): source inventories and released-file lists are losslessly compressed in `evidence/`; SHA256/byte sizes refer to the original uncompressed JSON retained on PFS and BOS. Raw metadata transport remains locally untracked.
- [Full storage audit](../storage-audit-baige-20261107-r1/RESULT.md) · [Original candidate review](../storage-audit-baige-20261107-r1/OUTPUTS-ARCHIVE-REVIEW.md).
