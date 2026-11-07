# Restore an archived output selection

Archives preserve each selected original `outputs/<run>/...` hierarchy. The adjacent `<selection>.outputs-retention-20261107-r1.json` pointer and root-local `COLD-ARCHIVE-outputs-retention-20261107-r1.json` identify the exact object, SHA256, verified inventory and original location. The original compact records and directories remain on PFS.

Use Linux with GNU tar and zstd. Stage the required selection back to PFS before starting a GPU reader. Do not point GPU inference or training at a cold BOS symlink.

1. Read the selection's receipt. Verify the archive's complete SHA256 against `archive_sha256`. Reconcile the adjacent `.inventory.json` against `source_inventory_sha256`.
2. Extract into a fresh, empty local staging directory with enough space, rather than directly over a current output tree. For example, for the staged DEV64 models:

   ```sh
   outputs_restore_stage=$(mktemp -d /tmp/outputs-restore.XXXXXXXX)
   outputs_archive=/mnt/bos/adgfyl/strong-plant-archives/outputs-retention-20261107-r1/outputs__robocasa-ppo40-matched-dev64-20261104-r1__staging-r1.tar.zst
   sha256sum "$outputs_archive"
   tar --acls --xattrs --same-owner -I zstd -xf "$outputs_archive" -C "$outputs_restore_stage"
   ```

3. Verify all staged members against the recorded inventory: exact paths, file SHA256/size, mode/owner/mtime, symlink targets and hardlink groups. The run's qualified `verify_restore()` function implements this check. Full staged restoration passed before every hot payload release.
4. Consult the run's `<selection-key>.released.json` list. Restore only its missing original-path payloads, using the verified staged bytes and original metadata/hardlinks. Stop if an existing destination differs; do not overwrite changed files, active-run inputs or compact records retained hot. Confirm the intended reader is idle before changing its inputs.
5. Recheck every restored payload SHA256 and record the explicit migration/restaging in the consuming run before launch. Remove only this restoration's own temporary staging after verification.

The receipt's `restore_command` records the original tar layout. It is not an instruction to overwrite the current tree without these checks. Existing older `COLD-STORAGE.json`/`.bos-archive.json` pointers inside a restored tree remain dependencies on their original, separately retained archives.

[Archive plan](PLAN.md) · [Execution results](RESULT.json) · [Final reconciliation](FINAL-VERIFICATION.json).
