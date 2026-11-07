# Restore AnyViewPhys to PFS

The original project path is `/mowenjie/anyviewphys`. Its small `COLD-ARCHIVE.json` / `RESTORE.md` stub and adjacent `/mowenjie/anyviewphys.COLD-ARCHIVE.json` resolve the exact private BOS object, complete inventory, SHA256 and execution source. The archive retains the original `anyviewphys/` relative hierarchy, including models, environments, Git state, runs and caches.

1. Read RECOVERY.json and verify the **entire** cold archive SHA256. Copy it through a reliable byte stream into fresh local staging if the BOS FUSE mount cannot feed tar directly. Never overwrite the current project or an existing restore attempt.
2. Confirm enough free PFS space for the complete restored allocation plus overhead. Extract with Linux GNU tar and zstd into a fresh PFS staging directory, preserving ACLs, all xattrs, numeric ownership, symlinks and hardlinks. For example:

   ```sh
   anyview_restore_stage=$(mktemp -d /mowenjie/.anyviewphys-restore.XXXXXXXX)
   tar --acls --xattrs --xattrs-include='*' --same-owner -I zstd \
     -xf /path/to/verified-anyviewphys.tar.zst -C "$anyview_restore_stage"
   ```

3. Decompress SOURCE-INVENTORY.json.gz and compare every staged member: exact paths, regular-file SHA256 and size, mode/owner/nanosecond mtime, symlink target and internal hardlink topology. The pinned `tree.scan`, `tree.hash_rows` and `tree.verify` implement this check. Reconcile EXTERNAL-LINKS.json: links to base Python/OS tools or other outside-project data remain dependencies on those locations.
4. Verify no reader/job is using either path. If `/mowenjie/anyviewphys` is still only the expected two-file archive stub, preserve that stub under a fresh name and atomically rename the verified staged `anyviewphys/` directory into the original path. If anything else exists there, stop and reconcile it; do not overwrite newer work.
5. Check the recorded repository HEAD and clean/dirty state, restore inputs and relevant run receipts. Qualify the restored Python/container/accelerator runtime before compute. This is a file archive, not a VM/driver or process-memory snapshot.

The generated-score controller was already stopped before it started scoring. Its prepared plan and complete CPU inputs are preserved. Read QUIESCE.json and the original run's admission/hold before creating an explicit continuation; do not auto-start it merely because the files have returned.

Remove only your own verified restoration staging after successful reconciliation. Keep cold archives until a later explicit retention decision.
