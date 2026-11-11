# Restore archived outputs on Baige

The cold root is `/mnt/bos/adgfyl/strong-plant-archives/outputs-retention-20261111-r2/`.
Every group directory contains numbered `.receipt.json` files. Each receipt
contains the original path, object path, exact byte count, mode, owner, mtime,
and SHA256 for every payload. The source folders have
`COLD-ARCHIVE-outputs-retention-20261111-r2-*.json` pointers to their groups.
Small configuration files and run records remain at the original locations.

Run [restore_payloads.py](restore_payloads.py) on Baige with the receipt for
each required part. Check PFS free space first. A multi-file part also needs
temporary scratch for the object and its extracted members.

```bash
cold=/mnt/bos/adgfyl/strong-plant-archives/outputs-retention-20261111-r2
python3 "$cold/restore_payloads.py" "$cold/mikasa-reset-s1000/001.receipt.json"
```

Restore all parts covering a checkpoint before loading it. The command verifies
SHA256, stages each file in the original directory, and preserves file mode,
owner, and mtime. Existing identical files are accepted. A different file at
the same path stops the restore.

The [failed MIKASA qualification deletion record](MIKASA-QUALIFICATION-DELETION.json)
lists permanently deleted step-3 payloads. The r2 and r3 duplicate initializers can be restored using their adjacent pointers and
[the original file restore procedure](../storage-audit-baige-20261111-r1/RESTORE.md).
The step-2 tarball was already cold archived in the previous pass.
