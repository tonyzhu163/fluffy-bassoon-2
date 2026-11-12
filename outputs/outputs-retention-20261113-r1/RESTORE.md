# Restore the archived Baige outputs

Use the existing Baige BOS mount and a Linux allocation with Python 3, GNU tar, zstd and enough free space on PFS for the desired payload. The helper uses the temporary filesystem for a verified archive copy and extraction.

Each released PFS directory contains a `COLD-ARCHIVE-outputs-retention-20261113-r1-*.json` pointer. The `groups` entries in [VERIFICATION.json](VERIFICATION.json) give all eight directory pointers and each part's BOS receipt path. Restore one group by setting `archive_group` to its ID. This example restores the complete PPO step-one checkpoint across four parts.

```bash
archive_root=/mnt/bos/adgfyl/strong-plant-archives/outputs-retention-20261113-r1
archive_group=sidney-ppo-step-1
for archive_receipt in "$archive_root/$archive_group/"*.receipt.json
do
  python3 "$archive_root/restore_payloads.py" "$archive_receipt" || exit
done
```

The helper checks the archived content against the receipt and reads back the staged PFS file before publishing it at its original path. It accepts already restored files with matching hashes and raises an error for a conflicting file. For an archive tar, the helper restores regular-file content, ownership, mode and modification time. GNU tar also preserves archived ACLs and extended attributes in the staging extraction when those metadata are needed.

The PFS allocation needed for a full restore is about 47.03 GiB. Restore all parts of a checkpoint before loading it.
