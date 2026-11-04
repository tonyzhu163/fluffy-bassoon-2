# Restoring archived assets

Every reclaimed source has an adjacent `<name>.bos-archive.json` pointer on Baige. Its `archive`, `sha256`, `archive_bytes` and `restore_command` identify the complete object and exact restoration command. Compact receipts are mirrored in this audit folder; full per-file inventories remain on PFS and BOS.

Before restoration, confirm the source path is absent and that no active job will write there. Read back the archive and verify SHA256 against its receipt, then execute the recorded restore command from an allocated Linux environment. Restore to the original location so environment shebangs and editable package paths remain valid.

The `workspace-r2` archives use concatenated Zstandard frames and GNU tar streams. **Use the recorded `tar --ignore-zeros ...` command.** Omitting `--ignore-zeros` stops at the first tar stream and gives an incomplete restoration. All eight streams are required. The qualification restored every member, preserved hardlinks/symlinks/literal filenames and detected same-size content corruption with unchanged mtime.

The other run archives use an ordinary single GNU tar stream. Configurations, logs, evaluation evidence, downloaded dependencies and environment metadata are retained inside their archives.

The two Flora model directories remain available. Their eleven immutable weight shards now share PFS inodes after both former copies were independently hashed. Do not edit these frozen shards in place; the per-shard receipt gives both paths and hashes. This operation does not require a BOS restore.
