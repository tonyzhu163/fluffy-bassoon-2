# Restore a cold experiment bundle

Every reclaimed bundle has an adjacent `<run-id>.bos-archive.json` pointer on Baige. It records the BOS archive, SHA256, size and exact restore command. Full configurations, logs, evaluations and checkpoints are inside the archive; full file inventories and receipts remain on both PFS and BOS.

Confirm the destination is absent and no job will write to it. Read back the archive and verify its SHA256 against the receipt, then run the recorded command on allocated Linux resources:

```sh
tar --ignore-zeros --acls --xattrs -I zstd -xf '<archive from receipt>' -C /mowenjie
```

`--ignore-zeros` is required: each archive contains four concatenated GNU tar streams in Zstandard frames. Omitting it restores only the first stream. Qualification restored all members, hardlinks, symlinks and literal filenames, and detected same-size content corruption with unchanged mtime.

Hardlinks within each bundle are preserved. Restoring separate bundles can create separate copies of files that previously shared an inode across bundles; allow capacity for those copies.

Archival compares every member with the source, checks the source stayed unchanged, and verifies a full BOS SHA256 readback before removing any source. The final manifest reconciles receipts, pointers, archive sizes, hash sidecars and inventory hashes. Concurrent job writes affect filesystem free space; summed source allocation is archived payload, not an exact attribution of physical space reclaimed.
