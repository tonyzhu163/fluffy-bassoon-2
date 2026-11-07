# Restore the entire AnyViewPhys project

Read RECOVERY.json and its archive-parts manifest. Copy every archive_parts entry in the recorded order into fresh staging, checking each part byte count and SHA256. Concatenate those verified parts into anyviewphys.tar.zst and verify the complete archive_bytes and archive_sha256. The cold objects use32GiB parts; preserve their order from the manifest.

Extract the verified complete archive into a fresh PFS staging directory with Linux GNU tar/zstd, preserving ACLs, xattrs, numeric ownership and links:

```sh
anyview_restore_stage=$(mktemp -d /mowenjie/.anyviewphys-restore.XXXXXXXX)
tar --acls --xattrs --xattrs-include='*' --same-owner -I zstd -xf /path/to/verified-anyviewphys.tar.zst -C "$anyview_restore_stage"
```

Sync the fresh scratch filesystem before scanning. Decompress SOURCE-INVENTORY.json.gz and use the pinned tree.scan, tree.hash_rows(..., restored=True) and tree.verify to compare every staged path, regular-file SHA256/size, mode/owner/nanosecond mtime, symlink and internal hardlink topology. Ensure enough PFS capacity first. EXTERNAL-LINKS.json and EXTERNAL-RUNTIME-FINGERPRINT.json identify six shared Python interpreter links resolving to Python3.11.17 and3.12.8 under /root; qualify restored runtimes and accelerators before compute.

If /mowenjie/anyviewphys remains only its expected two-file recovery stub, preserve that stub under a fresh name and atomically rename the verified staged anyviewphys directory into the original path after confirming no readers/jobs. Never overwrite newer work. The original repository was clean at834804455987004b3ca69d1dac91ccfc49288871. QUIESCE.json records the already-paused scoring controller: prepared plans and CPU inputs were preserved, scoring had not started; reconcile its original admission/hold and create an explicit continuation before scoring. Do not automatically restart it merely because the files return.

Keep the cold parts, complete manifest, inventory and receipts until a later explicit retention decision. This archive preserves project files, not base OS/drivers or process memory.
