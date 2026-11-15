# Restore a fluffy-bassoon output from BOS

Run this inside Baige with the project PFS and its BOS mount. In [VERIFICATION.json](VERIFICATION.json), find the original path under `source` and its corresponding `bos_receipt`. A small cold pointer also remains beside each original path.

```bash
archive_root=/mnt/bos/adgfyl/strong-plant-archives/fluffy-bassoon-retention-20261115-r1
group=ctm-carry-s1001-qualification-lanes8-1
python3 "$archive_root/restore_payloads.py" "$archive_root/$group/001.receipt.json"
```

This verifies the cold object and atomically publishes the restored file on PFS after its staged SHA-256 matches. It accepts an already restored file with the expected hash and checks for collisions. Ensure sufficient PFS space for the selected objects. Restore all four `mikasa-carry-download-part-*` objects if the original split archive is needed. The assembled carry policy stays at the location recorded in the verification receipt.
