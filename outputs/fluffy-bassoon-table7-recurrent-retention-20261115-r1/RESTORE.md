# Restore Table 7 or recurrent flow on Baige

The [verification manifest](VERIFICATION.json) records each original PFS path under `source` and its corresponding BOS receipt under `bos_receipt`. Look up the files the run needs, then pass each distinct receipt to the restore helper from the same archive batch.

For the Table 7 r3 step 10,000 recovery, restore the model and the archive part containing its optimizer, sampler and RNG state.

```bash
bos=/mnt/bos/adgfyl/strong-plant-archives
r1="$bos/fluffy-bassoon-table7-recurrent-retention-20261115-r1"
python3 "$r1/restore_payloads.py" "$r1/table7/006.receipt.json"
python3 "$r1/restore_payloads.py" "$r1/table7/007.receipt.json"
```

Part 007 also restores the smaller Table 7 checkpoints and trainer states. Restore part 001 for the r1 initializer used by slim checkpoint exports. Parts 008, 009 and 010 restore the recurrent-flow initializers for seeds 1000, 1001 and 1002. To restore the checkpoint data for the six recurrent-flow training runs, use the second batch.

```bash
bos=/mnt/bos/adgfyl/strong-plant-archives
r2="$bos/fluffy-bassoon-table7-recurrent-retention-20261115-r2"
python3 "$r2/restore_payloads.py" "$r2/recurrent-flow-checkpoints/001.receipt.json"
```

Run these commands where the Baige PFS and BOS mount are available. The helper verifies the content hash and stages each file before publishing it at its original path. It accepts matching files and refuses to overwrite different files. Restoring multiple original paths from a shared BOS object creates separate PFS copies, so leave enough space for the paths listed in each selected receipt.
