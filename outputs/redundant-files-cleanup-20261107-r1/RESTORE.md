# Restore a released redundant file

Read SOURCE-INVENTORY.json and RELEASED.json for the exact original path, metadata and SHA256. All selected original bytes have a retained cold recovery copy.

- `verified-cache.tar.zst` is byte-identical to the original cold archive named by PLAN.json (`existing_cold_archive`, expected SHA256 `11a3e40fbb4103e8011a64dd20fe1f2ffef20b1969f50e1cc8558272bef3097b`). To restore this redundant download, copy that cold object to a fresh temporary file, hash the complete file, then place it at the recorded original path only if that path is still absent. The valid extracted motor was retained in place.
- The object ZIP, incomplete model download and selected deployment bundles are preserved in `selected-files.tar.zst` under this run's private BOS archive directory. Verify the full object SHA256 from RECOVERY.json, extract with GNU tar/zstd into a fresh empty Linux staging directory, and compare exact member coverage, SHA256, mode/owner/mtime and hardlinks against the selected source inventory (excluding the already-cold duplicate archive).
- Restore only the missing file(s) from that verified staging tree to their recorded original paths. Stop if a destination already exists; do not overwrite changed files or an active reader's inputs. Preserve the recorded metadata and hardlinks. Recheck the final file SHA256.

The old incomplete model download remains incomplete even when restored. Use the separate retained valid motor for model work. Keep simulator assets, configurations, compact download receipts and unselected bundles unchanged. Record any restaging in its consuming run before launch.
