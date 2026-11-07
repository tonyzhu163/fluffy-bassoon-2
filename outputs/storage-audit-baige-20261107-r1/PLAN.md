# Baige storage audit — 2026-11-07

Read-only metadata census requested by the user. Scope: all accessible /mowenjie projects including hidden files, current development-container overlay, mounted storage capacity, inode pressure, deleted-open files visible from the development container, large files, hardlinks, model/cache duplication, and current job dependencies. No payload deletion, migration, training, or GPU allocation. Audit receipts are saved locally; temporary remote receipts use /tmp to avoid consuming PFS headroom.

Count allocated blocks rather than apparent sizes; count shared hardlinked regular-file inodes once. Do not follow symlinks or cross filesystems. Secret/chat directory contents are not opened; expose only aggregate byte totals. A live scan is not an atomic snapshot; report timestamps, errors, and any discrepancy from filesystem usage. Cleanup candidates need separate dependency and recovery checks before removal.
