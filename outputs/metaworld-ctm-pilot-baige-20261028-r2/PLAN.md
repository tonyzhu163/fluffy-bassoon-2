# Meta-World admission recovery

Continue the [r1 bounded pilot](../metaworld-ctm-pilot-baige-20261028-r1/PLAN.md)
with the same six cells, settings and one-GPU/180-minute scheduler limit.
The r1 allocation stopped at the GPU selector check before any model or
simulator work. Initialize an unset `CUDA_VISIBLE_DEVICES` to container-local
ordinal zero while preserving Baige's scheduler-provided NVIDIA allocation.
Keep all count, CUDA, PCI and PID admission checks.

Deploy the corrected commit through Git into a fresh r2 detached worktree.
Record its exact SHA in the job command and all run configurations. The r1
worktree and failed receipts remain immutable. No scientific or resource
scope expansion is made by this recovery.
