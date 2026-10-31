set -Eeuo pipefail
TASK_BASE=/mowenjie/fluffy-bassoon-2/worktrees/dsrl-pi0-task57-20261101-s0-r2
TASK_CODE=/mowenjie/fluffy-bassoon-2/worktrees/dsrl-pi0-task57-20261101-s0-r3
TASK_SOURCE=17d5924ba6f66a3754355d23732c6d6f755e892a
timeout 60 git -C "$TASK_BASE" fetch https://github.com/tonyzhu163/fluffy-bassoon-2.git "$TASK_SOURCE"
test "$(git -C "$TASK_BASE" rev-parse FETCH_HEAD)" = "$TASK_SOURCE"
git -C "$TASK_BASE" worktree add --detach "$TASK_CODE" "$TASK_SOURCE"
cd "$TASK_CODE"
export XLA_FLAGS='--xla_gpu_triton_gemm_any=True'
bash scripts/diagnostics/dsrl_pi0_replication.sh "$TASK_SOURCE" dsrl-pi0-task57-20261101-s0-r3 /mowenjie/fluffy-bassoon-2/outputs/dsrl-pi0-task57-20261101-policy-r10 420
