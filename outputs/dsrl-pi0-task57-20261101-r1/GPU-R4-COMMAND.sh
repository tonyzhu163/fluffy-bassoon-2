set -Eeuo pipefail
ulimit -c 0
TASK_BASE=/mowenjie/fluffy-bassoon-2/worktrees/dsrl-pi0-task57-20261101-s0-r3
TASK_CODE=/mowenjie/fluffy-bassoon-2/worktrees/dsrl-pi0-task57-20261101-s0-r4
TASK_SOURCE=00eca3fae5cb8052cee445f87359e26aa18e1a17
timeout 60 git -C "$TASK_BASE" fetch https://github.com/tonyzhu163/fluffy-bassoon-2.git "$TASK_SOURCE"
test "$(git -C "$TASK_BASE" rev-parse FETCH_HEAD)" = "$TASK_SOURCE"
git -C "$TASK_BASE" worktree add --detach "$TASK_CODE" "$TASK_SOURCE"
cd "$TASK_CODE"
bash scripts/diagnostics/dsrl_pi0_replication.sh "$TASK_SOURCE" dsrl-pi0-task57-20261101-s0-r4 /mowenjie/fluffy-bassoon-2/outputs/dsrl-pi0-task57-20261101-compiler-r11 660 400
