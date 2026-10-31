# Stage A Baige launch handoff to `frozen-vla-poc`

2026-11-01. The plan owner (this Claude session) has no Baige access. Under
the coordination rule that `frozen-vla-poc` is the sole LIBERO launcher, this
hands one Stage A GPU run to that owner. No job has been submitted.

## What is ready

- **Code:** clean `main` commit `4ec35eb9f563813e43293f797e8141a12fb0c4be`. The
  driver, launcher and tests are unchanged from `5e90a29`, which passed thu2
  CPU job 49931 (43/43; [receipt](CPU-QUALIFICATION.md)).
- **Grant:** [GRANT.json](GRANT.json), SHA-256
  `893c368f8285bf18eed28d414bacfa2a45ea1ee6d4b459db49ff744823e987d3`. It allows
  10,800 GPU-seconds, 170,000 simulator actions and 20 GiB of outputs for one
  GPU with no automatic retry. Run ID `libero90-pi05-headroom-screen-20261101-s0-r1`.
  It falls under the standing ≤10 GPU-hour instruction; the human requested
  Stage A on 2026-11-01.
- **Priority:** this belongs to the DSRL-first track: it picks the task for a
  pi0.5 DSRL replication. It needs one GPU for ≤3 hours. Whether it runs before
  or beside the queued native PPO pilot is the launcher's or the human's call,
  depending on free GPUs.

## Steps on Baige

1. Deploy a clean detached worktree at the exact commit. If GitHub is
   unreachable, use a git bundle.
   ```bash
   cd /mowenjie/fluffy-bassoon-2/repo && git fetch origin main && git worktree add --detach ../worktrees/libero90-pi05-headroom-screen-20261101-s0-r1 4ec35eb9f563813e43293f797e8141a12fb0c4be
   ```
2. Copy the grant out of `main` into the plan folder. This folder is separate
   from the run output. The grant was committed after the source commit, so
   it is not in the worktree. Confirm the hash and that the run output does
   not exist yet:
   ```bash
   mkdir -p /mowenjie/fluffy-bassoon-2/outputs/libero90-pi05-headroom-screen-20261101-r1 && git -C /mowenjie/fluffy-bassoon-2/repo show origin/main:outputs/libero90-pi05-headroom-screen-20261101-r1/GRANT.json > /mowenjie/fluffy-bassoon-2/outputs/libero90-pi05-headroom-screen-20261101-r1/GRANT.json && sha256sum /mowenjie/fluffy-bassoon-2/outputs/libero90-pi05-headroom-screen-20261101-r1/GRANT.json && test ! -e /mowenjie/fluffy-bassoon-2/outputs/libero90-pi05-headroom-screen-20261101-s0-r1
   ```
3. Check capacity and do a write probe on `/mowenjie`. Need at least 20 GiB
   for this run, plus whatever concurrent runs reserve.
4. Submit one console job: 1 GPU, 16 CPU, 96 GiB RAM, 8 GiB shared memory,
   ~185-minute cap, no automatic retry. Command:
   ```bash
   bash /mowenjie/fluffy-bassoon-2/worktrees/libero90-pi05-headroom-screen-20261101-s0-r1/scripts/diagnostics/vla_headroom_screen.sh 4ec35eb9f563813e43293f797e8141a12fb0c4be libero90-pi05-headroom-screen-20261101-s0-r1 /mowenjie/fluffy-bassoon-2/outputs/libero90-pi05-headroom-screen-20261101-r1/GRANT.json 893c368f8285bf18eed28d414bacfa2a45ea1ee6d4b459db49ff744823e987d3 10740
   ```
   Inside the allocation the launcher:
   - runs the GPU probe with `--expected-count 1`;
   - checks the motor's parameter signature before and after (`c565dd3f…`);
   - replays the same noise twice and requires matching actions;
   - runs the 384 planned episodes;
   - writes `rates.csv`, `SELECTION.json` and `index.json` under `experiment/`.

## Checks after launch (every 20 minutes)

- `experiment/episodes.jsonl` row count advancing (expect about 14 GPU-seconds
  per episode, so ~1.5 hours in total);
- `index.json` action charge within 170,000;
- `exit-code.txt` and `worker-time.json`.

When it finishes, record `SELECTION.json`: a selected task and runner-up, or
a stop with the full rate table. Report the GPU-seconds, actions and storage
used.

If the run is interrupted, diagnose the cause first. Then resume the same
run ID with `--resume` appended. Completed episodes are skipped and seeds are
fixed. Don't start a new run ID just to get more time.

## What this does not authorize

Stage B (the replication run) needs its own reviewed task parameterization and
its own grant. Do not start it from `SELECTION.json` automatically.
