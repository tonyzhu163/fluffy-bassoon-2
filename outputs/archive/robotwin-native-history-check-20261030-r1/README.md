# Native history trainer and deployment qualification

Qualified code: `1aeea4f9fcc307bb147def92374bb81a3e2cb50e`.
Baige CPU job `job-tmocet3s5ur4` succeeded on 2026-10-30 at 21:55:53
Shanghai. It requested one worker, zero GPUs, four CPUs, 16 GiB RAM,
2 GiB shared memory and a five-minute maximum. Scheduler running time was
110 seconds; the pod duration was 99 seconds. Automatic retry was disabled.
`scheduler-success.png` preserves the authenticated console receipt.

The exact clean detached code was deployed through a Git bundle because the
server's configured origin did not expose this feature branch. No collector
worktree was modified. The attached `run.sh` and `smoke.py` are the executed
harnesses; their script hashes were checked after transfer.

## Results

- 46 focused exporter/history/PPO/SAC tests passed in 9.30 seconds. This
  includes fixed-policy validation rejection for AD, actual teacher replay,
  source-only normalization, paired full/static inputs, portable frozen
  deployment and exact optimizer/sampler/RNG continuation after interruption.
- Re-exported the same frozen SAC five-round source prefix: 1,539 decisions,
  120 episodes, 30 complete K4 trials, 47,602 physical source actions.
- Exported the complete PPO iteration-10 source-validation panel separately:
  340 decisions, 24 episodes, six K4 trials, 10,552 validation actions. The
  93,851 committed source actions in its anchor are not validation actions.
- Both DPT datasets use the same frozen SAC five-round teacher. AD uses
  actual physical actions; its validation objective is explicitly BC.
- All four AD/DPT × full/static integration fits completed two updates each
  at toy width8/layer1/head2/memory8, batch1/window4. Their model files loaded
  through the portable loader. Each frozen wrapper completed two sequential
  physical-episode lifecycle calls without changing weights.

There were zero new simulator actions, zero frozen-VLA queries and zero GPU
seconds. The four tiny fits performed eight smoke SGD updates total. These
are integration checks, not production-size fitting or robot evaluation.
Deterministic teacher competence and evidence for ICRL remain unmeasured by
this check. No source collection restart or GPU fitting sweep was launched.

## Immutable handoff

Remote root:
`/mowenjie/fluffy-bassoon-2/outputs/robotwin-native-history-20261030-r1/`.

| Artifact | SHA256 |
| --- | --- |
| `train/index.json` | `4e8d4e64eb1845a234f710342f168d3a4cdb86366d64dd6c7cef6c7c5895a4af` |
| `validation/index.json` | `194ae0fd4588e7bd55c3cd7a0fc6eedfb010bcb8043e311eeedd8adff0b2472b` |
| source prefix manifest | `05680fb416e6b8031f04b9832e903906ea2c4cb3ee091499bf4c65eb54d4c37b` |
| source and teacher SAC recovery | `f2fb39571cc3d0e809dfdc454f7f6df7f48e585fc972fd4c2fb8c83d36e63a42` |
| PPO validation source recovery | `62a6f92819c1df3eaa1614fd0052e999f496fe23ba5f7944bf7a722ef4a0ffb9` |

`receipt.json` contains exact portable checkpoint hashes, counters, script
identity and per-arm smoke results. `tests.log` preserves the benign existing
Dash pytest-hook deprecation warning alongside the successful test count.

The execution owner has the concrete first-fit recipe in
`docs/robotwin-history-policy.md`: AD/DPT × full/static, seed0, native
width128/two layers/four heads/memory300/window128, batch8, AdamW LR3e-4,
gradient cap1, 100 updates and fixed25/100 exports. The 240-second example
is a per-invocation operational cap, not a measured throughput guarantee.
The owner reserves actual total compute, probes its GPU allocation, records
timing and any partial endpoint, and controls any real fitting launch.
