# Native PPO pilot preflight — 2026-10-31

This is the historical 13:00 preparation snapshot. Subsequent allocated CPU
checks and the current queued status are recorded in `CPU-RESULT.md`.

The public RLinf PPO reference is staged at
`/mowenjie/fluffy-bassoon-2/inputs/pi05-robotwin-ppo-adjust-bottle-1223d8b9-20261031-r1`.
Revision `1223d8b9c387eb22c51350900d842d0f3b86cf14` contains 17 files and
8,529,290,042 published file bytes. All LFS SHA256 and small-file Git blob OIDs
passed. The fresh download used no credentials and constructed no model.
Its READY.json file SHA is
`52ebf3a2f84cba5317fbc0bd9bed1e5c37c338309bb532acda1e2dd9c2235a39`;
the copied receipt is `released-ppo-READY.json`. Its own RoboTwin normalization
SHA `bff5fcadaba626c28c1c2350d3baa7f1a04befc9f8b73f3e558a09c1b9c6ee52`
equals the existing SFT initializer's normalization. SFT fa8 remains untouched.
The [public checkpoint repository](https://huggingface.co/RLinf/RLinf-Pi05-RoboTwin-PPO-adjust_bottle)
establishes this released reference; it does not establish VLRLA author identity
or a matched published evaluation protocol.

The historical native GRPO environment path has been deleted. The retained
executable is
`/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python`,
Python 3.11.17, whose shared-runtime .pth points to
`/mowenjie/fluffy-bassoon/envs/maniskill-pi05-rlinf-e1a82ebf-20261028-r1/lib/python3.11/site-packages`.
Filesystem package metadata confirms Torch 2.11.0+cu128, Ray 2.59.0,
RLinf-OpenPI 0.1.1 and SAPIEN 3.0.1; the local MPLib 0.2.1 overrides shared 0.1.1.
These are metadata findings. Current native FSDP-PPO imports and changed entry
tests still require the allocated CPU qualification; no import/test/model ran
on the development host. The ordinary source100 CPU-bank import firewall is
not part of this native path.

The current Pro6K queue shows 11 free cards in aggregate, with no node meeting
`free GPUs >=8`. A node meets the transparent four-GPU fallback:
`aihcn-o4krfblc7lo5`, four free RPBZZZ6 cards, 111 free CPU cores and 1288GiB
free RAM. These are dated availability observations, not health/isolation
qualification or a node reservation. Both filter screenshots are retained.
The official accumulation algebra gives global batch 2048 at microbatch 32
using 8 ranks×8 accumulation or 4 ranks×16 accumulation. The original
mean-of-masked-microbatch-means normalization is retained; no bitwise or
identical-rank-shuffle claim is made.

PFS free space was 1,156,851,957,760 bytes. The download write probe passed;
64GiB new phase storage, a separate 128GiB concurrent reserve and a 500GiB free
floor fit. The released PPO files occupy approximately 7.94GiB of the new
phase's storage allowance. Latest fetched main remains
`acca706d5e75d1da099ff7ce6fddd727dace0fba`; official RLinf source is clean at
`c605b9d55ed93a180119b099e6f836abff293b08`.

Only a new, unsubmitted 0GPU/8CPU/64GiB/30-minute CPU form has been prepared.
It has a safe exit-2 placeholder pending the committed native test launcher.
No CPU or GPU allocation has been submitted in this preparation. The root's
separate GRANT.json fixes the fresh 6GPUh/1M-action/64GiB pilot. The optional
released PPO evaluation cannot gate primary before/after native PPO training.
All pilot outputs remain separate from source6/SAC100/AD banks.
