# Original Pi0 DSRL task57 replication pilot

Requested directly in this chat on 2026-11-01: test DSRL with `pi0_libero`
on LIBERO90 task57 and assess replication of the published result. The existing
Pi0.5 task57 experiment is complete and stays separate: native and final SAC
both scored10/10, which supplies no success-improvement replication evidence.

Review base: latest fetched `origin/main`,
`4e422587484312bfe34d623c5b498ab303a96be0`. Canonical main and its dirty
coordination file remain intact. Work is in detached `dsrl-pi0-task57`.

First prepare an isolated CPU runtime with authors' DSRL commit
`7f48937d4553e95244cd81c79236a3256df80597`, its exact OpenPI submodule
`a6d2400d2534ce32e7bdf8747709b97aaef8ec04` and LIBERO submodule
`87edbd1b823e464bc66557d213cc45e9f88aa543`. Download and hash the original
public `s3://openpi-assets/checkpoints/pi0_libero` weights and normalization
assets. Do not load models on macOS or consume GPU time for downloads.

One fresh seed0 pilot, capped at10 cumulative GPU-hours,1M simulator actions
and64GiB new output, including startup, qualification, evaluation and failures.
CPU preparation is one <=30minute,0GPU/4CPU/32GiB job; it is not a GPU launch
or evidence of learned behavior. No old unused grant is reused.

The intended scientific recipe is the original Pi0 H50/execute20, native
unconditioned `env.reset()` population on task57,400-step episodes, fresh
pixel SAC with256 batches, UTD20,10 critics, replay>500 query warmup, and
500k optimizer updates (whole-burst overshoot retained). Initial native uses
independent H50x32 Gaussian noise; learned evaluation uses stochastic SAC.
Evaluate10 episodes every10k optimizer updates and retain the full curve with
replay/query/primitive/optimizer units distinct. Match the original shared
training/evaluation environment and reset/RNG semantics wherever possible.

Add bounded execution and complete recovery around the original collection,
relabeling, SAC and evaluation functions without changing those functions.
Record any unavoidable runtime/protocol differences before launch. Qualify
the added recovery/accounting on allocated Linux, then prove actual allocated
CUDA/renderer binding, frozen motor and real episode/update advancement.

One seed is a replication pilot. The paper's reported roughly20%→100%
improvement at roughly10k online samples is a three-seed result; reproducing
that aggregate requires a separately scoped seed expansion after the pilot.
Sources: [released recipe](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/examples/scripts/run_libero.sh),
[paper](https://arxiv.org/html/2506.15799v2#S5.SS5).

Preparation update (2026-11-01 01:42 Shanghai): isolated runtime imports and
all9 Linux CPU tests now pass on source `ee2c65d3` in Baige
`job-no6xtx6vxax9` (33.75s). Eight initial tests passed in R5; its last
assertion compared MessagePack byte order rather than state values. R7
compares all recovered optimizer/state leaves exactly and the next update
numerically. It also retains each evaluation policy and pre-panel RNG.
Original S3 returns NoSuchBucket; official GCS has the original Pi0 LIBERO
checkpoint (19 files,12,014,131,888 bytes). Download/hash verification is
pending. Legacy S3 byte equality cannot be independently established.
CPU preparation uses separate immutable identities, with cumulative ceiling
1800 worker-seconds including failures; prior reservations990s plus R7<=780s.
GPU budget remains unused. Latest main reviewed before this qualification:
`89f578ab`; subsequent main changes are coordination and evidence only.
