# First-seed complementary RL screen

Date: 2026-10-29, Asia/Shanghai. Authorized by the ten-hour autonomous CTM/CMM
objective. Exact source 47a37b7ddc34de413623481999c63bc4915fd377; reviewed latest
main remains 65b48a457f9cf1c9c9dcd1c9813ac9e53d5e0455 at 00:00.

Admission: Baige job-1txxb3w3qvak / metta-bench-qualification-20261029-r1
completed exit0 at 00:06:52. 173 tests passed, 3 skipped. All eight real
updates independently audited for source, complete checkpoint hash, replay,
finite metrics, real interactions, active optimizer samples and parameter cap.
PointMaze privileged oracle 96/96; raw control means independently recomputed.
The source and executables are identical for this learning screen.

Allocate one RPBZZZ6, 16 CPU/64GiB, 8GiB SHM, eight persistent simulator
workers; one model process at a time. Scheduler hard cap 200 minutes; parent
screen budget 180 minutes with a 10-minute finalization reserve, each training
cell at most 30 minutes before final evaluation. No retry/deletion. The full
autonomy deadline is Unix1793237291 (2026-10-29 09:28:11 Shanghai). No new
cell starts near the deadline. Existing four ML10 jobs are untouched.

Run identities: metta-screen-baige-20261029-r1. Order: HalfCheetahVel,
PointMaze, ML1 Push; within each: GRU, Transformer-XL, persistent CTM, CMM.
Seed1000; parameter cap250k, width matched85–100%; T1; CTM trace32/bounded
synchronization/decay per observation, CMM trace32/LTM32, XL memory64. These
are architecture controls; GRU/XL are not relabeled as original method baselines.

HalfCheetah: 50lanes/K5/H100, target1M real steps, 40collections, 10PPOepochs.
PointMaze: 25lanes/K10/H500 maximum, target1.5M real steps, at most100collections,
5PPOepochs. Early successes end episodes; fully finished trials pad their suffix.
Masks exclude padding from losses, normalizer and environment counters. Every
real transition is sampled once per PPO epoch; wholly padded replay windows
are omitted. Actual optimizer steps therefore depend on real trial lengths.
ML1 Push: 50lanes/K5/H500, target2.5M steps/20collections/10PPOepochs, hidden
goal and fixed task instance. Before its training cells, one actual CMM ML1
update checks its interface/replay on this allocation.

All use cached100 TBPTT, minibatch10 trials' worth of windows, common state
bypass/64x64 actor-critic, native task rewards, full-trial GAE and saved source
contracts. HalfCheetah evaluates24lanes/K10 including target velocities[2,3]
separately; PointMaze12lanes/K10 on fresh and held-out cell-pair configurations;
ML1Push10lanes/K10. Final normal/reset/reward-blind are paired. Checkpoint every
complete update includes optimizer/sampler/normalizer/all RNG states.

A capped or absent cell is incomplete, never extrapolated or ranked as trained.
This stage is a single-seed screen, not a confirmed architecture win. Use its
measured costs and raw curves to choose confirmation seeds, separately trained
reset controls and extended-context checks. CMM paper-calibration state is not
freshly observable because the required school alias is unavailable; do not
claim qualified paper parity. VLA steering remains conditional on useful
held-out adaptation and a repeatable advantage on a requested claim axis.
