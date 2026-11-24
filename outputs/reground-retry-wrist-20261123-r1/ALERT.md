# Native retry01 exited before the first rollout

**22:14 Shanghai, 2026-11-23.** Baige Distributed Training job `job-iynjhzjijoyj` returned launcher code **2**. All four distinct RTX 4090s passed exclusive CUDA admission. The driver failed to start. PFS contains **0/70 sealed batches**, no partial batch, no rank admission receipt and no physical interaction journal. The measured launcher interval was 22:12:43–22:14:01, or **312 GPU-seconds (0.0867 GPU-hours)**. Scheduler setup and teardown require the parent's allocation receipt. [Launch](evidence/retry01/launch.txt) · [Exit](evidence/retry01/exit.txt) · [GPU probe](evidence/retry01/gpu.json) · [PFS inventory and receipt hashes](evidence/retry01/pfs.json)

The submitted scientific pin was `0bfa5b087a2b5d5ba9d6ecebafd9db5288947186`. Its launcher exported the variable `here`. The real cache wrapper assigned that same exported variable to the reference `transport/` directory before running the inner rank shell. Ranks 0, 2 and 3 therefore asked Python to open the nonexistent `transport/driver.py`. Rank 1 left an empty log. The launch log also recorded an unbound `PS1` in the shell startup file. The approved shell revision `ac28987f221e336cef2ea64dc16dfffc9f0e74c9` sets strict options after shell startup and waits for individual ranks. A CPU replay through the real cache wrapper showed that all four ranks still received the wrong driver path with that revision alone. [Original launch log](evidence/retry01/launch.log) · [Rank 0 log](evidence/retry01/rank0.log) · [CPU negative control and correction](evidence/recovery-qualified.json)

## Qualified command for the parent

The prepared shell launcher passes the absolute driver path as a positional argument across the cache wrapper. It records its own SHA in the attempt. The environment, policy, reader, worker, validator and resulting batch contracts use the exact approved `ac28987f…` checkout. The operational checkout is pinned separately at `d6de8c725cfaf2dbb01a6a9c4895197441b8a6af`. Its Git diff from the approved code adds only the recovery launcher and its CPU qualifier. The launcher on PFS has SHA-256 `c46a5996c99f28b93c965873ed1fd7e40f58e782a972f9afaea1127bfaeb4041`. [Recovery launcher](recovery/launch.sh) · [Portable Git bundle](recovery/launcher.bundle)

On Baige CPU an isolated copy of the command exercised the pinned environment and real cache wrapper with a private output store and stand-ins for the GPU probe and GPU lane. All four corrected ranks received the pinned driver, the real driver processed the 1,400-row historical CPU fixture, and a forced rank-3 failure propagated code 43. The identical trial with the approved unmodified launcher returned 2. The recovered path and supervisor are CPU-qualified. GPU rank/EGL mapping, first real rollout and measured throughput remain open. [Qualification code](recovery/qualify.py) · [Pinned deployment qualification](evidence/recovery-qualified.json) · [Real CPU driver readback](evidence/recovery-driver-readback.json)

Use the following prepared command only if the parent elects to resubmit. This version assumes at most **5 minutes of gross four-card allocation for retry01** and an **85-minute scheduler limit for retry02**. Its 4,950-second payload leaves the handoff's 150-second scheduler margin. The job shape is four exclusive RTX 4090s, one master, 40 CPUs, the original memory/image profile and the read/write project mount. Nothing has been submitted or stopped by this monitor.

```bash
root=/mowenjie/fluffy-bassoon-2/outputs/reground-retry-wrist-20261123-r1
pin=ac28987f221e336cef2ea64dc16dfffc9f0e74c9
launcher_pin=d6de8c725cfaf2dbb01a6a9c4895197441b8a6af
launcher="$root/recovery-src-$launcher_pin"
test "$(git -C "$launcher" rev-parse HEAD)" = "$launcher_pin" &&
    git -C "$launcher" diff --quiet HEAD &&
    bash "$launcher/recovery/launch.sh" "$pin" retry02 4 4950 full
```

## Scarce-budget boundary

For planning, charge the two attempts together against the handoff's **6.0 gross GPU-hour** target. This is the conservative figure inside the registered 5–7 GPU-hour range. The measured 312 GPU-seconds leave **at most 5.9133 GPU-hours** on the six-hour target, or at most 6.9133 against the seven-hour outer limit. The parent’s scheduler receipt determines the actual remainder. Time in the submission queue gives no measure of GPU usage.

If retry01 used more than five gross minutes, let `G` be its elapsed wall seconds from four-card allocation to release. Its gross cost is `4 × G` GPU-seconds. The maximum whole-minute scheduler limit for retry02 is `floor((5400 − G) / 60)`. Set the payload to `60 × scheduler_minutes − 150` seconds, using at most the original 5,250 seconds. A nonpositive payload leaves no runnable budget under this convention. There is no measured production rate from which to guarantee completion. The historical sizing reference was 4.12–4.44 GPU-hours on a different workflow. [Registered budget and sizing](HANDOFF.md)

After the first new batch seals, run `bash "$root/evidence/verify-first.sh" retry02` on Baige CPU for independent native validation, then check publication every 20 minutes. On 70/70 publications, the standard CPU readback command uses `src-ac28987f221e336cef2ea64dc16dfffc9f0e74c9/readback.sh` with the same full SHA and phase `full`. The failed attempt directory remains available for the budget ledger. [Failure result](RESULT-NATIVE.md)
