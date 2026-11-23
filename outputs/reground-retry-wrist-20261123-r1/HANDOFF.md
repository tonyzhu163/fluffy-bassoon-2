# Released Panda retry with a separate passive support

**Parent launch target** is the standalone Git commit `ac28987f221e336cef2ea64dc16dfffc9f0e74c9`. Its detached checkout and bundle are on Baige under the run's own directory. The three historical 1,400-assignment arms stay in their original reference store. The scientific output of this run starts empty. This implementation worker used CPU only.

## Cases and interpretation

The frozen arms get exactly the same support tape and third seed-41 estimate. Each test start has one preassigned support on the same task, with a different initial object layout. The frozen policy is queried under native Base on that support. After controls 5, 10 and 15, the archived reader gets only its public camera, EEF, jaw and last five issued commands. The retries freeze that estimate. Both use the original streaming adapter's canonical policy-input mapping anchored at the query's settled EEF. Only the wrist arm multiplies the bounded angular delta by the inverse estimated yaw before the native shift. This is a planar rotation of angular XY, with angular Z unchanged. Both arms retain the identical translational gain and gripper rule.

Label 0 covers all 280 unique test task–start pairs. The five released seed labels 0–4 share their frozen policy/reader and noise assignments, although closed-loop outcomes can vary. The 280 new pairs have 35 tests per task, two scored arms, 120 distinct support task–starts, and 280 freshly collected supports. The support ID is `(query_id - 15) % 15`. All 280 assigned bank pairs differ in object or fixture position. Even the smallest pairwise maximum root displacement was 6.45 mm. [Registration](PLAN.md) · [Native layout verification](evidence/layouts.json) · [Per-pair evidence](evidence/layout-banks.json.gz)

For one deployment, count 15 passive controls (0.75 simulated seconds at 20 Hz), two explicit task resets and 20 settle controls, plus the scored controls until native termination. For the complete two-arm physical experiment, count 280 support resets, 560 query resets, 4,200 support controls, 8,400 settle controls and the sum of the 560 scored trajectory lengths. The native query caps are 280 for spatial/object, 300 for goal and 520 for long tasks. Each fresh worker also initializes one simulator and calls its internal construction-time reset before the registered task reset. The opening capture piggybacks on the first scored arm. The second arm rechecks the actual native noise and uses that single shared proposal. Every later requery is a live native forward pass. There are zero dedicated probe commands, target labels or target fitting transitions.

The release's original seed-0 streaming arm succeeded on 81/280, including 35/35 on goal_1 and 0/35 on goal_2. The original five-label totals there were 171/175 and 0/175. Report gains and losses separately on the two goals. The new paired wrist-minus-translation contrast isolates the wrist correction. Contrasts versus historical streaming include the separate support, correction at query control zero, frozen calibration, and fresh reset history. Readback measures historical opening overlap, retains the original outcomes and denominators, and supplies the five-label within-start sensitivity. [Original result](../reground-rotation-reference-20261123-r1/RESULT.md)

## CPU evidence

- Numerical composition with the release's forward shift was checked at six positive, zero and negative angles, plus nonunit gains, grip signs, clipping, and unchanged non-wrist coordinates. Maximum unit-gain component error was 3.36×10⁻⁸. On native goal_2 start 15, ten nonzero rigid-motion commands with a known quarter turn reproduced all nominal 25-D states and speeds exactly, including a fresh nominal repeat. Leaving the wrist uncorrected gave a maximum 0.278 rotation-coordinate difference. Known angles are used in this standalone reciprocal test only. [Frame and simulator measurements](evidence/frames.json)
- A native four-lane batch crossing goal_1 and goal_2 completed both 300-control retries with a noise-driven policy fixture, real camera rendering, the released unsafe scorer and the frozen seed-41 reader. The native-spawn and preloaded-worker paths matched every raw/mapped opening leaf, full simulator state, noise tensor, all 50 proposals, issued support actions, entire scored traces and all non-timing native row fields. Their elapsed times were 267 and 92 seconds in separate CPU trials with varying shared load. [Native spawn](evidence/native01.json) · [Preloaded workers](evidence/native02.json) · [Independent comparison](evidence/transport-and-readback.json)
- The actual reference π₀.₅ strictly loaded on Baige CPU and emitted two live proposals for the 15-control support at goal_2 start 0. Seed-41 returned yaw 72.10°, gain 1.178 and grip +1. Both retries reset to goal_2 start 15. Their full native inputs, noise and opening proposal matched. The first ten proposed commands agreed, and the issued wrist components differed by up to 0.179. Each arm ran through its own live control-10 requery and one further control. Those two 11-control traces test integration only. [Real-policy CPU receipt](evidence/real-policy.json) · [Strict-load and simulator log](evidence/real-policy.log.gz)
- The paired statistics fixture joined all 1,400 historical labels and recovered the known 199/280 discordance against an all-success fixture. A separate end-to-end readback joined real CPU simulator rows to the historical key, nonce and robot reset. Interrupted and confirmed ledger events have separate counts. [Statistical and journal check](evidence/readback01.json)

The release candidate `0bfa5b087a2b5d5ba9d6ecebafd9db5288947186` also retained the four raw public support snapshots, actual support proposals/commands, speed trace and native support safety summary. Independent readback matched the stock four-lane fixture through every action and all 2,400 scored controls. The deployed pin changes only shell exit collection. A Baige CPU check exercised simultaneous exits and individual rank failures over 100 four-process trials. All returned the expected status. The raw CPU attempt directories and logs are in `/mowenjie/fluffy-bassoon-2/outputs/reground-retry-wrist-20261123-r1/cpu/` on Baige. [Candidate qualification](evidence/qualified-native.json) · [Trace and ledger readback](evidence/qualified-readback.json) · [Deployed launcher check](evidence/launcher.json)

## Source, environment and deployment

| Component | Pin |
|---|---|
| Worker code, complete independent Git bundle | `ac28987f221e336cef2ea64dc16dfffc9f0e74c9`, bundle SHA-256 `3fea50286ec91a06083de4f0aca7da95737a7ed24a0b2cd57078be5309c70178` |
| Main workspace reference reviewed | `757a44cd4374833cad790852df8b87ddf412e638` |
| ReGround upstream | `16a98486f560676b9036d1a6d067b9dd51a5917a` |
| Reference π₀.₅ / tokenizer / LIBERO assets | `7152f13e34c6c115d808aea0e5d08aca74eac14b` / `35e4f46485b4d07967e7e9935bc3786aad50687c` / `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` |
| Frozen streaming bridge / full-opening helpers | `cab855a03c8ff1a3f1926513369099bf11c4b859` / `c3efd5042a29c68248c07b2d6375d367df6378ef` |
| Archived transforms and estimator / reader bridge source | `ff2ddf0ab03e6fbe243979556d8d8f2ebb9bc29c` / `fd909227c2013cc58be1ced519b03ea1328199ea` |
| Seed-41 weights SHA-256 | `c12bace5cb391499d6abd7cc5767f94b1519de7d77940efb0cfcb71ca0f686a4` |
| Historical 1,400-row table SHA-256 | `8aa0fdc6eee00c69779bb43911746032e18a7dd951ecdacaea6774a3a3195263` |

The shared runtime, offline Hub cache, LIBERO banks, LeRobot 0.6.1 and archive JAX reader are the previous release reproduction's installations. Startup checks the reference checkout, upstream scientific code and 12 runtime receipt/activation digests. The reader independently checks its archive and weight digest. The released LeRobot policy must confirm strict weight loading. The Linux simulator workers use the original upstream worker RPC and simulator. A dedicated single-threaded Python import server amortizes startup across fresh processes. The server verifies it has no live CUDA or EGL context. Each simulator child creates its own context after the rank selects its EGL device. [Launcher environment](env.sh) · [Released runtime](../reground-rotation-reference-20261123-r1/RUNTIME.md)

The checkout below is already deployed. To reconstruct on a new mount with the same pinned runtime, transfer the supplied bundle with SCP, clone it into a fresh directory, and check out the pinned SHA. The commands verify the existing deployment as well.

```bash
# On the Mac, only if a new copy is needed
run=outputs/reground-retry-wrist-20261123-r1
pin=ac28987f221e336cef2ea64dc16dfffc9f0e74c9
root=/mowenjie/fluffy-bassoon-2/$run
scp "$run/worker.bundle" "baige-dev:$root/$pin.bundle"

# On Baige, on the shared project mount
run=outputs/reground-retry-wrist-20261123-r1
root=/mowenjie/fluffy-bassoon-2/$run
pin=ac28987f221e336cef2ea64dc16dfffc9f0e74c9
src="$root/src-$pin"
if [[ ! -d "$src" ]]; then
    git clone "$root/$pin.bundle" "$src" && git -C "$src" checkout --detach "$pin"
fi
test "$(git -C "$src" rev-parse HEAD)" = "$pin" && git -C "$src" diff --quiet HEAD
```

## Parent's 4090 launch

Reuse the Distributed Training queue, single master, system image and read/write `/mowenjie` mount from the completed 4090 reference job `job-hl6vn153l9d3`. Use four exclusive RTX 4090s, with the reference job's 40 CPU allocation and memory profile. One rank owns one CUDA/EGL device and four native environments. Set the scheduler wall limit to 90 minutes (6.0 GPU-hours at four cards); the command has a slightly shorter 5.83 GPU-hour payload ceiling. The older common-opening sensitivity jobs used 848–914 one-card seconds for 16 complete capture/base/streaming cases including cold loads. Scaling those gross rates to 280 gives 4.12–4.44 GPU-hours before differences in horizons, calibration and startup. This is a sizing reference. Confirm the new rate on the first sealed batches and charge the scheduler's actual gross allocation time.

```bash
root=/mowenjie/fluffy-bassoon-2/outputs/reground-retry-wrist-20261123-r1
pin=ac28987f221e336cef2ea64dc16dfffc9f0e74c9
bash "$root/src-$pin/launch.sh" "$pin" retry01 4 5250 full
```

The single allocation keeps its policy loads warm. Each rank processes disjoint four-case batches, first covering native batches 017–034. These contain all 70 goal starts plus long task 3 starts 48–49. Complete all 70 batches for the full inference. The optional `goals` phase stops after the first 18 batches. Use it only if a separate operational stop is needed. Each complete four-pair directory is atomically renamed into `scored/batches/batchNNN/`. Partial native journals, tapes and traces stay in `scored/attempts/ID/`. A later invocation under the same pin and a fresh attempt ID validates and skips sealed batches. An interrupted batch repeats its support and both arms. Size any subsequent wall limit from the remaining cumulative budget. The wrapper's ceiling is per invocation.

Check `scored/attempts/retry01/gpu.json`, `rank{0,1,2,3}.json`, rank logs and the first sealed batch. The probe must report the exclusive 4090 allocation. Each rank verifies CUDA name/UUID/PCI and maps EGL before its first native reset. Check advancing publication every 20 minutes. After the goals or full panel seals, run the readback on Baige CPU.

```bash
root=/mowenjie/fluffy-bassoon-2/outputs/reground-retry-wrist-20261123-r1
pin=ac28987f221e336cef2ea64dc16dfffc9f0e74c9
bash "$root/src-$pin/readback.sh" "$pin" full
```

`scored/readback-full/cases.json` retains every new row, its estimate, native safety components, original five-label outcomes and same-start opening comparison. `summary.json` contains all-task and task-specific means, paired success and unsafe differences, discordant counts, 10,000-draw 95% paired-start intervals and a support-layout cluster sensitivity. Rates and differences are proportions. Multiply by 100 for percentage points. Intervals condition on the eight fixed tasks and support assignment. The journal readback includes completed and partial work, per-stage reset/settle/control counts, true batched forwards and verified opening replays. Each production forward evaluates four slots, including native padding by ended lanes. The per-case useful queries equal the ceiling of scored controls divided by ten. Support unsafe and union-of-support-or-query unsafe are reported alongside the native query metric. [Analysis](summarize.py) · [Native runner](retry.py)
