# Native π₀.₅ action-head co-training

The question is whether changing an existing action-expert weight jointly with a causal history branch improves source imitation and native cross-arm performance over either intervention alone. The trainer fits the four conditions in the [registered plan](../PLAN.md). Evaluation loads the same stock checkpoint once and uses the [locked deployment API](API.md).

| Condition | Effective stock `Pi0.action_out_proj.kernel` | History offset at every denoising step | Trainable scalars |
| --- | --- | --- | ---: |
| Original | Stock BF16 kernel | Zero | 0 |
| Policy only | BF16(W₀ + A B), rank ≤4 before rounding | Zero | 4,224 |
| Adapter only | Stock BF16 kernel | 52→16→320, tanh, bias-free | 5,952 |
| Joint | BF16(W₀ + A B) | Identical adapter architecture and initialization | 10,176 |

The stored factors represent a dense change to the actual 1,024 × 32 projection. Its bias and all upstream parameters stay frozen. Mathematically the unrounded change has rank at most four. Casting the merged matrix back to the deployed dtype can increase the rank of the realized difference. The history branch reads four completed transitions in the [source format](../data/API.md), converts its metric fields to centimetres, and emits a 10 × 32 velocity offset. A zero history yields exactly zero offset. Aligned and lagged values are supplied by the evaluator. The current state, action noising, frozen transforms, action decoding and ten-step Euler sampler are stock.

## Fit contract

The recovery engine differentiates the pinned `Pi0.compute_loss(train=True)` separately with respect to the enabled parameters in each condition. All three calls execute in one compiled per-example graph. Each call receives a newly reconstructed NNX model, the same frozen parameter tree, observation, expert action and random key. Only deterministic public input transforms are reused. Every visit runs OpenPI's own image augmentation and noising. Its flow target is ε − a at xₜ = t ε + (1 − t) a, with t sampled as 0.999 Beta(1.5, 1) + 0.001. The 32-coordinate native loss supplies every optimizer gradient. At scoring, the program calls the same native loss for all four conditions, with augmentation disabled for held-out rows.

Each of the 192 train and 64 held-out windows receives two fixed independent draws, keyed exactly as in gpu01. The trainer verifies both HDF5 files and rejects overlapping demonstration tapes. One singleton forward per condition and example preserves the native per-example PRNG semantics. It sums FP32 factor gradients across the same shuffled set of 32 windows, divides by the actual count, and takes one Adam update at 0.001. Eight epochs on 384 draws give 96 updates per fitted cell. The shared seed is 20261124. The program writes the draw-to-trajectory mapping and each epoch's permutation digest. Held-out scoring uses the final weights.

Deployment materializes each condition's BF16 `action_out_proj.kernel` before sampling. For a zero history offset it runs the stock native sampler directly on that weight tree. For nonempty history in the fitted history conditions it computes the offset once and adds it at every native denoising step. The parity receipt reports decoded actions from ordinary stock, an independent materialized no-op kernel and the earlier fused zero-factor graph. Only the materialized kernel path supplies the no-op deployment control. The fused graph's numerical deltas are diagnostic.

`fit.json` gives 32-dimensional native MSE, normalized velocity MSE on the first five issued-step coordinates 1–7, per-trajectory MSE and paired source uncertainty from 2,000 trajectory-level bootstrap replicates. It records the effective kernel and bias hashes, factor hashes, checkpoint identity, reloaded no-op parity, actual device and process memory. The held-out trajectory count is small. Source scores cannot establish native transfer. The [evaluation panel](../evaluation/PANEL.md) defines the separate native test.

## Baige Distributed Training

Clone the admitted training job **job-9v3iytmhh9ig** in Baige 分布式训练. The qualified RTX 4090 placement uses cluster `aihc-sw5hua6i3hyv`, queue `aihcq-chqrelndaaj4`, one worker/master, one GPU, 12 CPUs and 64 GiB RAM, with the existing CUDA driver image and project share mounted at `/mowenjie`. Request **90 minutes**, zero retries. An equivalent exclusive single RTX PRO 6000D allocation on its corresponding queue is supported. The parent owns submission and checks placement. [Source template receipt](../../frozen-vla-source-view-20261123-r1/SCHEDULER.md) · [Queue identity](../../frozen-vla-multiarm-confirmation-20261123-r1/FINAL-NATIVE.md)

Run the following command from the deployed clean checkout, using the SHA and absolute launcher from the parent's `recovery-ready.txt`. The binding to the digest below prevents substitution of another source selection.

```bash
set -eu
run=/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-cotraining-20261124-r1
pin=$(git rev-parse HEAD)
here=$(git rev-parse --show-toplevel)/outputs/frozen-vla-cotraining-20261124-r1/model
bash "$here/launch.sh" "$pin" gpu02 "$run/data/dataset-v1" 05209732b5e3f04308c8acbef2f6c8c8a2f173673840fdb055e1b094ff7d6f40
```

The launcher verifies its checkout, the source manifest, exclusive device and model type. It wraps the whole process group in a 5,340-second timeout with 10-second kill grace. The scheduler enforces 90 minutes per attempt and the parent tracks the six-GPU-hour training-stage cap. A fresh job writes under `$run/model/gpu02`. The training stage makes no simulator calls. Both cameras use the same `to_openpi_observation` conversion as the registered live policy.

| Dependency | Exact location or version |
| --- | --- |
| Training interpreter | `/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python`, Python 3.11, JAX/JAXlib 0.5.3, Flax 0.10.2, Optax 0.2.4, NumPy 1.26.4, h5py 3.13.0, TensorStore 0.1.74 |
| Pinned OpenPI | `/mowenjie/fluffy-bassoon/third_party/openpi-215abfb` at `215abfb217dbac7d5f1273282331b9b1866c0479` |
| Stock JAX checkpoint | `/mowenjie/fluffy-bassoon/inputs/pi05-libero-openpi-20261027-r1`, verified against all 12,439,085,481 bytes in `READY.json` with SHA-256 `ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170` |
| Fixed action/state normalization | `assets/physical-intelligence/libero/norm_stats.json`, SHA-256 `b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84` |
| Tokenizer and import overlay | `/mowenjie/fluffy-bassoon/cache/openpi-215abfb` and `/mowenjie/fluffy-bassoon-2/setups/xemb-icml-campaign-20261121-r1/cv2-headless-overlay` |
| Exclusive training probe | Tracked `scripts/[train]gpu_probe.py`, `--expected-count 1 --purpose training --baige-exclusive-training`, using `/mowenjie/fluffy-bassoon-2/setups/robocasa365-native-20261101-r1/env/bin/python` |
| Data/converter | Pinned checkout's `data/source_data.py` and `outputs/libero-xemb-ur5e-sysid-20261120-r1/ur5e_motor.py`. Their exact file digests are written into `config.json`. |

Check `native-parity.json`, then the first advancing optimizer step in `console.log`. Each completed global batch prints the measured rows per second, elapsed time, three losses and step number. The first batch includes compilation. `optimizer.msgpack` is atomically replaced after each update with all factors, Adam moments, batch cursor and accumulated receipts. The immutable data, seeds and deterministic permutation recover the remaining samples. A fresh attempt accepts a previous direct-engine attempt directory as its fifth argument. The trainer publishes `fit.json` after scoring and deployment reload. The stage uses no simulator.

**Failed reference path.** GPU01 on RTX 4090, job `job-9v3iytmhh9ig`, passed device admission and exited 1 after 94 launcher seconds, before the first optimizer update. The cached `history_in` gradient differed from the native gradient by 7.7105442e−6 in L2 against a native norm of 1.0504765e−4 (7.34%). CPU agreement did not transfer to this GPU. The prior bound remains recorded per factor as a diagnostic. GPU02 uses direct native gradients irrespective of this comparison. The failed attempt produced no trained weights. Its PFS receipts are in `$run/model/gpu01/{console.log,gpu.json,exit.txt}`.

## Linux CPU qualification

Two Baige CPU cores (32–33), CUDA hidden, restored the stock checkpoint and executed the full native loss and ten-step stock/injected sampler on a seeded OpenPI fixture. The cached base loss matched the native per-token loss exactly. Each of the four enabled factors had nonzero native gradients at the perturbed point and zero measured difference from the cached gradients. The decoded stock and zero-update [10,7] actions matched bitwise. A nonzero kernel/branch changed the decoded chunk. This check took 177.3 seconds from checkpoint verification and peaked at 19,916,104 KiB RSS. The initial eager reference materialized an extra BF16 rounding boundary and disagreed with fused stock loss. Compiling the reference as it runs during fitting resolved it. [Full checkpoint receipt](evidence/native-cpu.json)

A separate test read only the real output projection from the checkpoint. With two synthetic [10,1024] feature sequences, its zero-initialized heads matched native projection output bitwise. An update changed the BF16 kernel and all four factors received gradients on the next pass. The zero-history joint path agreed bitwise with applying its merged kernel to the stock layer. [Restored layer receipt](evidence/projection-cpu.json)

To repeat the fixtures in the tabled environment, set `CUDA_VISIBLE_DEVICES=''`, `JAX_PLATFORMS=cpu`, `OPENPI_DATA_HOME` and the overlay as in the launcher, then run `taskset -c 32,33 "$python" qualify.py /fresh/path/projection.json` and `taskset -c 32,33 "$python" native_check.py /fresh/path/native.json`. The recovery job records cached/live comparisons on a registered training row and uses the direct engine for every update. GPU throughput, source scores and native reward are pending the parent's jobs.

The original gpu01 revision was `6785abc8323473b81973b9d69c93863bf523e4d0`. Its detached checkout is `/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-cotraining-20261124-r1/model/code-6785abc8323473b81973b9d69c93863bf523e4d0`. The integrated data reader is byte-identical to the source worker's `97bedc86` revision. The shell syntax, source manifest, converter and training probe were checked in that checkout. Its job receipts are under the sibling `gpu01/` path.

A third CPU fixture exercised the production optimizer and scorer on 37 synthetic hidden sequences using the real stock projection. It covered a full batch and a five-example remainder, each intervention, per-trajectory scoring, serialization and recovery of the fixed final epoch. All recovered parameter hashes and epoch records agreed bitwise with the uninterrupted fixture. The experiment's held-out scores come from released demonstrations on the distributed job. [Optimizer fixture receipt](evidence/optimizer-cpu.json)

## Recovery deployment

The parent submitted `gpu02` as `job-7hmrgbcahi6l` on one 4090 at revision `29e9e5e5209db26ce0cbf695387c64cdf79b98fa`. The scheduler maximum recorded in the parent receipt is 150 minutes. The checked-out launcher's independent 5,340-second timeout and 10-second kill grace bound this attempt to under 90 minutes. The parent owns the six-GPU-hour training ledger. [Parent scheduler record](../EXECUTION.md)

That revision evaluates three directly differentiated singleton OpenPI losses in one JIT and accumulates only each cell's enabled factors. In a two-draw CPU test, it exercised all three actual optimizer paths, the partial batch, a bitwise checkpoint recovery and four-way direct scoring. The decoded chunks differed across the four fitted/control parameter sets on the source window. The ordinary stock and independent materialized-kernel no-op outputs agreed. This bounded test used the real projection, source camera transforms, recorded actions and native Pi0 loss/sampler code with lightweight upstream embeddings. The earlier full-stock CPU matching test used the original engine. [Launched revision's CPU receipt](evidence/direct-cpu.json)

A separate staged revision, `9ae699829359c7949fe417a0167706d8e7ca21f6`, additionally vmaps the directly differentiated loss over the three parameter sets, with fixed zero blocks for inactive factors. Its two-draw CPU test produced identical initial native loss in all three cells and passed optimizer recovery and sampler checks. The parent launched the preceding revision. The batched revision has no GPU throughput or performance result. [Batched CPU receipt](evidence/batched-direct-cpu.json)
