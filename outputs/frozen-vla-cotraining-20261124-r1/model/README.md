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

Training calls the pinned `Pi0.compute_loss(train=True)` directly on stock-normalized expert action chunks. A trace-local hook captures its terminal hidden features. Its internal sampler constructs xₜ = t ε + (1 − t) a with t sampled as 0.999 Beta(1.5, 1) + 0.001, and target ε − a. Stock preprocessing supplies random image augmentation. Held-out features use the same flow distribution and unaugmented images. Each of the 192 training and 64 held-out windows receives two fixed, independent draws. The two splits contain 21 and eight disjoint demonstration tapes respectively. The trainer verifies the split-file digests and rejects overlapping state/action tapes before fitting.

Adam at 0.001 runs for eight fixed epochs, batch 32, over the same shuffled 384 training tuples for each fitted condition. All 96 updates per condition consume the same ordered batches. A common seed of 20261124 initializes enabled factors. Each output factor starts at zero. The original receives no optimizer updates. The 128 held-out tuples are scored after the fixed last update. Exactness of the cache applies to the realized augmentations, times, noise, and immutable upstream network. Native deployment recomputes the full denoiser on the changing action chunk and injects the merged kernel and offset inside every stock step.

`fit.json` gives 32-dimensional native MSE, normalized velocity MSE on the first five issued-step coordinates 1–7, per-trajectory MSE and paired source uncertainty from 2,000 trajectory-level bootstrap replicates. It records the effective kernel and bias hashes, factor hashes, checkpoint identity, reloaded no-op parity, actual device and process memory. The held-out trajectory count is small. Source scores cannot establish native transfer. The [evaluation panel](../evaluation/PANEL.md) defines the separate native test.

## Baige Distributed Training

Clone job template **job-69hluhi5wgb7** in Baige 分布式训练. The qualified RTX 4090 placement uses cluster `aihc-sw5hua6i3hyv`, queue `aihcq-chqrelndaaj4`, one worker/master, one GPU, 12 CPUs and 48 GiB RAM, with the existing CUDA driver image and project share mounted at `/mowenjie`. Request at most **360 minutes**, zero retries. An equivalent exclusive single RTX PRO 6000D allocation on its corresponding queue is supported. The parent owns submission and checks placement. [Source template receipt](../../frozen-vla-source-view-20261123-r1/SCHEDULER.md) · [Queue identity](../../frozen-vla-multiarm-confirmation-20261123-r1/FINAL-NATIVE.md)

Run the following command from the deployed clean checkout, using the SHA and absolute launcher from the parent's `training-ready.txt`. The binding to the digest below prevents substitution of another source selection.

```bash
set -eu
run=/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-cotraining-20261124-r1
pin=$(git rev-parse HEAD)
here=$(git rev-parse --show-toplevel)/outputs/frozen-vla-cotraining-20261124-r1/model
bash "$here/launch.sh" "$pin" gpu01 "$run/data/dataset-v1" 05209732b5e3f04308c8acbef2f6c8c8a2f173673840fdb055e1b094ff7d6f40
```

The launcher verifies its checkout, the source manifest, exclusive device and model type. It wraps the whole process group in a 21,500-second timeout with 10-second kill grace. The scheduler's six-hour wall limit governs the allocation. A fresh job writes under `$run/model/gpu01`. The training stage makes no simulator calls. Both cameras use the same `to_openpi_observation` conversion as the registered live policy.

| Dependency | Exact location or version |
| --- | --- |
| Training interpreter | `/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python`, Python 3.11, JAX/JAXlib 0.5.3, Flax 0.10.2, Optax 0.2.4, NumPy 1.26.4, h5py 3.13.0, TensorStore 0.1.74 |
| Pinned OpenPI | `/mowenjie/fluffy-bassoon/third_party/openpi-215abfb` at `215abfb217dbac7d5f1273282331b9b1866c0479` |
| Stock JAX checkpoint | `/mowenjie/fluffy-bassoon/inputs/pi05-libero-openpi-20261027-r1`, verified against all 12,439,085,481 bytes in `READY.json` with SHA-256 `ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170` |
| Fixed action/state normalization | `assets/physical-intelligence/libero/norm_stats.json`, SHA-256 `b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84` |
| Tokenizer and import overlay | `/mowenjie/fluffy-bassoon/cache/openpi-215abfb` and `/mowenjie/fluffy-bassoon-2/setups/xemb-icml-campaign-20261121-r1/cv2-headless-overlay` |
| Exclusive training probe | Tracked `scripts/[train]gpu_probe.py`, `--expected-count 1 --purpose training --baige-exclusive-training`, using `/mowenjie/fluffy-bassoon-2/setups/robocasa365-native-20261101-r1/env/bin/python` |
| Data/converter | Pinned checkout's `data/source_data.py` and `outputs/libero-xemb-ur5e-sysid-20261120-r1/ur5e_motor.py`. Their exact file digests are written into `config.json`. |

Check the first native parity receipt, then the first cache block and epoch in `console.log`. Each split receipt records extraction wall time, execution batch and block hashes. Epoch logs report elapsed time and update count for all three heads. `launch.txt` and `exit.txt` record process wall time. `gpu.json` binds the card and scheduler job. `optimizer.msgpack` stores all factors and Adam moments at epoch boundaries. The epoch, deterministic seed and permutation rule restore the sampler. Fixed per-window PRNG keys reproduce the remaining cache tuples. To recover into a fresh attempt, append the absolute previous attempt directory as the fifth launcher argument. Completed cache blocks are shared by read-only hard links. `fit.json` is published after all three fits, reporting and a deployment reload check.

## Linux CPU qualification

Two Baige CPU cores (32–33), CUDA hidden, restored the stock checkpoint and executed the full native loss and ten-step stock/injected sampler on a seeded OpenPI fixture. The cached base loss matched the native per-token loss exactly. Each of the four enabled factors had nonzero native gradients at the perturbed point and zero measured difference from the cached gradients. The decoded stock and zero-update [10,7] actions matched bitwise. A nonzero kernel/branch changed the decoded chunk. This check took 177.3 seconds from checkpoint verification and peaked at 19,916,104 KiB RSS. The initial eager reference materialized an extra BF16 rounding boundary and disagreed with fused stock loss. Compiling the reference as it runs during fitting resolved it. [Full checkpoint receipt](evidence/native-cpu.json)

A separate test read only the real output projection from the checkpoint. With two synthetic [10,1024] feature sequences, its zero-initialized heads matched native projection output bitwise. An update changed the BF16 kernel and all four factors received gradients on the next pass. The zero-history joint path agreed bitwise with applying its merged kernel to the stock layer. [Restored layer receipt](evidence/projection-cpu.json)

To repeat the fixtures in the tabled environment, set `CUDA_VISIBLE_DEVICES=''`, `JAX_PLATFORMS=cpu`, `OPENPI_DATA_HOME` and the overlay as in the launcher, then run `taskset -c 32,33 "$python" qualify.py /fresh/path/projection.json` and `taskset -c 32,33 "$python" native_check.py /fresh/path/native.json`. The GPU job repeats cached/live and no-op sampler comparisons on an actual registered training row before producing a cache. GPU throughput, source scores and native reward are pending the parent's jobs.

The deployed training revision is `6785abc8323473b81973b9d69c93863bf523e4d0`. Its detached checkout is `/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-cotraining-20261124-r1/model/code-6785abc8323473b81973b9d69c93863bf523e4d0`. The integrated data reader is byte-identical to the source worker's `97bedc86` revision. The shell syntax, source manifest, converter and training probe were checked in that checkout. Job receipts will appear at the sibling `gpu01/` path.

A third CPU fixture exercised the production optimizer and scorer on 37 synthetic hidden sequences using the real stock projection. It covered a full batch and a five-example remainder, each intervention, per-trajectory scoring, serialization and recovery of the fixed final epoch. All recovered parameter hashes and epoch records agreed bitwise with the uninterrupted fixture. The experiment's held-out scores come from released demonstrations on the distributed job. [Optimizer fixture receipt](evidence/optimizer-cpu.json)
