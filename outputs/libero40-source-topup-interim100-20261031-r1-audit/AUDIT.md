# Source40 interim 25/100 fitting: terminal audit

Baige job `job-wbvs39zbt0pw`, run
`libero40-source-topup-interim100-20261031-r1`, exited0. All four AD/DPT ×
full/static controllers reached the fixed100-update milestone. Both25 and100
archives contain a policy and complete optimizer/sampler/Python/NumPy/Torch/CUDA
RNG recovery. This is an interim source-size point, not completion of the1M
source-action or500-update phase, and not a robot-performance result.

The actual allocation GPU probe passed. The student used qualified OpenPI
Python3.11 and Torch2.7.1+cu128/CUDA12.8; all six focused profile/clock tests
passed. Final read-only audit passed26 metadata/checksum checks, including
rehashing all16 policy/recovery files. No models or tests ran on the development
host, and this fit executed zero additional simulator actions.

| Item | Verified identity or cost |
| --- | --- |
| Run source | `2bc15da3028ab6a3cbaed4b813f3f761a680f1d9` |
| CPU2 input-manifest file SHA | `6563d1fc76ba506cf9f44cb648c30455313a33602ab1154e30b37cc54bc54883` |
| Assembly **file** SHA | `89d28ad329b612ba52a4bcf108150a335de378a0e0b209e73b7bd5f47a347160` |
| Assembly **content** identity | `6a2dc1af43b290f35cda98ef6dc8bb07115bcbaeacef4aed8d977f758f83dd0a` |
| Corpus exposure | All40 train learner histories / fixed8 independent source-validation histories |
| Stored training / validation steering rows | 33,729 /8,202 for each objective |
| Train / validation all-in source actions | 351,297 /85,301 |
| Historical source qualification/control actions | 21,253 charged separately; not duplicated training rows |
| Cumulative source RESOURCE actions | 457,851 |
| Unique controller optimizer updates | 400; four trajectories of100, with25 as an intermediate archive |
| Worker time | 571 charged seconds =ceil(510.907 actual)+60 allowance; no overrun |
| Reserved worker time | 1,800 seconds;1,229 unused, subject to terminal scheduler/campaign reclaim |

Same architecture and optimizer for all four: seed0, width128/layers2/heads4,
context300, window128, batch8, AdamW LR3e-4. Per-controller cap300 seconds,
cumulative fitting cap1500 seconds. There was no recapture, reassembly, change of
dataset, continuation to500, or target-dependent checkpoint selection.

## Validation interpretation

| Controller | NLL at25 | NLL at100 |
| --- | ---: | ---: |
| AD full | 54.042404 | 51.806677 |
| AD static | 54.004303 | 51.806689 |
| DPT full | 13.508103 | 16.068619 |
| DPT static | 13.458465 | 15.892066 |

At the pinned run source, `vla_history_train.validate` initializes
`np.random.default_rng(734001)` on **every** call. Four validation batches of
eight with window cap128 therefore reuse the same sampled held-out windows
across25/100 and full/static. Trial/window draws in `sample_batch` do not depend
on reset mode. These are matched sampled windows, not the full validation corpus.

AD full/static validation NLL is effectively equal at100; this provides no
memory-benefit evidence. DPT validation worsened on the fixed sampled windows
from25 to100 for both modes. Overfit or source-local distribution mismatch are
possible explanations, not established diagnoses. Logged training losses use
different minibatches across updates, so those two train values alone do not
establish a decreasing training-loss trend. Absolute AD and DPT NLLs use different
target distributions and are not a robot-performance ranking.

The requested25/100 archives remain unchanged. No additional scale, source
checkpoint selection or robot-performance claim follows from these losses.

`audit.json` contains the compact terminal evidence, exact archive hashes,
complete recorded metrics, runtime receipt, clock, source costs and training
contracts. `assembly-binding.json` distinguishes the assembly file checksum from
its canonical content identity.
