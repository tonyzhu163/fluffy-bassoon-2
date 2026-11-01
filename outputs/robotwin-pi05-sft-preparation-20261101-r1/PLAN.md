# Fresh π0.5 SFT on clean RoboTwin before the PPO comparison

**HELD before allocation by the subsequent human request to set up RoboCasa
with a released π0.5.** This recoverable source draft is not launch authority.
No tests, model/data loading, conversion, optimizer steps or resource charges
were incurred here. The checkpoint correction identified below remains
unimplemented; do not interpret this draft as a qualified runtime.

Source proposal, 2026-11-01. The human clarification relayed by the protocol
owner is: “i meant we need to sft a pi0.5 on the robotwin first check vlrla's
hyperparams listed” and “for the pi0.5”. This supersedes the earlier plan to
start from the cached, jointly clean/random-trained Motus checkpoint.
`finish_assembly` retains the direct authority record, actual allocation,
deployment, qualification, launch and monitoring. No resource, dataset or model
was loaded or allocated by this source work; only code and compact metadata
were read. The unlaunched LIBERO search and all old campaigns remain held.
The later question about RLinf's released models is still being answered by the
protocol owner. This source proposal does not choose the adjust_bottle release
instead, nor authorize a fresh SFT launch while that baseline choice is open.
Exact source identities and the static-check scope are in
[SFT-SOURCE-AUDIT.json](SFT-SOURCE-AUDIT.json).

## What the paper establishes

VLRLA §5.1 uses an SFT π0.5 as the common baseline. Its Appendix A.2.1 begins
with an existing ALOHA checkpoint and specifies the later PPO stage. The paper
does not specify the initial RoboTwin π0.5 SFT learning rate, optimizer, batch,
step count, trainable mask, demo count or clean/random mixture. Table S1's
AdamW1e-4/batch64 belongs to the bridge, and actor5e-6/global2048 belongs to PPO.
Neither is the missing SFT recipe. This proposal chooses an official-runtime
SFT recipe; it does not claim an exact reproduction of unpublished settings.
The adopted prospective settings are in [RECIPE-DECISION.json](RECIPE-DECISION.json).
[Paper](https://arxiv.org/html/2610.19517v1#S5),
[retained stage reconstruction](../../docs/literature/vlrla-protocol.md).

## Reuse the pinned SFT implementation

Use RLinf `c605b9d55ed93a180119b099e6f836abff293b08`, already recorded for the
RoboTwin runtime. Its dataconfig already registers **pi05_aloha_robotwin**:
π0.5, H50, padded32/physical14, head plus both wrist images, discrete state
input and token limit200. ALOHA realignment is enabled; twelve arm joints use
delta targets, while both grippers remain absolute. The current official
RoboTwin SFT YAML is a π0 example; changing only its filename would not make
it π0.5. Our [config](../../configs/native_sft/robotwin_six_clean_pi05.yaml)
selects `model/pi0_5`, `task:sft` and the existing π0.5 dataconfig, without the
π0-only `state_proj` wrapper. No model, learner or data-loader framework is
added. The shared `openpi_fsdp` retains native FP32 precision islands.
[Pinned dataconfig](https://github.com/RLinf/RLinf/blob/c605b9d55ed93a180119b099e6f836abff293b08/rlinf/models/embodiment/openpi/dataconfig/__init__.py),
[official SFT guide](https://github.com/RLinf/RLinf/blob/c605b9d55ed93a180119b099e6f836abff293b08/docs/source-en/rst_source/examples/embodied/sft_openpi.rst).

Initialize from the official **pi05_base**, not Motus, a LIBERO SFT model, or
the adjust_bottle SFT model. First reuse a cached base with recorded provenance
if one exists. The owner must bind its actual path, revision, file hashes and
layout; no local availability is inferred here. The official origin is
`gs://openpi-assets/checkpoints/pi05_base`. If a format conversion is actually
needed, use c605's existing `jax_to_openpi` or `openpi_pytorch_to_openpi`
converter once, with source/output hashes and H50/padded32/token200. A base
already in RLinf layout needs no conversion. Do not start a checkpoint search.
SFT full weights can later be loaded directly as `full_weights.pt`; packaging
a second safetensors copy is not a mandatory gate.

## Clean demonstrations and normalization

The six tasks are handover_mic, pick_diverse_bottles, place_a2b_left,
place_object_stand, place_phone_stand and scan_object. The existing
[archive inventory](../archive/vlrla-protocol-20261029-r1/robotwin-datasets.json)
pins `TianxingChen/RoboTwin2.0@3dc3b798668feb99ac61cc9086d84cbcc3d79186`.
The six `demo_clean.zip` entries total1,968,673,953 bytes. This is archive
metadata, not proof of local download, successful extraction, episode count,
RGB order or matching embodiment. [SFT-DATA-PROVENANCE.json](SFT-DATA-PROVENANCE.json) selects exactly
these six entries; the alternative clean50 files and randomized500 files are
not additional training data. Reuse already converted data only if its exact
six clean inputs and transforms are recoverable. Otherwise stage only these
six archives after actual storage/write checks; no new demonstration collector.

The official RoboTwin preprocessing and ALOHA-to-LeRobot converter are
recoverable at RoboTwin `13c3c47ff4312dd62484bcd51be034af55c062d1`, separate
from the immutable simulator pin `dca9ec6`. Use their existing data path with
fresh output identity and deterministic episode order/prompt RNG; never execute
their overwrite-existing-directory behavior against shared data. Preserve raw
files. The preprocessing aligns observation at index t with the following
absolute joint target; it must not be relabelled as same-index state copying.
Verify actual joint/gripper order, three camera/color conventions, instruction
mapping, episode boundaries and action-window padding in allocated Linux.
The converter's nominal `fps=50` is a dataset-index convention, not a measured
native simulator video clock or Flora timing qualification.
[Official converter](https://github.com/RoboTwin-Platform/RoboTwin/blob/13c3c47ff4312dd62484bcd51be034af55c062d1/policy/pi0/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py),
[preprocessor](https://github.com/RoboTwin-Platform/RoboTwin/blob/13c3c47ff4312dd62484bcd51be034af55c062d1/policy/pi0/scripts/process_data.py).

Use all successfully verified clean episodes from those six archives, with
their actual per-task episode/frame counts recorded before fitting. Do not
assume50 from a filename or silently replace broken episodes. The existing
shuffled map-style loader samples the combined frame-index dataset; this is
not a new task-balanced sampler. Compute state/action normalization from these
actual clean TRAIN records with the existing `calculate_norm_stats.py`, using
`--config-name pi05_aloha_robotwin`; preserve its transform order and bind that
same exact norm file at inference/PPO. No random evaluation data, held-out
tasks, target labels or Motus joint-domain statistics enter normalization.

## Our fixed recipe and honest validation scope

| Setting | Chosen value / origin |
| --- | --- |
| Objective and mask | Native π0.5 flow-matching SFT; full backbone and action expert; no LoRA/value head. Actual requires_grad/optimizer membership is the authority, not an RL-only flag. |
| Optimizer | AdamW, peak2.5e-5, β(.9,.95), epsilon1e-8, weight decay1e-10, gradient norm1; official OpenPI/RLinf defaults. |
| Global batch / seed | 32 /0; current reference default. Proposed4 ranks ×micro4 ×accumulation2; tune execution microbatch only while preserving global32. |
| Endpoint / schedule | Fixed20,000 updates from π0.5 RoboTwin dataconfig; warmup1,000, openpi_cosine horizon30,000/end2.5e-6. The fit ends before the decay horizon; do not silently retime it. |
| Recovery / primary | Save at5k/10k/15k/20k, fixed20k primary. No target-outcome checkpoint selection. |
| Precision / execution | FP32 master parameters/optimizer; BF16 backbone computation; FP32 reduction/buffers and native islands; full_shard/use_orig_params, nonreentrant checkpointing. These are current-runtime execution choices, not paper SFT claims. |

The source draft has only static syntax/source checks. The owner must run the
real composed-config validator and actual loader/model/loss/gradient checks on
allocated Linux. Count those first real updates inside the fixed20k fit rather
than launching a separate optimizer pilot. Record strict base loading,
trainable tensors/counts, representative nonzero finite gradients and parameter
changes in vision/VLM/action expert, actual loss/LR/updates and resource cost.
Preserve full optimizer/scheduler/data-iterator/RNG recovery and explicitly
validate any resumed continuation; weights alone are not a resumable fit.

One concrete recovery gap remains in the pinned source: the official SFT loader
returns OpenPI's loader wrapper, but `FSDPVlaSftWorker.save_checkpoint` saves its
extra `data.pt`/`rng.pt` only for `StatefulDataLoader`. The generic FSDP save path
does cover weights/optimizer/scheduler; it does not establish the official
loader's exact position and RNG continuation. Do not advertise a complete
resume from this config. Inspect the actual loader on allocated Linux and make
only the needed existing-worker recovery correction, verified by matching the
next batch/loss/update after save/restore. This is core checkpoint behavior;
it is not a reason to add run-bookkeeping tests or a disposable fit job.

The c605 embodied SFT worker's `get_eval_model_output` raises
`NotImplementedError`. This proposal therefore leaves periodic validation
disabled and does not pretend a held-out flow-loss curve exists. Training loss
alone is not success or generalization evidence. The fixed final checkpoint
must be freshly evaluated on clean and randomized native cases before PPO.
Do not replace that endpoint with an earlier checkpoint chosen on target score.

## Proposed owner dispatch, then the shared comparison

`SFT-PROFILE.proposed.json` is a finite proposal, not a reservation: CPU data
preparation at most3,600 worker-seconds on8 cores/64GiB; then one4-GPU/64CPU/
384GiB/32GiB-shm SFT allocation with a7-hour inclusive ceiling. This is at most
**28 aggregate GPU-hours**, explicitly not a claim to fit under the older10h
rule. The owner's recorded newer per-job≤8GPU/<8h authority and actual SFT-first
scope must be bound to its component ticket before dispatch. Reserve at most
512GiB new data/checkpoint/output storage, without new purchase/quota; actual
free capacity, write probe and retention plan must be checked. No claim that
20k fits in this ceiling is made before timing actual advances. At a finite
boundary preserve complete recovery; no automatic extra job, clock reset or
scientific endpoint change. Parent may bind a smaller adequate profile.

Use the official `examples/sft/train_vla_sft.py` with this repository config;
set `RLINF_PATH`, `EMBODIED_PATH=$RLINF_PATH/examples/sft`, and the required
`ROBOTWIN_SFT_{BASE_MODEL,TRAIN_DATA,NORM_STATS,OUTPUT}` paths to newly bound
inputs/outputs. The illustrative command is:

```sh
"$SFT_PYTHON" "$RLINF_PATH/examples/sft/train_vla_sft.py" \
  --config-path "$PROJECT_SOURCE/configs/native_sft" \
  --config-name robotwin_six_clean_pi05
```

After completed SFT, use that **same checkpoint, normalization, H50 and sampler**
for a fresh clean/random native baseline; Motus H32 measurements cannot carry
as its baseline. Then copy the identical SFT initializer into S/F PPO on the
chosen randomized task, with identical fresh value-head initialization and
matched sampling. These later baseline/PPO phases need their own concrete
finite owner tickets. Clean-only local SFT followed by random-task PPO is online
adaptation, not zero-shot clean-to-random PPO transfer. Broader pretraining
exposure of pi05_base is not proven absent. No S/F PPO is launched or admitted
by this source proposal, and no old run or sealed balance resumes.
