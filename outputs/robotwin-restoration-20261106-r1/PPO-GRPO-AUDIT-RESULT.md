# RoboTwin PPO/GRPO audit: valid execution, effectiveness unestablished

Updated November8,2026. Student/ICRL accelerator work remains paused under the
user's requirement for an effective source learner. The100 GPU-hour allowance
is cumulative and includes failed jobs and engineering diagnostics.

## What the experiments establish

Both corrected methods complete two fresh1024-episode collections and30 native
Adam updates per actor, with complete checkpoints0/1/2. Full process restores,
owned optimizer/scheduler/scaler/RNG checks, frozen-VLM checks, original behavior
likelihood checks and independent whole-buffer advantage references pass. The
training/evaluation implementation can run correctly. This is distinct from
showing a useful learner.

| Fixed checkpoint2 readout | SFT successes | Trained successes | Gained/lost cases | Exact paired p | Interpretation |
| --- | --- | --- | --- | --- | --- |
| PPO DEV |6/64|10/64|6/2|.2890625|+6.25points; insufficient evidence|
| GRPO DEV |6/64|11/64|5/0|.0625|+7.8125points; selected before confirmation|
| Selected GRPO disjoint confirmation |7/64|10/64|8/5|.5810546875|+4.6875points; predeclared effectiveness gate fails|

DEV selected the method, not the training checkpoint: checkpoint2 was fixed
before training. The rule, selection commit and negative confirmation are
retained separately. Do not pool DEV with confirmation to override the result.
The gate was≥10 percentage points and exact two-sided paired p<.05 on
confirmation. A failed gate does not prove that the true effect is zero; it
means a useful improvement has not been established by this experiment.
There is only one training seed per method and no transfer result.

The exploratory checkpoint1 GRPO readout also passed full restoration and
independent cold/physical/noise verification: SFT6/64→10/64, four gains/no losses,
paired p=.125. Cross-run CPU verification matches all384 actual Gaussian draw
hashes,64 initial B2 observation hashes and all64 SFT outcomes. The larger DEV
curve is6→10→11 successes across checkpoints0/1/2. Checkpoint2 gains three
scenes and loses two relative to checkpoint1 (p=1.0); there is no evidence for
the suspected second-cycle regression, or for a reliable second-cycle gain.
[Checkpoint curve evidence](MOTUS-SOURCE-GRPO-CURVE01-READBACK.json). This
exploratory comparison cannot replace checkpoint2's negative confirmation.

## Why the earlier attempts were misleading

1. **Execution batch changed the likelihood reference.** On the actual unchanged
   policy, B32 reproduces original behavior likelihoods exactly on5958 valid
   decisions. B256 gives joint ratios0.520–1.883, with11.279% outside[.8,1.2].
   Replacing saved behavior probabilities with an actor recomputation hid the
   discrepancy in the initial replay check. The corrected controls use
   micro32/accum8/world8/global2048 and require original behavior agreement for
   every collection. The diagnosis does not isolate BF16 or prove this alone
   caused every weak endpoint.
2. **Timeouts and an optimizer guard interrupted usable endpoints.** Earlier
   jobs often stopped before a full collect/update/evaluation chain. The later
   readout failure was specifically a guard rejecting `{}` materialized by
   native offload/onload for registered zero-length unowned FSDP shards.
   Captured owned Adam moments and groups were unchanged. A narrow equivalence
   fix passed12 Linux core checkpoint tests, native offload/onload reference,
   and both methods' actual GPU restores. These are engineering failures,
   not evidence against PPO/GRPO learning.
3. **The port is a different scientific experiment.** Released RLinf pi0.5
   results use task-specific `adjust_bottle` SFT, H50 and clean scenes. This
   port uses joint Motus, the phone source task, H32 and H192. It preserves
   Motus transforms and the held-out task split. Calling it an exact release
   reproduction would be incorrect. Changing the native H32 transforms to
   H50 would not improve fidelity to the Motus motor.
4. **Two collections are a short learning test.** Thirty Adam steps mean
   `6144/2048*5*2`, not30 fresh-data iterations. VLRLA records25 fresh PPO
   iterations and CFG-Policy50 rounds under other configurations. More
   fresh-data training is motivated; improvement is not guaranteed.

The actual reward audit rules out an all-failure explanation: a corrected PPO
collection contains136 positive episodes among1024; the full GAE reference and
positive-episode advantage signs pass. GRPO has73/128 then67/128 tied groups,
so binary feedback wastes substantial group capacity but leaves informative
mixed groups. Neither reward shaping nor filtering was silently introduced.

Native printed training scalars are diagnostic only: PPO approximate KL
.040/.028, clip fraction.173/.121, critic explained variance−.049/.022;
GRPO KL.044/.026 and clip fraction.083/.060. These rounded logs are not
independent recomputation. In pinned RLinf `algorithms/losses.py:261,315`,
`actor/clipped_ratio≈.99` is the masked mean of the clamped probability ratio;
it is not99% clipping. `actor/clip_fraction` measures surrogate clipping. Poor critic fit, exploration, horizon,
scene diversity and over-updating remain plausible, unisolated explanations.

## Shared RoboCasa engineering is in the executed path

Training and evaluation use allocated GPU/renderer attestation, deferred scene
setup, environment offload and executor recycling, batched controllers,
skipped intermediate renders, frozen encoder eval mode, bounded asynchronous
I/O, NVMe staging and project BOS cold outputs. Actual combined simulator/model
reserve is monitored. Global batch, weighting, optimizer schedule and native
sampling are preserved. Source readouts retain all24 worker states per policy
and compare192 actual paired Gaussian tensors and32 initial B2 observations;
all48 original SFT predictions match on the first DEV panel. See
[engineering parity](ENGINEERING-PARITY.md) for evidence and limits.

Maximizing occupied VRAM is not a scientific objective. The measured B256
likelihood drift disqualifies that otherwise faster actor profile. Future
batch/kernel/compile changes require numerical qualification before use.

## Next decision and compute boundary

All current jobs and independent CPU audits are complete. The final diagnostic
released its eight GPUs at08:32:21Shanghai, using9184 conservative GPU-seconds
including startup/cleanup (worker8912). The cumulative ledger closes at356198
GPU-seconds (98.943889h), leaving3802GPU-seconds (1.056111h) inside the100h cap.
No GPU jobs are active and no additional complete training cycle fits safely.
Historical costs are not reset by changing run identity.

The next useful training experiment is a predeclared fresh-data learning curve,
with horizon/exploration/critic uncertainty kept explicit and one factor changed
at a time. Extend only a validated resumable boundary; retain original binary
reward, native B32 likelihood, global2048/fiveepochs and the held-out split.
Use existing DEV for model development; the consumed confirmation bank is
retired from fresh claims. A new effectiveness claim needs a newly predeclared
independent confirmation protocol, and stability needs multiple training seeds.
A [concrete longer-curve proposal](MOTUS-LONGER-CURVE-PROPOSAL.md) requests
a155h cumulative envelope; approval is pending. It adds eight fresh GRPO
collections and fixed DEV/new confirmation, after Linux qualification and
per-job admission. Do not launch above the explicit100h cap without approval.

The first2→6 continuation is now implemented and Linux CPU-qualified. Checks
cover checkpoint2 cold files/24 worker sidecars/eight owned-Adam descriptors,
all48 original checkpoint2 DEV sampling files, exact schedule prefix/rotation,
unchanged numerical sections and the pinned fresh-collection loop with RPC
stubs. The first qualification rejected an incorrect common sampling offset;
measured per-worker final DEV offsets now pass. Full-model resumed predictions
and new SGD remain GPU checks, and the6→10 input needs its actual checkpoint6
qualification. [Continuation CPU evidence](MOTUS-CURVE-GRPO-CPU02-READBACK.json).
No additional GPU allocation or new learning result follows from CPU qualification.

## Evidence

- [PPO training closeout](MOTUSBEHAVIORPPO01-CLOSEOUT.json) and
  [GRPO training closeout](MOTUSBEHAVIORGRPO01-CLOSEOUT.json).
- [PPO paired DEV](MOTUS-SOURCE-PPO-DEV03-READBACK.json),
  [GRPO paired DEV](MOTUS-SOURCE-GRPO-DEV02-READBACK.json),
  [selection](MOTUS-SOURCE-SELECTION.json),
  [confirmation](MOTUS-SOURCE-GRPO-CONFIRMATION01-READBACK.json).
- [Joint likelihood audit](MOTUSJOINTAUDIT01-COMPLETE.json),
  [empty-shard CPU reference](MOTUS-EMPTY-STATE-AUDIT-CPU01.json),
  [native logged scalars](MOTUS-NATIVE-LOGGED-SCALARS01.json).
- [Primary-source literature review](../../docs/literature/robotwin-ppo-review.md)
  and [ordered audit/experiment plan](PPO-GRPO-AUDIT-PLAN-20261108.md).
- [Live cumulative ledger](LATEST-BUDGET.json); raw full artifacts remain on
  project BOS at the identities named in the receipts.
