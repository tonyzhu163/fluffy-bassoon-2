# Optional spatial action head for fresh source starts

The experiment asks whether two-view localization and a chunk-level choice among action modes improve fresh Goal 5 success on Panda and Sawyer at the same demonstration and update budget. R1 selected update 150 at Panda validation MAE **0.03492**. The three translation axes had MAEs of **0.05265, 0.07690 and 0.05203**. Both independent Panda starts reached 300 controls with zero reward. This motivates checking signed movement as well as aggregate imitation error. These results alone cannot identify action averaging as the cause. [R1 validation curve](../libero-paligemma-crossarm-20261120-r1/evidence/panda01/motor/curve.json) · [Fresh Panda episodes](../libero-paligemma-crossarm-20261120-r1/evidence/panda01/panda_native/report.json)

## Architecture

The R1 `ChunkHead` here projects the two contextualized 16×16 image-token grids, applies a strided convolution, then flattens and concatenates the results with the final prompt token and proprioception. Its readout regresses 35 scalars. The optional `spatial_bins_v1` head has **417,759** trainable head parameters versus **1,151,435** in that baseline, calculated from layer shapes at width 2,048. Both use the verified MIX-224 model, its real external and wrist camera encodings, identical state normalization and rank-eight LoRA on the query/value projections of Gemma layers 16 and 17. [R1 motor](../libero-paligemma-crossarm-20261120-r1/motor.py) · [Optional head and factory](servo_motor.py)

Four learned queries per view use the final instruction token and the eight public proprioceptive values. Each query pools a visual descriptor and a soft 2D patch location from its own camera. A fusion layer predicts a prior over three modes for the whole chunk. A shared decoder with five learned time-offset embeddings predicts action bins and fine corrections for each mode, control and axis. The time offset is internal to the proposal. Each replan starts from the current two images, prompt and public proprioception. The mode can represent a phase or maneuver visible in those observations. Its semantics emerge from the issued action labels.

For each axis, two discrete outcomes reproduce exactly −1 and +1. Nine equal interior intervals cover the remaining range, with a learned continuous offset in each. Training scores each mode against the **same entire expert chunk**, summing log bin probabilities and subtracting half the within-bin Huber error in half-interval units. Endpoints have zero offset loss. The objective marginalizes the three scores with their log priors and divides by the number of valid scalar labels in the anchor. Padding contributes zero. Gradients flow through the head into the actual Gemma LoRA. Inference chooses the highest-prior mode once for the chunk, then the highest-probability bin and bounded offset for each axis. It returns a deterministic float32 proposal of shape `(5, 7)` through the inherited `act(obs)`. Ambiguous public observations can still cause the gate to choose the wrong maneuver.

## Wiring contract

The implementation lives in `servo_motor.py` and uses the existing R2-to-R1 Python import path. The following edits belong to the entry-point owners. The default architecture is `chunk_huber_v1`.

1. **Training.** In `motor_train.py`, import `servo_motor as servo` after setting up the R1 path. Add the train-only flag, substitute the constructor and recipe, and store the architecture in the top-level fit contract. Keep the existing recipe updates and the `validate_every` removal.

   ```python
   fit.add_argument("--architecture", choices=servo.ARCHITECTURES, default=servo.BASELINE)
   # In train(args)
   motor = servo.build_motor(args.weights, args.architecture)
   fit_recipe = servo.training_recipe(args.architecture)
   # After constructing contract, before manifest write or resume comparison
   contract["architecture"] = args.architecture
   ```

   In `update`, replace the prediction, Huber computation and loss reduction with the following line. Keep `loss.backward()`, global-batch accumulation, both optimizer groups, first-update LoRA gradient check, clipping and restart handling. Record the accumulated value as `train_objective` and keep the historical `train_huber` field for the baseline only. Add `HERE / "servo_motor.py"` to `source_ids()` so new fits, stages and transfer bind the same implementation. Offline `base.validate` already calls the decoded forward path.

   ```python
   loss = servo.loss_per_anchor(motor, obs, target, mask).sum() / len(batch)
   ```

2. **Source development and confirmation.** In `motor_eval.FrozenMotor.__init__`, import the same module and replace `base.Motor(weights)` with `servo.build_motor(weights, servo.architecture_name(contract))`. `select` can then restore each candidate with its existing strict parameter-key, shape and dtype checks. The selected file is still the byte-for-byte promoted candidate at `FIT/selected/checkpoint.pt`.

3. **Transfer.** In `transfer_gate.confirmed_motor`, import the same module and replace `motor = ev.load_motor(args)` with `motor = servo.load_selected(saved, args.weights)` inside the existing shared lock, after `confirm_receipt`. The factory uses the already-opened checkpoint, checks the full base identity, restores `best["model"]` and freezes the motor. The existing pre/post tensor inspection and checksum checks still apply. `transfer.py` already includes the producer's `source_ids()` and passes this motor into both fit and evaluation. Historical checkpoints without an architecture field dispatch to the original R1 head. Unknown names raise an error. Existing campaign seals and code pinning continue to govern eligibility.

## Source-only decision

Use the existing R2 recipe in two fresh fit directories, one per architecture. Match bank hashes, train/validation partitions, seed, sampled anchors, global batch, updates, learning rates, offline shortlist size and native rollout budget. On validation layouts 28–31, rank each fit's checkpoints by the existing equal-arm MAE. As a mechanism check on those same labels, report per-arm, per-translation-axis counts of expert actions at exactly +1 and −1 and the fraction of each whose decoded action has the same sign and magnitude at least 0.9. Report the denominator for each sign. This diagnostic does not select the checkpoint. [Bank and sampler](motor_bank.py) · [Motor recipe](MOTOR.md)

Compare both fits on the same native development resets, layouts 32–35 per arm, executing all five actions before replanning. Report paired wins and losses by arm against the matched Huber motor. Apply the existing development ordering across the two fit winners (minimum arm successes, total successes, offline MAE, then earlier update). Break an exact tie in favor of the baseline. Only a winner with **at least 3/4 on each arm** enters the single campaign seal. Freeze it before opening layouts 40–47. Require **at least 6/8 separately on both arms** on the same frozen checkpoint. If both fits miss development, stop there. Only a passing confirmation can release transfer. [Source plan](PLAN.md) · [Panel and ranking](motor_eval.py) · [Transfer gate](transfer_gate.py)

A gain in endpoint recovery without a gain in paired native development success counts against action attenuation as a sufficient explanation. Tied native results leave the fresh-start benefit unestablished. This comparison tests spatial pooling and the modal decoder together. It cannot attribute a gain to either component alone, and confirming one selected motor qualifies only that motor on the sealed panel.

## Implementation evidence

Static inspection traced the four public inputs from `Episode.observation` and the rollout's `public_obs` into the inherited motor. The bank supplies pre-action camera pairs and masks the final partial chunk. The trainable delta contains the new head, the existing LoRA and both state normalization tensors. The camera coordinate grid is constructed from the versioned raster convention. Python syntax and entry-point signatures were checked locally. CUDA execution, gradient values, checkpoint round-trip and native success are unmeasured for this variant. [Observation and targets](../libero-paligemma-crossarm-20261120-r1/motor_data.py) · [Rollout inputs](../libero-paligemma-crossarm-20261120-r1/adapt_eval.py) · [Source evaluation](motor_eval.py)

The R2 trainer, source evaluator and transfer loader now dispatch on the saved architecture field. Full comparisons use the same immutable expanded source bank.
