# Public original π0.5: native loading and minimal preprocessing

Static source preparation,2026-11-01; no tests, tensors, conversion, native
episode or GPU job was run here. The existing owner selected the public original
checkpoint after verifying that Aaron's100k endpoint is gated and its custom
StructRL source returns404. This is the original Human300 step75000 model,
**not** the Aaron Composite-Seen100k continuation or generic LeRobot release.
The latter's different state/action order and normalization epsilon do not carry.

## Exact source and small delta

Initializer: `robocasa/robocasa365_checkpoints@c484448aba1a9b60a04c9b0ca117241518ea69f3`,
prefix `pi05_pretrain_human300/multitask_learning/75000`. Stage only its
`params/`, `assets/` and checkpoint metadata needed by the existing converter;
exclude `train_state/` and optimizer archives. Actual file inventory, sizes,
hashes and accessible local paths must be bound by the owner before download.
Public native source is
`robocasa-benchmark/openpi@5a6beda9ff99da30b4e1b59320f6a32971d7c397`.
No new SFT, model family or alternate checkpoint comparison is needed.

That source's Groot converter fixes16-D state order to relative-eef position3,
relative-eef quaternion4,base position3,base quaternion4,gripper qpos2. Its
12-D action order is eef position3,eef rotation3,gripper1,base motion4,mode1.
These match RLinf's existing RoboCasa365 policy boundary. The public policy
requires all3 cameras in left-agentview,wrist,right-agentview slots, pads state
to32 **before** mean/std normalization and discrete-state tokenization, then
returns the first12 native actions. Do not insert the generic LeRobot
permutations or quantile statistics.

[RLINF-PUBLIC-PI05-PREPROCESSING.patch](RLINF-PUBLIC-PI05-PREPROCESSING.patch)
adds two opt-in fields to the existing d80 RoboCasa dataconfig:
`use_quantile_norm:false` and `pad_before_normalization:true`. Existing defaults
remain None/False. No new policy, environment, loader or training framework.
[PREPROCESSING-SOURCE.json](PREPROCESSING-SOURCE.json) binds original/patched
file hashes. Apply/commit exactly in a fresh RLinf checkout and record its new
Git pin; do not change an already admitted setup or ended run in place.

The same patch also makes the existing environment honor explicit gripper and
base-mode score mapping: the native PandaOmronKeyConverter thresholds both
at0.5 into−1/+1. Continuous controls and the caller's stored PPO action array
remain unchanged. This existing action boundary had an unused config field;
the new recipe explicitly enables both mappings. Do not substitute raw0/1
scores for the native controller commands or mutate actions used for logp.

Five numerical CPU cases are prepared in
`tests/test_robocasa_pi05_preprocessing.py`. They check source-ordered state
values/padding, mean/std before tokenization, all3 camera slots, and native
action unnormalization/order, and exact score-control agreement with the actual
native PandaOmron converter. They construct only configuration/statistics and
synthetic arrays, with no simulator, checkpoint, tokenizer download, policy or
GPU. On allocated Linux with that patched source:

```sh
"$NATIVE_PYTHON" -B -m pytest -q \
  "$PROJECT_SOURCE/tests/test_robocasa_pi05_preprocessing.py" \
  --junitxml="$FRESH_OUTPUT/preprocessing-junit.xml"
```

This is a new affected numerical selector, not a repeat of the old Flora67/5
or SFT tests. It may fit the already admitted CPU setup clock only if its exact
source is adopted before that job starts; otherwise use the later separately
bound policy-loading/qualification profile. No test is claimed passed here.

## Existing conversion and first native case

Use RLinf's existing `jax_to_openpi` converter on allocated Linux if the public
checkpoint has no already verified compatible Torch copy. Bind H50,padded32,
token200 and the exact bundled native mean/std file. Preserve both original
checkpoint and converted tensors/hashes; do not convert through a task-SFT or
LIBERO checkpoint, train-state archive or arbitrary weight rename.

```sh
"$NATIVE_PYTHON" -m rlinf.utils.ckpt_convertor.openpi.convert \
  --mode jax_to_openpi --input-model "$PUBLIC_STEP75000" \
  --input-norm-stats "$BOUND_NATIVE_NORM" \
  --output-model "$FRESH_TORCH_MODEL" \
  --output-norm-stats "$FRESH_TORCH_MODEL/robocasa365/norm_stats.json" \
  --action-dim 32 --action-horizon 50 --max-token-len 200
```

This is an illustrative command, not an admitted conversion allocation or a
claim it has succeeded. Actual strict native loading must match every backbone
key without missing/unexpected tensors; a later PPO model may add only the
same identically seeded value head to both arms. Native model sampling uses
the public H50/10-flow-step convention and3 views, with actual sampler/normalizer
and transformation outputs qualified from the bound source. Exact JAX task
score reproduction is not an extra gate.

The source-prepared first-case config is
`configs/native_ppo/robocasa365_pi05_eval.yaml`. It genuinely selects pi0_5,
the opt-in mean/std/padding fields and3RGB, and contains no policy-path fallback.
Use existing `evaluations/eval_embodied_agent.py` with that config on an actual
GPU allocation. Its first case is native OpenDrawer,target/atomic_seen,
registryH750 at the proposed RoboCasa1.0.1 source, with raw video annotations
disabled and zero optimizer steps. Record actual GPU/CUDA/rendering/import
proof, accepted case, control clock, shapes/actions, termination/masks, output
and before/after motor hashes. This is a native setup/competence case, not a
leaderboard score or PPO qualification. Actual source/normalizer/GPU/cost/clock
and full case budget must be admitted by the sole owner before execution.

## Matched PPO boundary

Keep RLinf's PPO optimizer/loss and two matched arms. Upstream365 uses
delta-success reward and success termination; our terminal comparison must
prospectively change **both** arms to full native horizon, no success-driven
termination and zero intermediate labels. At the same complete-batch pre-GAE
barrier, S receives terminal ground truth0/1; F receives only
`(expected_score-1)/4`, with truth kept in the monitoring side channel. Reuse the
existing isolated Flora scorer/pre-GAE/recovery path after its real365 port;
the LIBERO44 hardcoded schedules/H400/20Hz are not already qualified for365.

The proposed fallback task ScrubCuttingBoard has nativeH1200. At16 environments
and4 complete rollout rounds it yields1,536H50 chunk samples; the protocol owner
accepted global512, giving3 minibatches per PPO epoch. Keep4 PPO epochs and
the default LRs/Adam/loss. Max20 is only an initial iteration upper bound:
native baseline, qualification, failures and all complete evaluation cases must
fit the1M all-in cap per arm. A fixed smaller schedule must be bound before SGD;
do not advertise12 rounds plus a baseline64 as fitting that cap. No old task44
baseline64 gate, automatic new SFT, target-checkpoint selection, partial source
batch or wider task search carries into this pilot.
