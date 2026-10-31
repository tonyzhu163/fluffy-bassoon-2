# Native PPO collection contract: action histories, not frozen-motor noise labels

This amendment is necessary capture work before GPU dispatch. The algorithm,
policy, sampler, reward and optimizer are unchanged. The new CPU fixture is
pending actual allocated-Linux validation; this document is not a runtime PASS.

| Requirement | Actual retained field |
| --- | --- |
| Three raw current RGB views, proprioception, instruction | `traces/rank-*/NNNNN.pt.gz:current_observation`: native `main_images`, two `wrist_images`, `states`, `task_descriptions` |
| Next and final observations | `outputs[0][-1]`; `outputs[4][-1].final_observation` preserves pre-auto-reset final frames when present; source auto_reset=False retains true final directly |
| Source initialization | Every reset under `initial-states/source-*.pt`: public arrays/hash, scene reset IDs, actual affected env indices, episode serials, global RNG, sampler generator and cursor |
| Task/case/episode/time identity | `task_key`, rank/stage, global/local env IDs, reset IDs, episode serials, round0–3, chunk index, primitive time before, collection iteration and producer iteration |
| Executed native action | Submitted decoded14D chunk plus actual training `valid_action_mask`; per-env durations distinguish executed prefixes and padded slots. No invisible padding is called executed. |
| Feedback and boundaries | Raw per-action rewards, terminations and truncations separately in `outputs[1:4]`, full raw infos; pre-SGD dones, loss masks and final bootstrap rows remain untrimmed |
| Policy statistics/processed input | `training-rollouts/*.pt:rollout_batch`: old log probabilities, values, advantages/returns, masks, versions, raw replay input tensors and tokenized instruction |
| Native flow state | Full `[time,batch,6,50,32]` chains, five denoise indices, normalized/padded32D model action, decoded14D action; shapes/dtypes are written in each JSON sidecar |
| Actual Gaussian parameters | `sampling/rank-*/*.pt.gz:actual_gaussian_parameters`: each original sampler call's actual returned mean/std, integration index, method, coefficient and step count |
| Sampler randomness | Per-prediction Python/NumPy/TorchCPU/CUDA RNG before/after, actual initial Gaussian `chains[:,0]`, and complete realized flow chain. No extra sampling or model pass is used by the logger. |
| Terminal value-only calls | Sampling logs explicitly mark `terminal_value_only`; these unused action predictions are not AD behavior samples. |
| Provenance/units | `resolved-native-config.json`, preflight source/initializer/norm/seed hashes, versioned `collection-contract.json` plus per-batch shapes |
| Complete recovery | Official actor/Adam/scheduler DCP, scaler/counter, worker/global/motor RNG and source/evaluation seed cursors at completed optimizer boundaries; interrupted prefixes remain raw/charged |

The primitive observation stream is at policy decision boundaries and final
states. Intermediate RGB frames for all50 motor substeps are not asserted.
All50 submitted14D actions and the actual leading execution mask remain.
The logger preserves raw native coordinates and the author's normalization/
delta/decode configuration rather than inventing a single physical unit for
all14 coordinates. Normalized32D model actions and decoded14D commands are
different named fields.

`EmbodiedRunner.run` calls each worker's setter with its completed-iteration
clock before collection. In c605 `EmbodiedFSDPActor.set_global_step:840–846`
sets **version**, not `global_step`. The wrapper now stores that authoritative
clock explicitly. Collection1 uses policy0; the checkpoint named step1 is
after its PPO update. Archives have collision guards and cannot overwrite an
earlier policy version. This also fixes per-iteration mask/invariant filenames.

The actual pre-SGD batch layout is `[chunk_index, round*local_env_count+env,
...]`, with final bootstrap rows retained. Flattening/shuffling for PPO occurs
only after this archive. Every episode outcome remains, including failures and
post-termination padded rows with explicit zero execution masks. Training and
evaluation are separately marked; the fixed128 evaluations never enter the
training-rollout manifest. Parallel environment slots are identifiable; a
reset seed alone is never claimed to prove a distinct physical configuration.

Direct-action AD/BC and an action/flow-matching bridge can consume these
behavior histories. **Noise-steering AD on the unchanged SFT motor is not
label-ready**: native PPO updates the action expert, so its Gaussian inputs
do not define equivalent steering actions for another frozen motor. That route
needs an explicit inverse or differentiable compatibility mapping. Keeping the
chains/RNG is useful evidence, not a shortcut around this mismatch.

DPT final-teacher actions are distinct from behavior actions and are not
fabricated during collection. A later immutable label generation can query a
preselected final source teacher on these recorded raw query observations,
with exact checkpoint/config/norm hashes and prescribed sampling seeds, without
new environment interactions. Its competence is a measured robot result; a
saved endpoint is only a candidate, not automatically an expert. Preserve the
behavior actions/history and label-teacher identity separately.

One allocated CPU fixture covers two ranks, four rounds, success/failure,
episode boundaries, true durations/post-terminal padding, all three current/
next/final views, instructions, AD prefixes and DPT query/teacher references.
It uses the actual author unflattening and GAE before capture, then demonstrates
that later training flattening cannot remove the archived bootstrap/masks.
The same fixture checks the Gaussian observer returns the exact original
mean/std objects with one original call and no additional RNG draw. The
separate version regression invokes the actual setters0/1 without a fabricated
global_step and checks two archives and unchanged first-file SHA.

Effective loss behavior remains the pinned author's implementation, including
its masked_mean_ratio reduction and rank-local advantage normalization. The
4/8 execution minibatch arithmetic alone is not a claim of identical effective
normalization or identical source samples. No fidelity discrepancy is silently
fixed in a capture patch.
