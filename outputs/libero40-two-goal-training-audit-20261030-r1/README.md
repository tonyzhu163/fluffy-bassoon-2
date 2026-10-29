# Two-source-goal controller training

Baige `job-ctqxbzx4bhhv`, source
`0c897640033373b1f3efbac63830ff6a626b0292`, completed exit0.
79 allocated Linux checks passed. Four AD/DPT full/static controllers each
completed500 optimizer steps (2,000 unique steps total), with immutable
25/100/500 policies and complete recoveries. Elapsed1858.58s; zero new
simulator actions.

Source tasks: Spatial00 and LIBERO10:00. Training has1,551 chunks
(742+809), independent validation1,664 (617+1,047). Shared source costs,
counted once across controllers:16,167 train and17,252 validation actions,
including settling. These are two of the intended40 source goals.

The independent saved-artifact audit checked all12 policy/recovery hashes,
policy equality with recovery tensors, exact Adam step counts and complete
sampler/Python/NumPy/Torch/CUDA RNG. The source actors remain candidates,
not demonstrated experts. Validation banks differ from the one-goal scale;
raw cross-run validation losses cannot measure a source-diversity gain.
Robot outcomes on common target cases are required.

Evidence: [independent audit](independent-artifact-audit.json),
[immutable scales](scales.json), [assembly qualification](two-goal-qualification.json).
Bulk policies/recoveries remain under
`/mowenjie/fluffy-bassoon-2/outputs/libero40-two-goal-scales-20261030-r1/`.
