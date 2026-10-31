# Fixed optimizer-scale training audit

Baige `job-xla7geuytgau`, run `libero40-subset-scales-20261030-r2`, source
`1a415032f6b93bcc5b5e5d62c0496aad19d048bb`, exited0. Four AD/DPT full/static
controllers each completed500 unique optimizer updates. Fixed25/100/500
exports are12 checkpoints along four trajectories, not12 separate trainings.
Elapsed1863.26s;71 Linux checks passed. No new simulator actions.

The [independent artifact audit](independent-artifact-audit.json) verifies all
milestone policy/recovery hashes, Adam steps and complete sampler/Python/
NumPy/Torch/CUDA RNG. [Scale matrix](scales.json) binds742 train chunks and617
validation chunks from disjoint physical states of one Spatial00 goal.
[Data-amount assemblies](data-amount-scales.json) are318/603/742 chunks from
8/16/20 whole source cases. The8-case prefix has no teacher update and is
raw-only for AD; these assemblies are prepared, not additional trained models.
Full teacher interaction costs remain attached, so data-amount ablation is not
a source-interaction-efficiency claim.

AD's own validation NLL worsened while train loss fell; DPT's own validation
NLL improved. Their target entropies differ, so raw AD/DPT loss comparison
does not rank robot performance. Primary exports remain fixed requested final
updates. [Subsequent robot comparison](../libero40-subset-target-audit-20261030-r1/README.md).

First attempt `job-nhd0iek20qwx`/source `80c48bb` stopped naturally at321
updates after its300-second per-controller ceiling. Immutable25/100 exports
and full321-update recovery remain on Baige; it did not produce a valid500
export. R2 used a fresh identity and an explicitly longer measured ceiling.
