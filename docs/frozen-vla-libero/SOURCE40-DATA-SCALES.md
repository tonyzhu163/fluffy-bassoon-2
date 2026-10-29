# Whole-case source data amount diagnostic

`vla_source40_data_scales` takes one qualified source goal's complete train
history and its independent validation history. Its explicit
`--qualification-subset` flag is required. It selects chronological whole
cases, never individual successful transitions or outcome-chosen scenes.
The normal all40 source assembly gate is unchanged.

```sh
python -m fluffy_bassoon.meta_rl.vla_source40_data_scales \
  --train-index <qualified-source-train>/index.json \
  --validation-index <independent-source-validation>/index.json \
  --output-dir <fresh-data-scale-directory> \
  --case-counts 8 16 20 --qualification-subset
```

The hash-bound `index.json` records actual cases/episodes/chunks and original
source/teacher costs at every scale. Each learning-history scale writes a
standard small-pilot `cases-NNN/assembly.json`, usable by existing training
code only with its explicit small-pilot admission. Validation arrays are
unchanged for every scale. Teacher-label provenance continues to point to
the original final source actor; output datasets do not modify raw records.

For the first capped Spatial00 corpus, episode receipts show:

| Train cases | Episodes | Query chunks | Source update range in episode receipts |
| --- | ---: | ---: | ---: |
|8 |32 |318 |0 only |
|16 |64 |603 |0–2,140 |
|20 |80 |742 |0–4,920 |

The8-case prefix is warmup-only and **cannot be AD**. The existing fixed
teacher-clock guard remains in force. The helper retains its executed-noise
targets as explicitly named behavior cloning, and its final-source-actor
DPT labels; it writes no matched AD/DPT assembly for that scale. An AD
dataset exists only where actual row teacher clocks change. The numeric
clock values in the final manifests come from dataset rows, which precede
the postepisode update counters shown in the table.

All scales retain the full original source collection and final teacher
training costs. DPT labels for the8/16-case slices come from the final
20-case source actor. These are curves over **training-array amount**, not
claims that a DPT teacher learned from318 or603 chunks. Original interactions
are counted once across reuse; unchanged validation/source costs are never
presented as newly collected actions. No source policy is called an expert.

Optimizer scales25/100/500 are a separate axis. Hold the data-scale assembly
fixed when comparing those checkpoints, and hold optimizer budget/seed/model
fixed when comparing data amounts. A useful robot result still requires
separate fixed-weight support/query episodes, ordinary steering controls and
actual interaction costs; dataset exports and loss changes alone do not
establish adaptation, memory or high robotic performance.
