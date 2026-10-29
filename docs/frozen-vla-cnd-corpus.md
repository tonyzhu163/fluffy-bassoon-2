# Paired CND/DPT corpus for frozen π₀.₅ steering

`collect_noise_distillation` keeps its existing flat `{train: Path,
validation: Path}` return value for CND datasets. Set `export_paired_dpt=True`
to additionally write `dpt-train.npz`, `dpt-validation.npz` and
`paired-collections.json`. This requires no additional teacher queries or
robot rollouts. CND targets remain the actually executed noisy mixtures; DPT
targets are the deterministic expert coordinates already recorded at those
same pre-action observations. Observations, executed noise, actual rewards,
episode boundaries and source update identities stay bitwise identical.

Pass `benchmark_contract` with `suite`, `task_id`, `semantic_task` (canonical
BDDL stem), `bddl_sha256`, and `teacher_contract_sha256` (the caller's complete
source-teacher contract hash). The caller must derive these from the actual
simulator and teacher recovery contract, rather than task labels supplied by a
policy. The paired receipt records actual trial/scene identities, dataset
checksums, motor receipt, feature/noise hashes and chunk length. These are
provenance only; they never enter student observations.

`vla_cnd_corpus` assembles a complete corpus or a first partial shard against
the frozen `outputs/libero-pi05-corpus-20261029-r1/split.json`. It verifies that
file's metadata content hash, currently
`4599ca659a48e94e35a22641f0bb90477d958aed0570d0df46a021ef4492fe08`.
The shard JSON has this form:

```json
{
  "format": "frozen_vla_cnd_corpus_shard_v1",
  "collections": [
    {
      "task_key": "libero_90:57",
      "collection_manifest": "/path/to/task57/cnd-data/paired-collections.json",
      "teacher_contract_sha256": "actual full teacher contract SHA-256"
    },
    {
      "task_key": "libero_90:02",
      "collection_manifest": "/path/to/task02/cnd-data/paired-collections.json",
      "teacher_contract_sha256": "actual full teacher contract SHA-256"
    }
  ]
}
```

Each entry may also provide `collection_manifest_sha256` to pin its paired
receipt before assembly. Source collections must share one released
`pi05_libero` motor receipt, canonical feature-v2 contract, bounded noise
contract and executed chunk length. The assembler rejects held-out/excluded
tasks, wrong BDDL/teacher identities, changed coordinates, duplicate task
collections or trial identities, out-of-bank source states, inconsistent costs
and changed CND/DPT contexts. Different tasks in one semantic family may
coexist within the same frozen partition.

Both local scene splits inherit their task's global partition. In particular,
a training task's locally named `validation` histories remain global training
data when their states belong to its declared source bank 0–39. They cannot
become primary held-out-task validation data. States 40–44 and 45–49 are
reserved for the separate familiar-task checks and cannot silently enter this
source corpus. Every complete multi-episode trial stays intact. A partial
shard must contain at least one training and one validation task; the full
frozen task manifest and its counts remain the definition of the experiment.

Run assembly on allocated Linux CPU resources:

```bash
python -m fluffy_bassoon.meta_rl.vla_cnd_corpus \
  --split outputs/libero-pi05-corpus-20261029-r1/split.json \
  --shard /path/to/first-shard.json --output /fresh/corpus
```

The output contains CND/DPT train/validation datasets and `collections.json`.
Source teacher and collection interactions are counted once per task across
both objectives and both local scene splits. Training and validation source
costs are also recorded separately. Aggregated source cost is dataset
generation cost, not additional interactions consumed by each objective.
Fitting normalization and student gradients uses only global training data;
checkpoint selection uses global validation. Test tasks and final Plus/Pro
feedback cannot participate in either operation.
