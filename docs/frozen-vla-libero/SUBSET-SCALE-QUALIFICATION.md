# Source-subset optimizer-scale qualification

`vla_source40_subset_training` uses an explicit `--qualification-subset` mode to
train AD/full, AD/static, DPT/full and DPT/static on the existing source subset.
It calls the normal source40 assembly through its explicit small-pilot route;
the full40 production admission gates remain unchanged. It does not collect
new robot interactions or promote the source actor to an expert.

The default launcher uses the completed Spatial00 capped training stream
(742 query chunks,80 episodes) and independent source-local validation
(617 chunks,68 episodes, nominal states40–49). The validation cases are
already exposed during NLL evaluation. Robot probes on those cases are
development evidence, not a blind generalization test.

Each controller follows one optimizer trajectory, stopping and resuming at
updates25,100 and500. The optimizer, sampler and Python/NumPy/Torch/CUDA RNG
remain in the full recovery; there are500 total updates per controller,
not25+100+500. Each milestone receives an immutable full recovery and a
fixed-update policy export. Validation-selected `candidate.pt` remains a
separate diagnostic and is not substituted for the milestone.

These are **optimizer-scale** checkpoints on identical source interactions.
They do not increase source-data diversity. The four controllers share
architecture, optimizer, sampling and update budgets; static controls are
trained separately. Source normalization is fit only on training data.

```sh
bash scripts/diagnostics/vla_source40_subset_training.sh <exact-SHA> <fresh-run-id>
```

Run inside an allocated Linux GPU job. The launcher checks the exact clean
commit, actual GPU allocation and focused Linux tests, then uses the qualified
CUDA overlay for the Torch students. The module caps each controller at300
cumulative training seconds and the overall operation at1500 seconds by
default, retaining partial recoveries on early stop. A resume uses the same
output identity and contract via the Python module's `--resume` argument.

`experiment/scales.json` links each scale's matrix-compatible `index.json` and
the common assembly, with exact hashes. It also records the original frozen
source-SAC checkpoint/contract and source costs counted once. The SAC baseline
uses its released pixel64/proprio8 actor; students use frozen VLA-prefix4104
features. Both steer the same frozen motor through the declared latent-noise
interface; the representation difference must remain explicit.

The downstream evaluator must compare actual support episodes with a separate
query, fixed motor/controller weights, native/SAC and trained static/reset
controls, and separate physical costs. Near-ceiling source successes and lower
NLL do not establish improvement, useful history or sample efficiency.
