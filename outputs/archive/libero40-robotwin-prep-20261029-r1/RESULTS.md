# Preparation results, 2026-10-29

No model computation, GPU qualification, new GPU job or training was performed
by this preparation. The current recommendation is RoboTwin as the primary
candidate, conditional on recoverable failures and a competent common motor.
The LIBERO transfer contract remains available independently.

## LIBERO static split

Builder source `589db3e22ee2aa352840f7ab8ecbeba85266e6ae`, reviewed main
`4d4f46f4cdcd8cfd7c7056e95bf85d0a90e0ed59`. Executed on the allocated Baige
development machine against clean official LIBERO
`87edbd1b823e464bc66557d213cc45e9f88aa543`.

| Partition | Tasks |
| --- | ---: |
| Source: Spatial/Object/Goal/LIBERO-10 | 40 |
| Strict LIBERO-90 target | 68 |
| LIBERO-90 full-task overlap controls | 22 |

Full task equivalence is conservative and syntactic: transitive normalized
full goal plus referenced object bindings, or full normalized instruction.
It permits shared scenes and atomic skills and is not an exhaustive semantic
equivalence proof. All 90 can be reported secondarily; they are not 90
independent unseen full tasks under this rule.

Manifest content hash: `7554d6bdd570527a840e020bb45c5721e051b77a5f97f2f9cce6ccfcfc0a3be1`.
Static source inventory: `4715ee335abb30f7fdbbf88dfe17c1490d9449ac5d233f7dde101fb70e666b91`.
Three allocated Linux leakage-regression tests passed in 0.07 seconds.
Evidence: [split](split.json), [tests](tests.log).

## RoboTwin CPU environment

Inventory source `ee36dbbf483c6c983c11fb1ace966eedd404ef3b`; dependency
sources retain RLinf `c605b9d` and RoboTwin `dca9ec6`. Environment on Baige:
`/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1`, 1.6 GiB locally,
with a read-only `.pth` reference to the existing shared runtime.

All six seen and six unseen task modules, the RLinf RoboTwin adapter, model
factory and MPLib/SAPIEN integration imported successfully. Required asset
roots and cached checkpoint shard paths/normalization JSON exist. These
checks do not validate all assets by simulation or load the model.

| Package | Actual version |
| --- | --- |
| Torch | 2.11.0+cu128 |
| NumPy | 1.26.4 |
| SAPIEN | 3.0.1 |
| MPLib | 0.2.1 |
| Open3D | 0.19.0 |
| Transformers | 4.55.4 |
| rlinf-openpi | 0.1.1 |

The first CPU attempt discovered missing Open3D. It was installed in this
new overlay with its dependencies; the corrected setup script records that
requirement. Shared package-version receipts before/after are identical;
the shared MPLib remains 0.1.1. Both pinned dependency checkouts and the new
project run checkout remained clean for tracked files.

Optional Curobo/PyTorch3D are absent; the selected planner is MPLib. SAPIEN
warned that the development machine lacked a Vulkan ICD. No renderer or
simulator was instantiated, so a real allocation-local episode is still
required. The old GRPO smoke's renderer caveat remains unresolved here.

Evidence: [CPU receipt](runtime/setup.json), [import log](runtime/cpu-imports.log),
[before](runtime/shared-before.json), [after](runtime/shared-after.json).
CPU receipt SHA256: `06d52b34800dcee93540df3e01a9702879060716612c84db5f26f95e4523f165`.
Both shared-version receipts SHA256:
`19ffe3944b461b0e4402026951f33ac08cfc4a91cdbcb57ee0646da072a1f1ae`.

## Common motor remains unqualified

The cached adjust_bottle motor is a runtime initializer only. Public model
metadata and bounded safetensors headers were inspected for candidate common
motors without downloading weights, executing pickle metadata or loading
models. [Candidate receipt](motor-candidates.json) pins the resolved revisions.

- `motus-robotics/pi0.5_robotwin2`: card declares 50 tasks, clean and randomized
  training, delta joint actions and horizon 32. Its safetensors header has
  812 tensors. It is not the paper's documented horizon-50 interface.
- `madokalif/pi05-robotwin2-clean50-sft`: card declares 50-task full SFT from
  the Motus checkpoint; header has 812 tensors. Complete inference conventions
  and RLinf compatibility are unqualified.
- `wzzzq/robotwin-pi05-multitask-sft-h16-clean50`: card identifies horizon 16
  and RLinf/OpenPI export; weights use PyTorch checkpoint files. They were not
  deserialized. Complete training-task coverage is unverified.

The cached RLinf adjust_bottle index contains 821 parameter keys. Similar
model families alone do not establish compatible loading, normalization or
action conventions. None of these candidates is a verified VLRLA initializer.

Next: qualify a common motor's interface and competence on source-task
development scenes, verify that ordinary steering can improve independently
measured success, then collect real ordered RL histories with failures kept.
Large AD/DPT or recurrent meta-RL training is premature before this gate.
