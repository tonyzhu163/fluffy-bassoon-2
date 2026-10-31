# Official LIBERO-Pro environment-family resolution

Static/read-only finding, 2026-10-31. Source is clean at
`eafdb809426b13153aa1e4c42d6601844217dfec`. The original full public HF stage
and its READY hash remain unchanged.

**The remaining 40 cases require dedicated generated BDDL/init files.** They
are not runtime-config aliases to original LIBERO initialization banks.

| Official family | Registry entries | Source env BDDL/init pairs | HF env BDDL/init pairs |
| --- | ---: | ---: | ---: |
| libero_goal_env | 10 | 0 | 0 |
| libero_spatial_env | 10 | 0 | 0 |
| libero_10_env | 10 | 0 | 0 |
| libero_object_env | 10 | 0 | 0 |

The other 16 registered families have all 160 BDDL/init pairs in the pinned HF
stage. All 40 original BDDLs needed by the official environment transform are
present in the pinned source. Exact registry order, required relative paths,
original BDDL hashes and official implementation-file hashes are in
`receipt.json`. This is structural path resolution only; no goals were
reviewed, filtered or excluded.

Official source evidence:

- `libero/libero/benchmark/__init__.py:143–157` builds each task with its own
  `problem_folder`, `<stem>.bddl` and `<stem>.pruned_init`.
- `benchmark/__init__.py:246–273` resolves both files within that family folder;
  `:858–885` registers the four `_env` families. The registry entries are in
  `benchmark/libero_suite_task_map.py:1342–1390`.
- `perturbation.py:322–418` implements `EnvironmentReplacePerturbator`, changing
  environment, problem-token and fixture text. At this exact pin, line 410
  assigns the literal `living_room_table`; retain that pinned implementation.
- `perturbation.py:582–624` exposes the pure text transform
  `process_bddl_file_mixed`. Its output folder is `<suite>_temp`.
- `perturbation.py:561–574` invokes the official
  `notebooks/generate_init_states.py`. That script imports
  `OffScreenRenderEnv` and constructs an environment for every requested init
  (`:27`, `:53–68`). The default is 50 inits per case: 2,000 simulator
  constructions for all 40 cases. It requires no VLA/model weights.

The existing single-perturbation runner expects `<suite>_env`, while
`create_env()` emits `<suite>_temp`. A fresh generation must explicitly stage
the official transform's outputs under the registered `_env` folder names;
the wrapper does not perform that naming step. Passing an explicit integer
`seed` also avoids its `configs.get("seed", int)` default. The init generator
has no explicit seed argument and catches per-init errors, so 50 requested
states is not proof of 50 successful or physically distinct states.

The minimal future execution route, **inside an allocated simulator/render
runtime**, is:

```python
# All paths below belong to a fresh generation; original source/inputs stay intact.
from pathlib import Path
import shutil
from perturbation import PerturbFlags, process_bddl_file_mixed

for suite in ("libero_goal", "libero_spatial", "libero_10", "libero_object"):
    base = NEW_ROOT / "original_bddl" / suite
    shutil.copytree(PRO_SOURCE / "libero/libero/bddl_files" / suite, base)
    generated = process_bddl_file_mixed(
        input_dir=str(base), task_suite_name=suite,
        flags=PerturbFlags(use_environment=True),
        configs={"environment": str(PRO_SOURCE / "libero_ood/ood_environment.yaml")},
        seed=28,
    )
    shutil.copytree(generated, NEW_ROOT / "bddl_files" / f"{suite}_env")
```

Then run the unmodified official generator for each of the four suites:

```bash
"$SIM_PY" "$PRO_SOURCE/notebooks/generate_init_states.py" \
  --bddl_base_dir "$NEW_ROOT/bddl_files/${suite}_env" \
  --output_dir "$NEW_ROOT/init_files/${suite}_env" \
  --num_inits 50 --height 128 --width 128
```

`PRO_SOURCE` is the pinned repository; `SIM_PY` must be its qualified simulator
Python. Put the 16 existing HF family directories into this fresh composite
root by explicit references/copies, preserving the HF stage. Set a fresh
`LIBERO_CONFIG_PATH/config.yaml` before importing LIBERO, with `bddl_files` and
`init_states` pointing at the composite root, and `assets` at the pinned
source's `libero/libero/assets`. The config keys are `benchmark_root`
(`PRO_SOURCE/libero/libero`), `bddl_files` (`NEW_ROOT/bddl_files`),
`init_states` (`NEW_ROOT/init_files`), `assets`
(`PRO_SOURCE/libero/libero/assets`), and `datasets`
(`PRO_SOURCE/libero/datasets`). This avoids the package's default interactive
global config creation. A later metadata check can resolve all 200 registered
pairs using the AST registry without importing environments. Generated state
counts and runtime usability require allocated verification.

This cannot be completed by a small **metadata-only** zero-GPU job. It needs
MuJoCo/robosuite and offscreen rendering. The current qualified runtime uses
GPU/EGL; a zero-GPU software-rendering route has not been qualified here.
Runtime/cost for the 2,000 constructions has not been measured, so no new GPU
budget or launch is implied. Hold that simulator generation for an explicitly
bounded allocated run; source collection remains independent.

No official-package/environment imports, tensors, pickle loads, simulators, tests,
generated data, input edits or allocations were executed in this inspection.
The existing stage receipt remains immutable, and full200 runtime readiness or
a full-suite score is not claimed.
