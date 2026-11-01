# RoboCasa365 environment setup, independently of policy loading

2026-11-01, static source handoff. This supplies the concrete inputs for the
existing [CPU admission](ROBOCASA365-SETUP-CPU-ADMISSION.json): one fresh
0-GPU/16-CPU/64-GiB worker,29 scheduler minutes, at most1,800 inclusive worker
seconds,64 GiB new storage, zero simulator actions or optimizer updates.
`finish_assembly` owns deployment, actual allocation/proof, deadlines, storage,
submission and teardown. No setup, package import, model load, test or simulator
was executed by this source inspection.

## Exact native source and dependency inputs

[SETUP-SOURCE.json](SETUP-SOURCE.json) binds these sources:

- RLinf `d80cf00e95e545f579e33432e7d7f14e86bb39d7`, its current official
  `requirements/install.sh` and benchmark-native `robocasa365` adapter.
- RoboCasa `3443cdf3dd20f9ae8ed41adba442059b506891d0`, declared package
  **1.0.1**. This is a concrete current registry/setup candidate, not a claim
  to reproduce the leaderboard's1.0.0 environment. Literal imports and the
  selected checkpoint's actual compatibility still need qualification.
- Robosuite `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`.
- The existing installer's OpenPI dependency `RLinf/openpi` at
  `c5dc4b9296a1a4739bf52828f28a579f12dce763`.

Use [RLINF-SETUP-PINS.patch](RLINF-SETUP-PINS.patch), exactly two floating Git
dependency URL replacements. Apply it against the hash-bound d80 installer in
a **fresh** checkout, commit that exact delta through Git, and record the new
deployment commit before submission. No active source or old environment is
modified. Pre-stage a detached RoboCasa3443 checkout and set `ROBOCASA_PATH`:
the official installer then reuses it without its optional main-branch pull.
The pinned model-reference OpenPI5a6 source is separate from this environment
dependency; never describe them as the same implementation.

The native setup.py requires NumPy2.2.5,Numba0.61.2,SciPy1.15.3,MuJoCo3.3.1
and LeRobot0.3.3. Preserve those in this new isolated environment; the old
NumPy1.26 LIBERO/RoboTwin environment must not be overlaid or moved. The official
installer also pins dm-control1.0.28 and supplies qpsolvers/quadprog. Its resolved
Torch,OpenPI,Transformers,Ray and remaining dependency versions/origins must be
saved after the actual install; the static recipe is not a completed lock.

Literal invocation after exact code/storage admission, using owner-bound
absolute fresh paths:

```sh
export ROBOCASA_PATH="$PINNED_ROBOCASA_SOURCE"
export ROBOCASA_ASSETS_PATH="$FRESH_ROBOCASA_ASSETS"
export DOWNLOAD_DIR="$FRESH_PROJECT_CACHE"
export HF_HOME="$FRESH_PROJECT_CACHE/huggingface"
export UV_CACHE_DIR="$FRESH_PROJECT_CACHE/uv"
bash "$PINNED_RLINF_SOURCE/requirements/install.sh" embodied \
  --model openpi --env robocasa365 --python 3.11.14 \
  --venv "$FRESH_ENV" --no-flash-attn --no-root --install-rlinf
```

Use the actual existing package cache where verified to fit the finite ticket;
never modify a shared environment. The installer already downloads only the
OpenPI tokenizer in its `--assets openpi` stage, not policy weights. No scorer,
actor checkpoint, JAX base download/conversion, CUDA computation, EGL context,
environment constructor, reset or physics belongs to this CPU stage. Keep all
setup subprocesses inside its original wall clock; no automatic retry/reset.

## Public assets and the64-GiB storage boundary

[ASSET-MANIFEST.json](ASSET-MANIFEST.json) pins91 public ZIP archives,
**11,213,250,500 compressed bytes**: five official packs plus17 Lightwheel
fixture and69 Lightwheel object packs. The datasets/revisions are
`robocasa/robocasa-assets@1b92c3d02ca4354984fec961357db0bff7b32166` and
`nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF@420a04af939c34873e6839a586b70844baf28aab`.
Per-file size and published LFS SHA256 are retained. This is a fresh compact
metadata audit, not a download, extraction or local availability claim.

Fetch only those exact revisions/files. Preserve the official layout:
textures→`textures/`,generative_textures→`generative_textures/`,objaverse→
`objects/objaverse/`,aigen_objs→`objects/aigen_objs/`,Lightwheel fixtures→
`fixtures/`,Lightwheel objects→`objects/lightwheel/`. Retain fixtures.zip as the
official recipe requests, without overwriting the benchmark Lightwheel fixture
tree. Do not silently switch to Lightwheel-only object sampling to fit storage.

Internal upper bands are12 GiB retained archives,24 GiB extracted assets,
20 GiB isolated environment/sources and8 GiB temporary/receipts. Before any
extraction, sum the actual ZIP central-directory sizes and check paths/links,
actual shared capacity/quota and exclusive write/flush/fsync/unlink. Stop cleanly
if a band or the64-GiB total cannot hold; these bands are bounds, not measured
expanded sizes. Existing verified assets can be reused by exact identity in
place, charged once. Do not scan unrelated corpora, delete shared data, expand
the storage grant or overwrite existing asset directories.

## Literal import and registry checks inside the allocated CPU job

First retain the actual0-GPU allocation proof before any artificial visibility
mask. Then use this fresh environment, with the pinned RLinf source on its
normal import path:

```python
import importlib, importlib.metadata as md, json
modules = ["numpy", "mujoco", "robosuite", "robocasa", "openpi",
           "rlinf.envs.sim.robocasa365.robocasa365_env"]
origins = {name: importlib.import_module(name).__file__ for name in modules}
versions = {name: md.version(name) for name in
            ["numpy", "numba", "scipy", "mujoco", "lerobot", "torch", "ray"]}
from robocasa.utils.dataset_registry import TARGET_TASKS
from robocasa.utils.dataset_registry_utils import get_task_horizon
names = list(TARGET_TASKS["composite_seen"])
print(json.dumps({"origins": origins, "versions": versions,
                  "composite_seen": names,
                  "native_horizons": {n: get_task_horizon(n) for n in names}}))
```

These are imports and registry metadata, not simulator/model construction.
Retain actual source Git identities, resolved dependency freeze, asset hashes,
expanded/output sizes, import stdout/stderr, cleanup and actual scheduler cost.
There is no unrelated old test suite to rerun. Any new affected numeric adapter
tests must use their exact later source pin; none is claimed passed here.

## The next native-policy phase stays distinct

The human-supplied Aaron Composite-Seen π0.5 checkpoint/source/access is under
the existing protocol/runtime owners. It is not replaced by the documented
RLinf π0 checkpoint, generic LeRobot weights, Motus or new SFT. Public reference
OpenPI5a6 uses the16-D relative-eef/base/gripper order and12-D eef/gripper/base/
mode action order matching RLinf's existing365 policy boundary; it pads state
to32 **before** normalization/tokenization and preserves mean/std. Generic
LeRobot's different order/epsilon must not be carried without a selected-model
binding. Three actual camera slots and mean/std/padding remain the small
compatibility delta to qualify, not a new trainer or simulator framework.

For a prospective Composite-Seen candidate, `ScrubCuttingBoard` has a native
registry horizon1,200 at the proposed RoboCasa pin and is in the16-task exposed
set. This is a proposal before any outcomes, not known headroom or an OOD task.
Native8 cases would have at most9,600 control actions, with qualification charged
as the first case. At16 training environments ×4 complete episodes ×1,200
actions, each PPO round costs76,800 training actions; a20-round claim would
exceed1M. Whole-batch/global1024 divisibility also needs the official budget
resolver at H50 (1,536 chunk samples here). Do not silently truncate episodes
or claim the upstream400 field overrides the registry. The parent binds the
small matched pilot's final task/batch/update/resource ticket after native
qualification; this CPU setup ticket launches no policy or PPO.
