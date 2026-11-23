# Released ReGround runtime on Baige

**Ready for offline reference inference on Baige.** On the RTX 4090 development machine, the released `reground.policies.vla.VLA` strictly loaded all pi05 checkpoint keys, took two live 256 × 256 RGB views, generated a finite 1 × 50 × 7 action chunk, and executed its first action in `spatial_0` with the released `rot_z/90` shift. The simulator advanced one step with a finite 25-dimensional state. CPU reset/render/step and a separate end-to-end CPU policy step also passed. [GPU policy step](runtime/qualification/dev-gpu-smoke.json) · [Device probe](runtime/qualification/dev-gpu-probe.json) · [CPU simulation](runtime/qualification/cpu-smoke.json) · [CPU policy step](runtime/qualification/cpu-policy-smoke.json)

The GPU host was `mwj-clean` (`dv-0d0ac1c839`), running driver 580.159.04. EGL reported the NVIDIA RTX 4090 renderer. Torch peak allocation was 9,523,264,000 bytes. The development check used **726 wall seconds on one card, 0.202 GPU-hours** of the 0.25 allowance, from 15:06:43 to 15:18:49 Shanghai. The cold shared-filesystem load exceeded the initial 620-second wrapper. The process continued under an 875-second wall guard and wrote the completed receipt after closing the simulator. On resumption the original outer `timeout` returned 124. The shared `dev01.log` contains the strict-load message and the same final receipt. [Wall-time record](runtime/qualification/dev-wall.json)


## Use the shared runtime

Run the following in Bash on Baige. The activation script sets the isolated environment, upstream import path, offline Hugging Face cache, LIBERO configuration and run-local interpreter caches.

```bash
root=/mowenjie/fluffy-bassoon-2/outputs/reground-rotation-reference-20261123-r1
source "$root/runtime/activate.sh"
cd "$root/upstream"
```

The released wrapper accepts an explicit revision. Native commands that omit it resolve the same commit. The noise seed, chunk index, observation processing, sampler and postprocessing all pass through the released implementation.

```python
from reground.envs.libero_env import TaskEnv
from reground.policies.vla import VLA

vla = VLA("pi05", "cuda", revision="7152f13e34c6c115d808aea0e5d08aca74eac14b")
env = TaskEnv("spatial_0", shift="rot_z/90")
try:
    reset = env.reset(init_id=0, seed=0)
    actions = vla.predict_chunk([reset["obs"]], [reset["description"]], [0], 0)
    step = env.step(actions[0, 0], render=True)
finally:
    env.close()
```

| Component | Verified identity |
|---|---|
| ReGround | `16a98486f560676b9036d1a6d067b9dd51a5917a` in `$root/upstream` |
| Runtime | CPython 3.12.15, uv 0.9.5, LeRobot 0.6.1, Torch 2.11.0+cu128, torchvision 0.26.0+cu128, TorchCodec 0.11.1 (CPU codec build), Transformers 5.5.4 |
| Policy | `lerobot/pi05_libero_finetuned@7152f13e34c6c115d808aea0e5d08aca74eac14b`, all 9 official files, 7,473,127,943 bytes |
| Tokenizer | `google/paligemma-3b-pt-224@35e4f46485b4d07967e7e9935bc3786aad50687c`, six tokenizer/processor files and the matching 1,027-byte `config.json` |
| Simulation | `hf-libero==0.1.5` supplies BDDL and initial states, `robosuite==1.4.0`, `mujoco==3.8.1`. The official `lerobot/libero-assets@0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` snapshot contains 586 verified files, 422,320,936 bytes. |

The model and tokenizer revisions matched their official current heads when checked on November 23, 2026. The paper does not identify its historical checkpoint SHA. For every listed file, setup reads its bytes on Baige and verifies the official Git blob identity. For LFS files it computes the payload SHA-256 and reconstructs the canonical pointer to check its Git identity. This also checks the gated tokenizer objects whose public listing masks the LFS SHA. The policy weights SHA-256 is `877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae`. [Model tree](https://huggingface.co/lerobot/pi05_libero_finetuned/tree/7152f13e34c6c115d808aea0e5d08aca74eac14b) · [Tokenizer tree](https://huggingface.co/google/paligemma-3b-pt-224/tree/35e4f46485b4d07967e7e9935bc3786aad50687c) · [Verified files](runtime/qualification/hf-verified.jsonl) · [Asset tree](https://huggingface.co/datasets/lerobot/libero-assets/tree/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6) · [Asset receipt](runtime/qualification/assets-verified.json)

## Offline resolution

The cache at `$root/hf/hub` exposes revision-specific snapshots and pins both `refs/main` files. LeRobot 0.6.1 accepts a revision in its pi05 loader, then looks up the weights through Transformers without forwarding that argument. Its tokenizer also uses a bare repository ID. The run-local refs make both lookups deterministic. `HF_HUB_OFFLINE=1` disables Hub network access. The sole metadata hook answers the released wrapper’s unqualified `HfApi().model_info("lerobot/pi05_libero_finetuned").sha` from the checked local ref. An explicit revision skips that call. [Offline hook](runtime/offline/sitecustomize.py) · [Upstream wrapper](https://github.com/umersiddique94/ReGround/blob/16a98486f560676b9036d1a6d067b9dd51a5917a/reground/policies/vla.py)

The policy snapshot links to `/mowenjie/fluffy-bassoon/inputs/pi05-libero-lerobot-v044-7152f13e-20261027-r1`. Tokenizer bytes come from the authenticated `/mowenjie/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c` cache. Transformers 5.5.4 requires the model config to select its tokenizer class. The matching official config comes from `/mowenjie/fluffy-bassoon-2/inputs/libero-witness-paligemma-pt-20261119-r1/config.json`. Only verified content is linked into the run-local cache.

The asset snapshot is copied under `$root/assets/<revision>` and linked at the installed package’s `libero/libero/assets` path. The released `python main.py setup` then writes `$root/libero-config/config.yaml`. The full asset list is checked independently of the directory names. [Asset manifest](runtime/libero-assets.tsv) · [Upstream setup](https://github.com/umersiddique94/ReGround/blob/16a98486f560676b9036d1a6d067b9dd51a5917a/main.py)

## Rebuild and qualify

From a Git checkout containing this run, deploy its setup files and the repository GPU probe to the shared run directory. For a separate rebuild, choose a new root and deploy a checkout of the exact ReGround revision there. The setup script can also clone the release where GitHub is reachable. It expects Linux x86-64 with a Python 3.12.15 interpreter, uv 0.9.5, system EGL/OpenGL with an appropriate graphics driver, and FFmpeg.

```bash
p=outputs/reground-rotation-reference-20261123-r1
root=/mowenjie/fluffy-bassoon-2/$p
rev=2bbc8ce4cca262c71a80b68b3651d32b961b583d
set -o pipefail
git archive "$rev:$p/runtime" | ssh baige-dev "mkdir -p '$root/runtime' && tar -xf - -C '$root/runtime'"
git archive "$rev" -- ':(literal)scripts/[train]gpu_probe.py' |
  ssh baige-dev "mkdir -p '$root/probe-source' && tar -xf - -C '$root/probe-source'"
```

On the Baige CPU development machine, the script can reuse the verified wheelhouse in `$root/wheels`. It installs all 150 pinned distributions from local files. On a fresh rebuild, direct PyPI and PyTorch URLs come from the package manifest. Baige’s mirror can supply PyPI bytes under the same publisher digests. The original upstream requirements file has SHA-256 `4d2cdcf5b9506b17396e94413ef296ad40e5207927e89f6f86af1858353315a2`. The parent install resolved the original requirements with `uv 0.9.5 --torch-backend cu128`. The public downloads were slow, so the recovery used that same resolution and checked the CUDA Torch, torchvision and Triton wheels against the official PyTorch hashes. The byte-verified wheels, original `install01.log`, successful `install02.log` and replay `install04.log` are on the shared root. [Rebuild script](runtime/setup.sh) · [Package URLs and hashes](runtime/packages.tsv) · [Version lock](runtime/requirements-linux.txt)

```bash
export REGROUND_RUNTIME_ROOT=/mowenjie/fluffy-bassoon-2/outputs/reground-rotation-reference-20261123-r1
root=$REGROUND_RUNTIME_ROOT
REGROUND_PYPI_MIRROR=http://mirrors.baidubce.com/pypi bash "$root/runtime/setup.sh"
source "$root/runtime/activate.sh"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 EGL_PLATFORM=surfaceless \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python "$root/runtime/qualify.py" --device cpu --output "$root/cpu-smoke.json"
```

On the dedicated single-GPU development machine, `timeout -k 5s 875s bash "$root/runtime/dev-smoke.sh"` first runs the repository’s GPU probe by device UUID, then binds native inference and EGL to the single visible card. The smoke requires LeRobot’s strict all-keys-loaded confirmation, a finite 1 × 50 × 7 chunk from two live images and a subsequent real simulator step. [GPU entry point](runtime/dev-smoke.sh) · [Qualification code](runtime/qualify.py)

All 150 installed package versions matched the original lock and `uv pip check` passed. The CPU host also executed a torchvision NMS operation, decoded an 80-sample PCM waveform through TorchCodec/FFmpeg, ran a SciPy norm, and tokenized a string offline through `GemmaTokenizer`. [Package checks](runtime/qualification/package-check.json)

The GPU smoke used the setup at `8bbfae90`. Revision `2bbc8ce4` added the optional `--cpu-policy` check to the same qualification script. The pinned upstream worktree was clean after qualification.
