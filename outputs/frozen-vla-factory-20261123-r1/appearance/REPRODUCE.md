# CPU reproduction

Run on Linux with software EGL (Mesa), Python 3.11.18 and uv. The repository baseline is `757a44cd4374833cad790852df8b87ddf412e638`. Start in this directory and provide an absolute external scratch path in `WORK`. The rendering and policy environments use different NumPy versions. The lock files list the actual installed distributions. The render environment installs the dependencies needed for these static LIBERO scenes only.

```bash
set -eu
: "${WORK:?set WORK to an absolute scratch path}"
mkdir -p "$WORK"
uv venv --python 3.11.18 "$WORK/sim"
uv venv --python 3.11.18 "$WORK/env"
uv pip install --python "$WORK/sim/bin/python" --no-deps -r render-requirements.txt
uv pip install --python "$WORK/env/bin/python" --no-deps --index-url https://download.pytorch.org/whl/cpu torch==2.7.1+cpu torchvision==0.22.1+cpu
uv pip install --python "$WORK/env/bin/python" --no-deps -r policy-requirements.txt
python fetch.py "$WORK"
pkg=$("$WORK/sim/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
ln -sfn "$WORK/assets" "$pkg/libero/libero/assets"
export LIBERO_CONFIG_PATH="$WORK/libero-config"
printf 'n\n' | "$WORK/sim/bin/python" -c 'import libero.libero'
export PYTHONHASHSEED=0 NUMBA_DISABLE_JIT=1 LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=egl
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$WORK/sim/bin/python" render.py "$WORK"
"$WORK/env/bin/python" measure.py "$WORK" "$WORK"
"$WORK/env/bin/python" summarize.py "$WORK"
```

The generator uses all ten BDDL files from the released LIBERO-Goal suite in filename order, seeds 3200–3209, and both arms from robosuite 1.4.0. The source prompt follows LeRobot's LIBERO filename-to-language convention. The motor uses seeds 9100–9109 and a CPU torch generator with an explicit `(1,50,32)` noise tensor per pair. The same-seed flows take the checkpoint's ten integration steps. Raw tensors and generated frames stay under the scratch path.

The model checkpoint, external tokenizer/config and LIBERO assets have immutable Hub revisions in `fetch.py`. The model SHA-256 is checked before strict loading. Using the full LIBERO-finetuned checkpoint means the separate base-VLM weight file is unnecessary. The official environment processor rotates both images 180 degrees and forms the eight-dimensional end-effector pose/gripper state. The checkpoint's preprocessor applies renaming, newline, tokenization, device placement and normalization. Its postprocessor converts the seven returned channels to unnormalized policy action coordinates. The local processor path and CPU device override select the pinned tokenizer and execution device.

The archive of task scores contains within-task RMS differences, not signed velocity or distance in meters. `first_xyz_rms` uses the three translation components at step zero. `xyz_rms` uses all 50 steps × 3 components. Rotation and gripper scores use their own channel groups. `action_rms` combines seven channels with different scales and often emphasizes the gripper. `summary.json` reports equal-weight task means, percentile task-bootstrap intervals and the number of positive values among ten tasks. It resamples the same task indices for paired differences (20,000 draws, seed 20261123).
