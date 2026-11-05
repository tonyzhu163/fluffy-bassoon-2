# Runtime preparation

Model process candidate: official LeRobot commit
`92f96f33b3a842ead3a115cc4f701ba2ad8aa3b5`, the last main commit before the
checkpoint card's publication time. This is a reproducible runtime candidate,
not a claim that the checkpoint author trained at that exact source commit.
It understands the saved disabled relative-action processor definitions.

Created a fresh uv environment at
`/mowenjie/fluffy-bassoon-2/envs/lerobot-sidney-92f96f33-20261105-r1`, using the
existing LeRobot Python 3.12 interpreter. Installed the pinned source editable
with `uv pip install --no-deps`, then `transformers==5.5.4`,
`sentencepiece==0.2.1`, and `numpy==1.26.4`. A `sidney-base.pth` adds the
unchanged `/mowenjie/fluffy-bassoon/envs/lerobot-v044-8fff0fde/lib/python3.12/site-packages`
for retained Torch 2.7.1+cu128, torchvision 0.22.1, SciPy 1.14.1, and other
base dependencies. The new environment's packages and explicit new LeRobot
PYTHONPATH take precedence. `RUNTIME-FREEZE.txt` records the overlay;
`RUNTIME-ALL.txt` inventories both layers, including shadowed versions.
The actual loaded model runtime versions are written to allocated `READY.json`.
No shared environment was modified.

Use the unchanged RoboTwin Python 3.11 simulator runtime and external source
pins from the earlier qualification. Exchange NumPy arrays through job-output
files and a private child process stdin; no network service or port is opened.
The simulator and model share only the scheduler-assigned GPU.

Use the retained PaliGemma tokenizer snapshot
`35e4f46485b4d07967e7e9935bc3786aad50687c` in the existing Hugging Face cache,
recording its file hashes at runtime. Offline mode prevents a moving tokenizer
revision or a gated download during the job. Protect that retained snapshot
while any queued or active qualification depends on it.

Preparation used only Git, downloads, package installation, file hashes and
package metadata. No model computation, imports for qualification, simulator
episodes or tests ran on the development host. Native runtime support remains
unqualified until the allocated job succeeds.

The first allocated attempt (`qual01`) stopped at the probe because this Baige
container left CUDA_VISIBLE_DEVICES unset. Its wrapper also omitted the
LYCO_BAIGE_GPU_MAP export needed by renderer binding. Both were already
present in the repository's qualified RoboTwin launchers. The retry uses
ordinal zero only when CUDA_VISIBLE_DEVICES is unset, retains the scheduler's
NVIDIA_VISIBLE_DEVICES allocation unchanged, runs the same full GPU probe,
and exports that probe's map. It also links the retained PhysX cache in the
fresh container, matching those launchers. A processor-only numerical/import
check runs in the allocation before constructing the model; it neither runs
a model on CPU nor substitutes for the actual GPU episode.

`qual02` passed the scheduler/CUDA/PCI/PID probe on the actual GPU, then
failed during the processor import check: the retained pandas requires
NumPy >=2.0.2, while the first overlay pinned 1.26.4. No model was constructed.
For qual03, create a fresh environment ending `20261105-r2`, using NumPy
2.2.6 to match the retained Python3.12 base (also within SciPy1.14.1's
NumPy range), keeping LeRobot, Transformers, Torch and simulator pins.
The old overlay and failed attempt remain immutable. Record package metadata
constraints and the new overlay freeze separately in `RUNTIME02-*`.
The metadata audit also pins Pillow12.0.0, packaging25.0 and fsspec2026.2.0
to satisfy the installed LeRobot/datasets requirements; it reports zero
non-extra requirement conflicts across the 15 inspected packages. This is
metadata validation, not a successful native import or model qualification.

## Deterministic reference retry (qual08)

Qual07 shares one explicit Gaussian tensor between chunk and queue APIs, yet decoded predictions differ by maxabs0.0047388077. A third chunk prediction also changes. Parameter versions remain unchanged; no policy actions or SGD executed. This is not evidence of an incorrect checkpoint or of a specific CUDA defect.

Qual08 holds weights, H50/K10, BF16 and saved mean/std fixed. It explicitly selects eager attention for all Transformers modules, enables Torch deterministic algorithms and cuDNN determinism, disables TF32 and fused SDP kernels, and sets CUBLAS_WORKSPACE_CONFIG=:4096:8 before Torch initialization. Input and model-buffer hashes localize hidden mutation. The original1e-5 parity tolerance is unchanged. These are numerical reference settings rather than the model card's compiled runtime.

Reference: https://docs.pytorch.org/docs/2.7/notes/randomness.html
