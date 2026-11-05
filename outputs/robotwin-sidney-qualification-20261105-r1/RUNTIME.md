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
