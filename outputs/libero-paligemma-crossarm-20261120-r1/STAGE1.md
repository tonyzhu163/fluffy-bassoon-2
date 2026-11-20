# Independent source motion pretraining

Can a reader fitted on Panda and Sawyer predict each arm’s motion on a fresh reset more accurately when it receives that arm’s practice? The reader can fit while the motor trains. The existing motor qualification gate still applies to joint fitting. [Evaluation protocol](EVAL.md)

## Run on the allocated Linux runtime

From this run’s code directory, use the existing Torch and native LIBERO simulation environment configured by the launcher, including `CROSSARM_SIM_PYTHON` when the simulator has its own interpreter. Supply a fresh output path and the deployed revision. [Runtime setup](run.sh)

```bash
python pretrain_reader.py --out "$PILOT_DIR/reader-stage1" --revision "$CODE_REVISION"
```

Add `--from-stage DIRECTORY` to reuse existing native `practice-{body}-{pose}.npz` tapes. The command checks only the ten expected names, copies available tapes into its output, and collects missing tapes with the native simulator. Eight tapes from an earlier source fit therefore require just the two pose-28 resets. Ten matching tapes allow a full replay without simulation. A present tape with inconsistent reset labels, native task identity, commands or transition records stops the run.

## Fitting and independent check

| Role | Poses per arm | Use |
| --- | --- | --- |
| Original source supports | 20, 21 | Encode 16 command–motion pairs per reset. Compute the seven motion scales across both arms’ supports. |
| Source fitting queries | 22, 23 | Predict observed native motion on a separate reset. |
| Held-out source check | 28 | Score all 16 commands on each arm after saving update 100. |

The command follows the initialization and sampling order in `adapt_eval.fit`. It initializes `ReaderAdapter` with Torch seed 41, uses one Torch thread, and creates the NumPy sampler with `FIT_SEED`. Each scale is the root mean square of its support motion coordinate, computed in float64 and floored at `1e-6`. The shared `train_stage(..., motion=True)` runs 100 Adam updates of 32 examples. The report includes the fixed training-batch MSE before and after fitting, sampled minibatch MSE at updates 1, 50 and 100, and the reader parameter change. [Shared implementation](adapt_eval.py)

On pose 28, both arms execute the same pose-seeded random commands. For each query arm and each original support pose, the frozen response head scores three contexts using the same commands and measured motions. **Correct** uses the query arm’s original support. **Wrong** uses the other arm’s original support from the same pose. **Zero** supplies a zero embedding directly to the response head.

Each score averages squared prediction error over 16 commands and seven coordinates in support-normalized motion units. The per-arm result averages the two support-pose scores, and the aggregate weights each arm equally. The two supports share one physical query episode per arm. Lower correct-context error than both controls on both arms supports arm-specific prediction on this pose. A tie or reversal against either control counts against that interpretation. This check measures source motion prediction on one layout. Task success and transfer require the later native evaluation.

## Joint fitting handoff

`report.json` records completion status, the deployed revision, package versions, fit curve, tape origins and digests, checkpoint digest, and pose-28 scores. `reader-stage1.pt` stores the full reader, response head, initial adapter, motion scale, NumPy sampler state and Torch CPU RNG state immediately after the final source update.

For a completed stage and a qualified motor, the parent runner can restore the update-100 state as follows. Here `stage_dir` is the stage output as a `Path`. Compare the checkpoint and support tape digests with the stage report when consuming them.

```python
import numpy as np
import torch
from adapt_eval import FIT_SEED, ReaderAdapter, SOURCES, SUPPORT
from pretrain_reader import read_tape

checkpoint = torch.load(stage_dir / "reader-stage1.pt", map_location="cpu", weights_only=True)
torch.set_num_threads(1)
net = ReaderAdapter().train()
net.load_state_dict(checkpoint["state_dict"])
torch.set_rng_state(checkpoint["torch_rng_state"])
rng = np.random.default_rng(FIT_SEED)
rng.bit_generator.state = checkpoint["rng_state"]
practices = {body: [read_tape(stage_dir, body, pose) for pose in SUPPORT] for body in SOURCES}
```

Pass `net`, `practices` and the restored `rng` to the existing joint `train_stage(..., motion=False)` with the qualified motor’s demonstration queries. That stage constructs its own Adam optimizer for the reader and action adapter. [Standalone command](pretrain_reader.py)
