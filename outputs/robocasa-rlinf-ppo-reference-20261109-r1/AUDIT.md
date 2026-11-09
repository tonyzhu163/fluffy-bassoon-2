# Historical PPO recipe audit and frozen-policy scope

The user clarified the final goal: retain pi0.5 as a frozen ICRL policy.
The new VLA fine-tuning/cadence-training proposal and OpenVLA migration
recommendation are withdrawn. No training allocation has launched for this
run. This file retains the recipe research and qualified compatibility fix;
it is not an active training plan. The native SFT reproduction continues
under its immutable manifest; existing held RL runs remain untouched.

The original OpenPI SFT evaluation already uses C5. Its submission-linked
[default](https://github.com/robocasa-benchmark/openpi/blob/ca4c6d710db75e276bc7c866a57bd7e4aee5b6e8/examples/robocasa/main.py#L34)
and execution slice agree; the [maintainer](https://github.com/robocasa-benchmark/openpi/issues/1)
explicitly confirms the reported experiments inherited five controls.

Current upstream RLinf source inspected at5c415fa28bb25074fab68f0d09baa73d1b67c608:
[OpenDrawer PPO](https://github.com/RLinf/RLinf/blob/5c415fa28bb25074fab68f0d09baa73d1b67c608/examples/embodiment/config/robocasa365_opendrawer_ppo_openpi.yaml).
It selects pi0,not pi0.5. Its relevant settings are:

| Setting | Upstream OpenDrawer PPO | Historical local pi0.5 PPO |
| --- | --- | --- |
| Executed controls / denoising steps | 5 / 5 | 50 / 10 |
| Scene split | pretrain | target |
| Training / evaluation controls per episode | 400 / 750 | 600 / 600 |
| Environment count × rollout rounds | 32 × 4 | 128 × 1 |
| Global minibatch / PPO epochs | 2048 / 4 | 256 / 4 (one-epoch Stage1 also tested) |
| Actor / critic learning rate | 5e-6 / 1e-4 | same |
| Discount / GAE lambda / policy clip | .99 / .95 / .2 | same |
| Model / cameras | pi0 / 2 | Human300 pi0.5 / 3 |

The original pi0.5 checkpoint needs its own model architecture,state prompt,
three-camera mapping,50-action prediction horizon and native normalization.
These are compatibility requirements. They must not silently change the
executed chunk or be described as an exact pi0 benchmark reproduction.
The historical local C50/K10/noise0.3 combination was a local recipe choice.
Replay parity verifies consistent likelihoods and values,without establishing
that this recipe reproduces a published upstream success curve.

Before launch: resolve inherited upstream sampler defaults,bind a compatible
runtime and the exact checkpoint,document every necessary model adaptation,
and use a fresh output identity. Qualify the unchanged SFT through this exact
path,real C5 likelihood/value replay and separate400/750 episode boundaries.
Measure batch capacity and update time on allocated Linux. Size a bounded
learning run and fresh holdout evaluation from those measurements before
reserving compute. Preserve the objective and upstream scalar settings.

The existing local Flow PPO wrapper requires equal TRAIN/eval horizons and
uses its shared horizon in reset receipts,episode completion and terminal
observation archives. Copying the upstream400/750 settings into that wrapper
would currently fail validation. Any support change must preserve existing
same-horizon behavior and be numerically checked on Linux before training.
This change is implemented at1502237e: phase-specific receipts,episode
completion and terminal observations use the actual phase length. All68
Linux CPU tests passed,including400/750 boundaries,likelihood/value checks,
history/residual behavior and native simulator clocks. No GPU training ran.

## Direct pi0.5 RoboCasa reference and the later model question

The relevant direct pi0.5 RoboCasa PPO source is
[StructRL's sparse-PPO launcher](https://github.com/amazon-science/StructRL/blob/6caba1f9538167e0752a7338663e45aac8df965d/launch/train_pi05_sparse_rl_ppo.sbatch),
not just the pi0 OpenDrawer example above. It uses32GPUs,joint Composite-Seen
training,a target-composite SFT initializer,and explicit C50/K10/VLM critic
overrides. Its paper describes C10; its own reproduction guide maps this
launcher to the pi0.5 Sparse-RL table row. Thus C5 is the original Human300
SFT evaluator setting,not a universal original PPO setting. The local
historical C50/K10 is insufficient by itself to establish recipe fidelity.

The user next asked whether a pi0.5 PPO formulation exists,whether to switch
to OpenVLA,and to prioritize theoretical cleanliness. piRL supplies the
published flow-policy formulation; StructRL supplies direct RoboCasa outcome
evidence and released code. Mathematical likelihood clarity and exact
benchmark reproduction are separate questions. See [the model-choice audit](MODEL-CHOICE.md).
No automatic new training or model migration follows from this research.
The later explicit frozen-ICRL instruction supersedes the proposed C5/C50
VLA-training comparison. Keep pi0.5; any future controller learning must
preserve that motor and distinguish offline module fitting from deployment
adaptation through history alone. The complete atomic SFT benchmark is
verified. The extra composite evaluation was intentionally closed after the
user clarified atomic-first scope; all allocations are released. No new
VLA training,cadence ablation or model migration is pending.
