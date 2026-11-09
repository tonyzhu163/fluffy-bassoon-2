# Upstream settings before PPO cadence ablation

The user requested PPO C5 versus C50, then asked why the original settings
were not used. Establish the upstream baseline first. No training allocation
has been launched for this run. The separate native SFT reproduction continues
under its own immutable manifest; existing held RL runs remain untouched.

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
