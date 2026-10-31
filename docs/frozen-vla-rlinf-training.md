# Matched source-data training arms

`vla_rlinf_train_arm` prepares clean-only or clean-plus-declared-Plus training
data. Both use the same original clean validation NPZ/JSON bytes. Training and
validation trial/case identities are disjoint; paired AD/DPT contexts must be
bitwise identical. Only the exact RLinf-130 model, feature contract and noise
coordinates are accepted. Plus development/test data and Pro data are excluded.

Each arm trains AD/DPT full-history and independently trained static models
with the same seed zero, width 128, two layers, four heads, context 300, global
batch four, maximum window 64, learning rate 3e-4 and 500 optimizer updates.
The existing history trainer fits normalization on training rows only and
saves optimizer, sampler, Python/NumPy/Torch/CUDA RNG recovery state. Primary
candidates are exported from final update 500; best validation-NLL candidates
are separately labeled. NLL is not evidence of robot success or adaptation.

The `source-sac` phase trains a fresh official DSRL SAC actor/critic offline
for 5,000 updates at batch 32 on exactly the same training interactions.
Raw episode hashes and public feature/action/reward/done sequences are checked
against the student corpus before replay is built. Clean local validation is
excluded. No new robot interaction is collected. Full actor, critic, target,
temperature, optimizer, replay, sampler and RNG state are saved with original
source contracts and dataset/trace hashes. Target evaluation restores this
source checkpoint under its saved contract, then records an explicit deployment
migration; each target trial branches independently while retaining source replay.

Collections are counted once across objectives and controller copies. The
augmented arm includes additional Plus data and must report that added cost.
Five thousand SAC updates and 500 supervised updates are different optimization
budgets. This pilot has one original goal family; the RGB/state SAC actor does
not address future multi-task instruction conditioning or broad generalization.

Launcher arguments are `<commit> <fresh-run> <clean|augmented> <clean-index>
[plus-index|-] [supervised|source-sac|all]`. Schedule four supervised models on
one GPU for up to 30 minutes and source SAC separately for up to 15 minutes.
Combined `all` requires the corresponding larger allocation. A time-bound
partial run retains complete recovery and is not marked a completed matched arm.
