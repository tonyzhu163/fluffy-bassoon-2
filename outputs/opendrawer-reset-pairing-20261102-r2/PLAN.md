# Native reset pairing diagnosis, corrected launcher

Same finite24-reset/no-policy-action diagnosis as r1. The first attempt failed its GPU probe before native construction because the inline launcher omitted the established fallback for an absent CUDA_VISIBLE_DEVICES variable. Its console/27-second process clock are retained in r1. The repaired committed COMMAND.sh preserves an existing scheduler binding or selects the sole visible GPU index0, and requires distinct output/code/source environment values.

Source c652f054aa31dc3f1350ce5d77edc58b2c359466 at /mowenjie/fluffy-bassoon-2/worktrees/opendrawer-reset-pairing-20261102-r2; core runtime68f2c5aa unchanged. One GPU/16CPU/64GiB RAM/16GiB shared memory,20-minute operational limit, no automatic retries. No model weights, source PPO, teacher labels, case seeds or student stopping/selection rules change. Record all constructor controls/physics; compare raw image/full state/XML identities before any evaluation recovery.
