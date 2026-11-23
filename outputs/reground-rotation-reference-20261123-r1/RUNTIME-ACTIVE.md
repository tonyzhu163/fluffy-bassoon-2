# Runtime handoff

Baige shared root is `/mowenjie/fluffy-bassoon-2/outputs/reground-rotation-reference-20261123-r1`. The upstream worktree is `$root/upstream` at `16a98486f560676b9036d1a6d067b9dd51a5917a`. The new interpreter is `$root/env/bin/python`. At 14:52 Shanghai the parent install had resolved 150 packages and was downloading wheels. CPU and development-GPU qualification are pending.

All nine official checkpoint files and the six cached tokenizer files passed full byte verification. The run-local Hub cache is `$root/hf/hub`. The pins are `7152f13e34c6c115d808aea0e5d08aca74eac14b` for pi05 and `35e4f46485b4d07967e7e9935bc3786aad50687c` for PaliGemma. All 586 official LIBERO asset files passed verification at `$root/assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`. Receipts are `$root/hf-verified.jsonl` and `$root/assets-verified.json`. The BDDL and initial states come from the installed hf-libero release.

After qualification, use `source "$root/runtime/activate.sh"` in Bash. It sets `HF_HUB_OFFLINE`, the pinned cache, `LIBERO_CONFIG_PATH`, and the upstream import path. The released class accepts `VLA('pi05', 'cuda', revision='7152f13e34c6c115d808aea0e5d08aca74eac14b')`. Native calls without a revision use the narrow offline metadata hook in `runtime/offline/sitecustomize.py`. Development instance `mwj-clean` has one RTX 4090 and sees the shared run.
