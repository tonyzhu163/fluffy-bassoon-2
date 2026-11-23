# Runtime handoff

**Ready on Baige.** The released pi05 VLA loaded all checkpoint keys offline, predicted a finite 1 × 50 × 7 chunk from two live camera images, and advanced the 90°-shifted simulator with the first action. Independent CPU and RTX 4090 receipts are in [RUNTIME.md](RUNTIME.md).

```bash
root=/mowenjie/fluffy-bassoon-2/outputs/reground-rotation-reference-20261123-r1
source "$root/runtime/activate.sh"
cd "$root/upstream"
```

Use `$root/env/bin/python`. The pinned Hub cache is `$root/hf/hub`, the verified assets are `$root/assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`, and upstream setup wrote `$root/libero-config/config.yaml`. The model revision is `7152f13e34c6c115d808aea0e5d08aca74eac14b`. The real GPU smoke completed on `mwj-clean` and wrote `$root/dev-gpu-smoke.json`. The outer timeout wrapper returned 124 after the completed receipt. Development usage was 0.202 GPU-hours.
