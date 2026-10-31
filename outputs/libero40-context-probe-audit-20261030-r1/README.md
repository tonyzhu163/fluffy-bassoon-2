# Offline context input-sensitivity probe

Baige `job-7hpkjw790zxm`, run `libero40-context-probe-20261030-r1`, source
`3ab93f96db79524ac6485326a0563c8401c3b2bb`, exited0. Six allocated Linux
checks passed; actual single-GPU probe passed. Model computation1.35s, zero
simulator actions, zero optimizer updates, fixed policy parameters.

The hash-bound AD500 full policy replays the actual failed support of target
group008, feeding saved executed actions and public features. Genuine support
and first query proposals reproduce exactly (max error0). Alternate inputs
consume the same actual observations/actions with query Gaussian noise paired
within2.39e-7. Only the support rewards or episode memory reset differ.

| Input change | First-query mean L2 change | Clipped-noise L2 change |
| --- | ---: | ---: |
| Erase support memory |1.28944|0.95474|
| Mask all support rewards to0 |0.06592|0.04497|
| Flip terminal support reward−1→0 |0.06569|0.04621|

This establishes input sensitivity on one fixed recorded history. It does not
execute counterfactual physical queries or establish that reward use causes
success, improves adaptation, or changes later outcomes. Reward0 denotes
success in this controller's feedback contract; masking to0 is an input
intervention, not a separately trained reward-blind policy.

[Result](index.json), [Linux checks](focused-tests.log), [GPU probe](gpu-probe.json).
Complete context/generator snapshots stay on Baige; independently rehashed
SHA256 `c243b93aa9ecfec3cfbf91abad95da3ad71f93de5a5094e0d0c916129f4fdcb0`.
