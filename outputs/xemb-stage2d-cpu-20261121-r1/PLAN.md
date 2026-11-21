# Stage 2D · CPU development gate

Can the committed factored SO(3) path recover a compound rotation and expose usable factors when its visual head receives exact motion? A bias after supplying exact motion or a failure to decode the result blocks qualification at this revision.

**Parent command.** Run at the repository root on the allocated Baige Linux CPU. Use the existing OpenPI environment. The script selects the JAX CPU backend and verifies the three imported source files against commit `437da7f677632e9495c813aaa52071e207ba834e`.

```sh
run=outputs/xemb-stage2d-cpu-20261121-r1
/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python -Bu "$run/gate.py" > "$run/RESULT.json"
```

**Analytic geometry.** Use column vectors with physical translation `v = R D u`, where `R = Rz(47°) Ry(−31°) Rx(23°)` and `D = diag(0.65, 1.25, 1.55)` in issued XYZ order. The oracle issues `D⁻¹ Rᵀ v`. Wrist commands pass through. Physical jaw command is `s u₇`, with `s = −1` and positive physical command closing the jaw. The fixture uses 0.01 m of translation or jaw-width change per unit chunk command at nominal gain. These are synthetic units. The committed simulator wrapper implements yaw and a scalar gain.

- Read the actual `ACTIVE_PROBE_BURSTS` from pinned `xemb/embodiment.py`. Measure XYZ singular values and rank on all 15 controls, on prefixes of 5/10/15, and on the three burst means supplied to the reader. Report both jaw levels and the joint XYZ/jaw design rank. Jaw polarity is identified from its separate observed width change.
- Solve the full linear map from noiseless pairs, recover positive gains from its column norms, and test its inverse on 32 independently generated seven-channel commands (seed 2). Also test the supplied oracle in both directions, exact clean identity, paired nominal errors, wrist passthrough, and the wrong inverse order. Float64 max error must be below 1e−10. Nominal XYZ and jaw errors and the wrong-order XYZ error must each exceed 0.01 command units so the fixture distinguishes them.

**Pinned reader and decoder.** Exercise `apply_reader_v2` on the three burst means with synthetic local EEF and jaw changes. Record the unmodified forward contract at initialization seed 5. Then supply exact world vectors and constant confidence only at the visual head output. Check the compound, clean, and cleared cases. Run the public `predict_embodiment_v2` decoder and retain its exception or decoded shapes and consistency check.

- Compare the committed `procrustes_so3` on raw commands against both truth and the independent float64 polar factor of the cross-covariance. Repeat with gain-scaled commands as a positive control. Require maximum rotation-matrix entry error below 2e−5 for float32 agreement and report angular errors in degrees. With unequal gains and these oblique probes, the raw cross-covariance is `R D (UᵀU)`. Its polar factor can differ from `R` even under full excitation.
- Source inspection predicts a decoder failure. The pinned SO(3) branch returns three arrays, while the public decoder unpacks four planar outputs. Exit 0 means every required check passed, 1 records a failed qualification, and 2 records an execution error. Inspect individual checks in `RESULT.json` even after a nonzero exit.

**Learning and transfer.** Gain and jaw head outputs at initialization are diagnostics. Visual recovery from images and learned factor accuracy require held-out examples and fitted weights. Live frozen-policy transfer requires simulator rollouts and native rewards. This gate measures command-space error and has no policy-success endpoint.

**Source anchors at the pin.** `src/fluffy_bassoon/xemb/{embodiment,readers,equiv_v2}.py` supply the schedule, action convention, shared heads, SO(3) branch and decoder. The chunk and jaw field conventions follow `outputs/libero-xemb-ur5e-sysid-20261120-r1/collect_ur5e.py`. The numerical results are pending the parent's run.
