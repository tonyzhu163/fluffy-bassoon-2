# CTM T1 throughput qualification

Audited 2026-10-28 22:42 Asia/Shanghai. Baige job `job-6zp4ujvgjbqw`, immutable
source `91c16961a02aa739c6cda6cb3c8627145d51aeef`, completed with exit 0.
Remote evidence: `/mowenjie/fluffy-bassoon-2/outputs/metaworld-ppo-opt-baige-20261028-r1/`.
The implementation is integrated at `db507380b6961825bfb00f12b2585594fa9919fe`.
No additional job was launched by this audit.

One RPBZZZ6, 16 CPUs/64 GiB, CTM T1 width84/depth8, real ML10,
10 lanes × five 500-step episodes. Every timing processes 25,000 transitions,
one PPO epoch and five optimizer steps on paired behavior data and initial
weights. These are measured timings of one batch, not convergence estimates.

| Replay | Collection (s) | PPO (s) | Collection + PPO (s) | PPO speedup |
| --- | ---: | ---: | ---: | ---: |
| Original serial heads, full trial | 22.329 | 124.215 | 146.544 | 1.00× |
| Batched heads, full trial | 22.329 | 96.385 | 118.713 | 1.29× |
| Cached state, 100-step TBPTT | 21.983 | 3.981 | 25.964 | 31.20× |

End-to-end collection plus PPO improves 1.23× for full-trial replay and
5.64× for cached replay. Evaluation, checkpoint writes, startup and admission
are outside these phase times; the entire qualification used 345 active
allocation seconds, including both collectors, three update modes and tests.

The exact optimization batches actor/critic heads and distributions after
scanning recurrence, and avoids constructing initial states on all-false reset
rows. T1 is the MetaWorld driver's explicit default. Following qualification,
the user selected 100-step TBPTT as the training default. Explicit
`--replay-chunk-length 0` retains full-trial replay and cross-episode gradients.

The faster `--replay-chunk-length 100` preserves forward history and computes
GAE over full trials, but detaches recurrent credit at chunk boundaries and
changes minibatch membership. Subsequent windows start from detached behavior
states. It is a separate gradient estimator, not an exact optimization of
full-trial PPO. This qualification does not show that its learning is preserved.

Admission and checks:

- Allocated GPU identity/ownership probe passed. The probe labels this a
  training qualification, not a hardware-normalized timing comparison with
  external papers. The paired phase comparison uses the same allocation.
- 142 tests passed in 34.57 seconds, including CPU/CUDA output and full-history
  gradient parity for CTM/GRU/XL, categorical and continuous actions, partial
  resets, cached-state replay, explicit truncated gradients and exact next-update
  recovery through optimizer, sampler, normalizer and RNG state.
- Frozen likelihood/value replay errors were at most 9.54e-7 and 1.20e-7.
  Both collections were asserted bit-identical for all original rollout fields.
- Full-trial serial and batched losses both equal 1.6033668876; gradient norms
  differ by 4.7e-8. Truncated replay has a different gradient norm, as expected.
- All modes report five optimizer steps, 25,000 forward samples and 25,000
  active action samples. Peak allocated CUDA bytes: 501,698,048 / 474,161,664 /
  367,420,416 in table order.
- Static source comparison after integration finds no changes to the qualified
  MetaWorld/PPO/core implementations or qualification tests. Additional changes
  in the meta-RL subtree are the separate native-context pilot and evaluation prose.

Small copied receipts and an independent arithmetic/hash audit are in
`evidence/`. Preserve the remote raw receipts under their original run ID.
Next: validate sustained learning and episode-wise success using the selected
100-step protocol. These measurements alone
establish neither adaptation nor generalization.
