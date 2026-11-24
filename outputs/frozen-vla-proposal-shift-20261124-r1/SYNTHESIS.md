# Lane synthesis, November 24

Workflow `wfr-ee7ff79a408f4313ad384aa33fea172c` finished. No π₀.₅ forward pass or π₀.₅ native reward was produced by any lane: this VM has no GPU and no Baige session. The only fresh native rewards are frozen SmolVLA on CPU. [Program plan](PLAN.md) · [Harness](RESULT.md)

| Lane | Evidence | What it found | Ready for Baige |
| --- | --- | --- | --- |
| [State](../frozen-shift-state-20261124-r1/RESULT.md) | Archived records | On all 10 archived Kinova3 frames (layouts 14, 15) the hand quaternion is on the opposite sign branch, so the packed rotation X is −3.29 to −3.09 where the source convention is positive. Every native Kinova3 run packed the raw sign. Position differs from Panda by at most 7.4 mm with no fixed offset | Four adapters |
| [Second motor](../frozen-shift-smolvla-20261124-r1/RESULT.md) | Native reward, SmolVLA, CPU | Panda 3/12, Kinova3 0/12, IIWA 2/6 on the six failed layouts. Fisher p = 0.22 Panda against Kinova3. Pro2.0 instruction wording gives Panda 0/7; the LIBERO wording gives 2/6. A second noise seed flips Panda on 3 of 6 starts | CPU harness |
| [Source guidance](../frozen-shift-guidance-20261124-r1/RESULT.md) | Offline, six source tapes | A progress-aligned donor window predicts a held-out tape's next five controls with error 1.68 against 2.87 for time alignment, paired difference 1.20 [0.91, 1.46], 6 of 6 | K=4 adapter |
| [Sampling](../frozen-shift-sampling-20261124-r1/RESULT.md) | Offline | No archive holds a second noise draw on a failed Kinova3 start, so noise headroom is unmeasured. The Panda-only critic ranks the executed window first of four on 29/114, chance 1/4, and only late in the episode | 48-episode noise panel, critic adapter |
| [Phase prompt](../frozen-shift-phase-20261124-r1/RESULT.md) | Archived frames | The reader matches the audit drawer joint on 10/10 frames. Layout 15 never opens the drawer, so a clock would switch to placement over a closed drawer and the reader would not | Five adapters |
| [Body view](../frozen-shift-view-20261124-r1/RESULT.md) | Archived frames | Panda render changes 2,301–10,263 external pixels and 0 wrist pixels and newly covers 793–4,209 scene pixels | Three adapters, render twin unexecuted |
| [Attribution](../frozen-shift-attribution-20261124-r1/RESULT.md) | Archived proposals | Panda render moves 10 archived proposals 0.0045–0.0077 closer to donor controls from about 0.18–0.21. No noise floor is archived, so this is not a shift beyond noise | Forward-only job |
| [Embodiment prefix](../frozen-shift-prefix-20261124-r1/RESULT.md) | CPU stand-in | Fitting code keeps every base tensor byte-identical on a stand-in model. 171 fit rows from two IIWA tapes. OpenPI paths never executed | Not ready, needs a GPU smoke |

## Reading

- The state lane gives the most specific lead. A sign-branch error in the same packed channel took frozen π₀ on UR5e to 0/96 until the quaternion was canonicalized ([FINDINGS](../../docs/FINDINGS.md)). Kinova3 has run with the raw sign in every native study here. Whether that explains any of the six π₀.₅ failures is unmeasured, and Kinova3 does solve layouts 16, 31, 34, 35 and 39 with the raw sign, so it cannot be the whole story.
- If the sign branch matters, the earlier Kinova3 negatives (correction, cue, timing, selector) were measured on top of a state the policy never saw in training and need a rerun on the mapped state before they are read as architecture results.
- Policy noise and instruction wording both flip native outcomes on SmolVLA with weights untouched. No Kinova3 failure has ever been rerun under a second noise draw on π₀.₅, so the 48-episode noise panel is the cheapest measurement of how much any selection over samples could gain.

## Launch order for one RTX 4090 job

```bash
bash outputs/frozen-vla-proposal-shift-20261124-r1/run.sh "$(git rev-parse HEAD)" native01 --starts development
```

1. `state-mapped`, `state-dropped`, `state-shuffled`, `state-mapped-shuffled` with raw and repeat on the 11 development starts.
2. Eight noise draws on each failed start.
3. Phase-prompt and source-guidance adapters, on the mapped state if step 1 shows it matters.
4. Body view after a one-start smoke of the render twin. Prefix after a GPU smoke of the OpenPI path.

`run.sh` and the real policy and simulator paths have only been syntax-checked; start with `--starts 14` as a smoke.
