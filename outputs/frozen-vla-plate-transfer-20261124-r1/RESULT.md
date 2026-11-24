# Both source-trained history adapters failed on Kinova3 plate pushing

Frozen stock π₀.₅ pushed the plate to its goal on 62 of 64 independent Kinova3 placements. Both source-trained adapters failed on all 64. The mixed Panda–Sawyer adapter lost all 62 stock successes and rescued neither of the two stock failures. The registered two-sided test crossed its harm boundary at the first look. Evaluation for these fits ended after 192 native trajectories. [Native case records](eval/evidence/look-64.json) · [Independent statistics](stats/evidence/look-64.json) · [Stopping rule](stats/PLAN.md)

| Frozen policy on Kinova3 | Native successes | Success rate |
| --- | ---: | ---: |
| Original published π₀.₅ sampler | 62/64 | 96.9% |
| π₀.₅ with the Panda-only history adapter | 0/64 | 0% |
| π₀.₅ with the mixed Panda–Sawyer history adapter | 0/64 | 0% |

The mixed adapter changed success by **−96.9 percentage points** relative to stock (simultaneous 95% interval **−99.9 to −64.3 points**). Its two-sided exact paired McNemar p-value was 4.34 × 10⁻¹⁹ against a first-look threshold of 0.005. The p-value adjusted across the registered looks was **4.34 × 10⁻¹⁸**. The two adapter arms failed together on all 64 placements. Their observed difference was zero (simultaneous 95% interval −10.9 to +10.9 points). The registered source-diversity test required a mixed-adapter benefit over stock and its gate stayed closed. [Exact cells, intervals and decisions](stats/evidence/look-64.json)

## What was tested

Each adapter reads the last four issued commands with the observed gripper motion and opening, processes 52 inputs through 16 hidden units and adds a learned offset to the policy’s ten-step denoising velocity prediction. Only its 5,952 parameters were fitted. Both arms saw 192 expert windows on the same 16 source plate layouts and received 96 optimizer updates. One used Panda throughout. The other replaced half the windows on each layout with corresponding Sawyer examples. On four fresh source layouts the mixed fit reduced held-out Sawyer imitation loss relative to the Panda-only fit, from 1.0253 to 0.9746. That source-side improvement accompanied a native target score of zero. [Source collection](data/RESULT.md) · [Fit, held-out losses and frozen-weight verification](fit/RESULT.md)

The target was simulated LIBERO-Pro2.0 BASE, libero_goal task 5. We drew fresh physical placements from the released sampler and one independent noise stream per placement, then ran all three policies from the same verified state, camera images and noise. They shared the first five controls. Each policy then saw its own execution and ran to native success or the 300-control horizon. The analysis applies to this task, robot, placement distribution and pair of fixed fits. These are generated starts, separate from a score on the published fixed-layout bank. [Physical enrollment](eval/evidence/manifest.json) · [Native readback](eval/evidence/readback-64.json)

## Research decision

Retire both tested fits for deployment on this task. Stock left only two eligible rescues among the 64 placements, so this task chiefly tests whether an adapter preserves a working base policy. After the shared opening, the adapted policies also use a separately compiled sampler. The unissued zero-offset check in this first experiment measured the numerical difference on one proposal. We therefore registered a second native experiment on previously unused placements to isolate the effect of the learned output. [Sampler contract](eval/PLAN.md) · [Source-side zero-offset checks](fit/RESULT.md)

Baige recorded 4,204 allocated GPU-seconds for the four successful evaluation jobs and 2,000 for source collection and the three fit submissions combined. Total scheduler time was **1.723 GPU-hours**. The original four-GPU request stopped in queue without an allocation. The science inputs, training recovery, job IDs and immutable evaluation receipts are in the execution record. [Execution record](EXECUTION.md) · [Scheduler ledger](eval/cohort-scheduler-ledger.jpg)


## What caused the loss?

On **64 new Kinova3 placements**, the same compiled history sampler succeeded **63/64 with a zero residual and 0/64 with the learned residual**. The genuine stock policy succeeded **64/64**. Switching off the learned output recovered 63 successes and lost none, a paired gain of **98.4 percentage points** (simultaneous 95% interval **+69.3 to +100.0**, rounded). The exact two-sided p-value was **2.17 × 10⁻¹⁹**, below the separately registered first-look boundary of 0.005. All three policies had the same five-control opening. The two adapter conditions used the same executable, weights and precision after that opening. Issuing this learned residual caused the loss on this task. [Independent registration](../frozen-vla-plate-zero-branch-20261124-r1/PLAN.md) · [Native success, paired differences and uncertainty](../frozen-vla-plate-zero-branch-20261124-r1/evidence/look-64.json)

On all 64 original placements, both learned heads first changed the sixth issued control, from identical pre-action observations. The first changed control moved the tool farther in the negative y and z directions. By control 100 the median plate progress toward the goal was 210 mm for stock, approximately 0 mm for Panda-only and −5 mm for mixed. The source teacher kept the gripper closed on all 4,189 commands, so its gripper command and history-validity input were identical throughout training. On target, all 256 opening gripper-history slots were negative. Changing this input alone had a small effect on the first predicted correction. The audit traced the degradation to the learned intervention and highlighted a source–target mismatch. The contribution of each feature to the terminal failure remains unresolved. [Training contract, paired motions and feature intervention](../frozen-vla-plate-history-audit-20261124-r1/RESULT.md)

The follow-up adds 192 native trajectories and 1.157 scheduler GPU-hours. Across the two studies this brings the total to **384 trajectories on 128 independent placements and 2.880 GPU-hours**, including source collection and training. [Follow-up scheduler receipts](../frozen-vla-plate-zero-branch-20261124-r1/evidence/completed.jpg)
