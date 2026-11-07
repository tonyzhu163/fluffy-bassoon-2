# Literature handoff — 7 November 2026

The [primary-source review](../../docs/literature/frozen-vla-offline-icrl-review.md)
and [source ledger](OFFLINE-ICRL-REVIEW.json) extend the earlier PPO recipe
audit to the frozen-student objective comparison. This handoff changes no
active source worktree, run contract, budget or optimizer setting.

## Before the next offline comparison

1. Bind the actual extraction equation, not just the name `temperature`.
   Current `ChunkObjectiveConfig('ic_iql')` resolves temperature3 and
   `importance_weights` computes `exp(A / 3)`, capped100. The original IQL
   author MuJoCo code uses `exp(3 * A)`, capped100. Our pilot's effective
   inverse temperature is1/3. Preserve that identity for any existing fit;
   an original-beta3 arm needs a fresh contract and our temperature1/3.
   Compare weight/advantage distributions before declaring a preferred value.
2. Name current values IQL-style/DIT-style adaptations: single Q with EMA,
   independent actor/Q/V history encoders, full decoded residual density,
   shifted raw-feedback context. The original IC-IQL paper shares a value-
   trained backbone with detached actor gradients. Original IQL uses clipped
   double Q; the inspected DIT value estimators condition on return labels.
   Existing unit/reference checks qualify our arithmetic, not paper identity.
3. The policy0 case manifest has32 streams with4 distinct scene seeds each,
   eight replicas per seed. The exported order is round then replica, with
   simultaneous replicas ordered offline. Do not describe it as32 sequential
   attempts on a fixed scene. Make task-family versus fixed-scene adaptation
   explicit in the online trial contract and its controls.
4. Freeze actor architecture, source split, feature/bound profile, actor
   window stream and actor update count across objectives. Report added
   critic parameters and compute. Preserve the disclosure that DEV selected
   the shared radius margin and all student case partitions were teacherTRAIN.
5. Full-architecture runtime qualification is recorded in CPUDECODEDFIT01.json.
   It is not a trained-student simulator result. Complete frozen online
   integration and sourceDEV control-authority evaluation before any gain claim.

## Source and evaluation work that remains

* Finish repeated native PPO collect/update cycles, showing fresh rollout data
  after every teacher version. More Adam steps on one retained buffer do not
  produce chronological AD supervision. Matched Motus GRPO remains separate.
* Mixed-quality data can support DIT/IQL investigation without a converged
  teacher. Keep chronological AD conditional on multiversion histories and
  DPT conditional on good action labels at the actual query observations.
* Evaluate S=0,1,2,5,10,20 real support episodes followed by a separate query,
  using frozen motor/student/stats. Reset memory between trials. Compare
  static/reset/no-reward/order/wrong-task controls; include trained ablations
  so out-of-distribution interventions are not the sole causal evidence.
* Respect six source/six held-out task identities. Base joint SFT exposure is
  a separate axis; do not call a student-held-out task novel to the base VLA.
  Use independent case/trial units for uncertainty, with replicas clustered.
* Keep RoboCasa optimizations in the shared train/eval runtime. Qualify the
  full co-resident model/renderer/communication footprint and repeated offload
  cycle; retain global2048 and existing likelihood/loss/mask semantics while
  changing execution microbatches. Preserve headroom and full recovery.

No accelerator work was launched for this review. The existing cumulative
GPU cap and pending extension request are unchanged.
