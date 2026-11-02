# Complete independent DPT readouts

All300 full-model and100 static-model episodes passed the qualified independent CPU summaries, including original record/truth/snapshot/canonical hashes and exact physical/public state with the measured one-level RGB bound. Both full/static outputs use the same assembly, norm, quarter-std residual bound, candidate bindings, driver63a17336 and core68f2c5aa. Full recovery reused299 complete episodes and added only the missing ID48 query; no support or selected weights changed.

| Readout | ID OpenDrawer /50 | OOD CloseFridge /50 |
| --- | ---: | ---: |
| DPT/full K0 |29|32|
| DPT/full support |30|36|
| DPT/full K1 query |35|27|
| DPT/static K0 query |30|26|
| Initial PPO K0 |pending|28|
| Final PPO K0 |pending|30|

Full K1 versus K0: ID8 wins/2 losses/40 ties,p=.109375; OOD3/8/39,p=.2265625. Full K1 versus static: ID6/1/43,p=.125; OOD7/6/37,p=1. OOD full versus initial PPO7/8/35,p=1; versus final5/8/37,p=.5810546875. Static versus initial PPO6/8/36,p=.79052734375; versus final3/7/40,p=.34375. Exact paired tests are exploratory and unadjusted. ID PPO controls are still incomplete in the independent AD phase; no partial-case substitute is admitted.

These results do not establish a memory or population advantage. One source task/seed, different source scenes across PPO versions, only50 cases/cohort, foundation pretraining exposure and the qualified render roundoff limit interpretation. Support interactions count separately. Neither DPT nor static has an OOD net improvement over either PPO checkpoint on this bank.

Baige job-7fh78syofjo2 ended15:26:40 with299 rows because master success stopped a remaining worker. Its retained journal includes246900 controls and1 unresolved call:246750 in complete episodes and150 conservative controls in the incomplete query. Single-GPU job-cajo3v8oetx3 supplied the replacement query,750 controls and87 process GPU-seconds, terminal15:44:29. Static job-5qtsrh7a1khb ended15:40:08 with82500 controls,0 unresolved and5892 process GPU-seconds. The original full job has three known process clocks totaling12440 GPU-seconds; its killed node has no final process clock. TERMINAL-ACCOUNTING.json retains conservative scheduler lifetime bounds, explicitly including queue time. Final experiment-wide source/fit/failure/AD accounting remains outstanding.

The future driver92238b25 adds an allocation-only master completion guard; three allocated-Linux CPU lifecycle checks pass. Active source63a17336 worktrees were never edited. MEMORY-AUDIT.json in the pairing-recovery folder verifies actual selected AD/DPT checkpoints: independent actor caches, support feedback at query entry, exact fresh K0 reset, static history blindness and frozen weights/buffers on retained ID/OOD rows.
