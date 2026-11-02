# Repair target identification without dropping the wrong-drawer rule

The completed clause ablation recovers128 whenever the original target block
is removed, but also wrongly accepts112. Keep the opening-depth sentence and
holding permission fixed. Test two narrower target-description changes on
full causal prefixes, using the same6 videos/35 endpoints:

- drop_stack: delete only 'This is the highest drawer in that stack, above
  every other drawer.' from verbatim1.6. All other bytes remain unchanged.
- compact_target: preserve the native left/right instruction and describe
  'the drawer immediately below the countertop on that side'; explicitly
  reject a different drawer opening while this target stays closed.

The first arm isolates the stack-presupposition sentence. The second changes
the complete target block and is interpreted only as that composite change.
128 appears to have a single drawer above cabinet doors; this motivates the
hypothesis but does not establish Flora's internal reason for its old miss.
These selected targets pass the prior native height/geometry audit; this
wording is not a universal replacement for the native height filter.

70 new candidate grades plus4 unchanged1.6 repeats give74 new outputs and105
paired candidate/reference labels. Recover35 exact1.6 labels from the completed
clause ablation, preserving their provenance. Keep official messages, views,
sampler, model/parser/temperature and full history fixed. Audit all74 actual
renderer messages; no state, success labels or native clocks enter grading.
No full634 replay, new trajectories, robot actions, motor calls or SGD.

Use the same prospective two-score>=4 rule: probes exactly50 controls apart,
accept at the second and latch;0 false first acceptances and5/5 successful
cases accepted after native success and within100 controls, all4 anchors
matching. Also report raw Score5 and two-Score5 sensitivity. Never choose a
new threshold from this result. Even a pass is development evidence only:
the reused panel is sparse before success except112. A new small validation
panel and denser early probes are required before production qualification.

Deploy reviewed latest main through Git into a fresh immutable checkout.
Linux CPU qualification checks frozen prior artifacts, six source video
hashes, unchanged production worker, exact74 messages and actual24/50/64-frame
decoding at50/100/750 controls. One Baige NVIDIA RPBZZZ6,32CPU/128GiB RAM/16GiB
SHM,15-minute maximum, auto retry off; actual GPU probe and900GPU-s limit.
Keep finite780/810/860-second deadlines and preserve any failed partial run.
Prior prefix/chunk/clause worker cost3900GPU-s remains separate.
