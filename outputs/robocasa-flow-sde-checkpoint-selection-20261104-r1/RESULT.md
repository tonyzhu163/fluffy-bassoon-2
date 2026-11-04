# Native evaluation qualification

Nov4 2026,12:02:28 Shanghai. Baige Pro6K `job-1nfvvw2b93t5` succeeded,
one GPU released; source`2affc246`, unchanged inference/environment runtime
`0988b5f9`. **Fixed batching, paired initial states and initial likelihood
replay qualify. Exact repeated trajectories are not claimed.**

Three native eight-case panels completed: initial policy, update130, initial
policy again. All14,400 robot controls are accounted, reset controls0,
unresolved intents0, SGD0. Strict model hashes/loads passed. The initial B16
saved-chain likelihood error is exactly zero. All36 evaluation forwards use
the declared eight-lane/H50/12D layout and identical sampler RNG sequences.
All initial physical/public states and all three initial camera images match.

Repeated initial-policy first success is6/8 then8/8; update130 is3/8 (end2/8).
**These are qualification cases, not a performance estimate or a selection
panel.** The initial model's first H50 actions match exactly. After50controls,
public state still matches and the first observed difference is three RGB
components in the right camera, each by one uint8 intensity level. The second
action chunk differs by up to0.005564; later trajectories diverge. This
localizes the first observed difference in these records; it does not prove
whether rendering or an unrecorded physical difference caused it, or establish
a PPO or GPU-kernel bug. No bitwise deterministic kernel setting is adopted.

Native evaluation throughput is78.35,70.63,71.02 controls/s, including fresh
environment constructors/resets per panel. The wrapper clock is304GPU-s;
conservative scheduler creation-to-success accounting is349GPU-s
(**0.096944GPU-hours**, below the0.75 qualification ceiling). Source-campaign
costs remain separately closed at373.933889GPUh. This is not a currency bill.

[Audit](QUALIFICATION-R1-AUDIT.json), [raw panel readout](QUALIFICATION-R1-RESULT.json),
[repeat comparison](QUALIFICATION-R1-REPEAT-COMPARISON.json),
[GPU probe](QUALIFICATION-R1-GPU-PROBE.json),
[likelihood parity](QUALIFICATION-R1-INITIAL-PARITY.json),
[process clock](QUALIFICATION-R1-CLOCK.json). Full initial/decision RGB,
states, decoded actions, episode rows and action journals remain on Baige under
this run's `qualification-r1/`; the audit binds the files with SHA256 hashes.

The full five-task intermediate curve and DEV-selected/fresh-TEST readout
remain **unrun**. The [prepared manifest](SWEEP-PREPARED-R1.json) binds all80
candidate model hashes, fixed case banks, DEV order and selection/test rules.
The [plan](PLAN.md) proposes at most7,040episodes/4,224,000controls, about17–20
GPU-hours with a22GPU-hour all-attempt ceiling including this qualification.
That budget requires approval under AGENTS.md; no full sweep, new PPO training
or student fitting has been submitted.
