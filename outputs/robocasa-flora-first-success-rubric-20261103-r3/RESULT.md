# View guidance and first-success task rewrites do not pass

Baige job-qe1otzaxmeyi succeeded Nov3 03:06:36 Shanghai, source eada02cb.
All74 new outputs completed:35 prefixes under each of two candidate rubrics
and four verbatim1.6 repeats. The previous70 grades are recovered, giving140
candidate/reference grades plus4 repeats. All35 downloaded artifact hashes,
11 batch receipts,74 unique new IDs and official frame counts match. All four
repeated hard grades reproduce exactly:1/5/1/4. No new robot actions, motor
calls, simulator episodes or PPO updates.

| Rubric | False full scores /20 negative prefixes | Post-success full scores /15 | Correctly timed successful episodes /5 |
| --- | ---: | ---: | ---: |
| Cached1.1 |10|14|3|
| Cached1.6 |0|5|3|
|1.9:1.6 plus view guidance |1|8|3|
|1.10:view guidance plus sufficient opening and first-success wording |3|9|1|

Correctly timed means the first sampled5 occurs after native first success
and within100 controls. The frozen screen also requires0 sampled false5s
and all5 successful episodes recognised. Neither candidate passes.

1.9 recovers144 at control300 but gives a premature5 on72 at200 (native208),
then grades its250/300 successful prefixes1. 1.10 gives premature5s on72/144/264
at200, before their native208/222/216 crossings. Both still reject all three
sampled post-success128 prefixes as1. Both continue to reject every sampled
wrong-drawer112 prefix. The opening and temporal clauses change together in
1.10; these results do not isolate one clause's effect.

The same selected six trajectories and35 prefixes are used throughout. Only
112 has the full50-control grid; successful trajectories have four probes
each. Native clocks retain the prior scene/state replay qualification and
its missing-full-MuJoCo-state limitation. This remains development evidence,
not population precision or an online first-success qualification. No rubric
or reward-map promotion is justified.

The rendered messages expose a remaining protocol distinction: all criteria
in these arms are inside the user task text, while the shared system message
still calls the model a terminal-success rewarder and the final user text is
"Evaluate the terminal success of the task." This is observable prompt content,
not proof of why the model failed. A justified next comparison preserves the
native short task and moves the explicit rubric into grading instructions,
with a separately aligned first-success instruction. It must be tested before
being described as a repair.

Worker charge683 GPU-seconds; running691 and submission-to-terminal upper705,
all below the900 allocation cap. Prior failed520 and retry673 stay separate;
the three prefix attempts total1876 measured worker GPU-seconds. No live GPU.

Evidence: [metrics and anchors](RESULT.json), [paired raw grades](PAIRED-SAMPLE.json),
[actual prompts](scoring/PROMPTS.json), [copied hashes](COPIED-HASHES.json),
[CPU preparation](CPU-CHECKS.json), [closure](SCHEDULER-TERMINAL.json).
