# Causal-prefix retry: target rejection improves, successful controls are lost

Baige job-map1jxstcj93 succeeded Nov3 01:52:12 Shanghai, source17755d7a.
All70 grades on35 prefixes from six retained videos are complete:8 validated
baseline labels recovered from the failed allocation and62 new outputs.
All31 copied evidence files match their Linux SHA256 digests. No new robot
actions, motor calls or PPO updates. The worker charged673 GPU-seconds;
submission-to-terminal upper charge708, below this allocation's900 cap.
The prior failed worker remains charged520; combined worker charge1193.

The candidate1.6 says:

> Open the very top drawer directly underneath the countertop on the {side}
> of the robot's starting position. This is the highest drawer in that stack,
> above every other drawer. Pulling out a lower drawer while the top drawer
> stays shut is failure. The target must finish open by roughly half its
> front-to-back depth. Holding the handle is allowed.

Literal Score5 is the prespecified positive. Times below are seconds at20Hz.
These are the first positives **among the sampled prefixes**, not a dense
search through every earlier prefix of each successful trajectory.

| Case | Native first success | Verbatim1.1 first sampled5 | Candidate1.6 first sampled5 |
| --- | ---: | ---: | ---: |
|64|27.1|27.5|27.5|
|72|10.4|10.0, premature|12.5|
|112|Never; wrong drawer|15.0, wrong target|None in15 probes|
|128|19.2|20.0|None in4 probes|
|144|11.1|12.5|None in4 probes|
|264|10.8|10.0, premature|15.0|

There are20 sampled pre-success/wrong-target prefixes and15 post-success
prefixes. Baseline1.1 gives10 false full scores and14 post-success full scores;
candidate1.6 gives0 and5. It rejects every wrong-drawer112 prefix and avoids
the sampled premature positives on72/264, but loses the accepted128/144
controls. Both have three episodes whose first sampled positive occurs after
native success, with different identities. This does not pass a promotion
rule requiring error correction without losing accepted controls.

Case64 is native-successful earlier and then undone. Its full score at750 is
a current-state error but does not by itself invalidate first-success stopping:
both rubrics had already accepted it at550, eight controls after native542.

A post-hoc CPU screen uses only these saved outputs. Candidate Score>=4 on
two positive probes50 controls apart detects4/5 ever-successful episodes and
has no observed premature/wrong-target trigger; it still misses128. Requiring
two Score5 probes detects only1/5. Two positive sampled prefixes do not prove
uninterrupted success on consecutive frames. This fitted screen is a candidate
for a new development panel, not independent qualification or a production map.

All six native clocks come from retained-action CPU replays matching scene
metadata and all15 saved normalized public16D state boundaries exactly. No
full MuJoCo initial state was retained. Only112 has the full50-control prefix
grid; each successful trajectory has one pre-crossing, two post-crossing and
terminal probes. The panel is selected development evidence, not population
precision, recall or an online detector qualification.

Evidence: [paired raw grades](PAIRED-SAMPLE.json), [metrics](RESULT.json),
[post-hoc screen](OFFLINE-READOUT.json), [CPU preparation](CPU-CHECKS.json),
[batch completion](scoring/COMPLETED.json), [copied hashes](COPIED-HASHES.json),
[scheduler/cost closure](SCHEDULER-TERMINAL.json).
