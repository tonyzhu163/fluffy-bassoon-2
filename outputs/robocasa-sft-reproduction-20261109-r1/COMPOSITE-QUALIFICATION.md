# Conditional composite throughput qualification

Prepared before campaign01 finishes; no launch until its complete atomic gate
passes (or a completed 1.0.0 comparison passes). One fresh exclusive GPU,
21 CPU/224 GiB/16 GiB SHM,30-minute scheduler maximum,retryOFF. The original
batch-one JAX server serves four simulator clients with one complete episode
each: SearingMeat4350,ArrangeBreadBasket4350,WaffleReheat4050 and
PackIdenticalLunches3900. Maximum16,650 controls,zero training.

These are the four longest tasks in the50-task manifest. They qualify runtime
capacity and collection time; these diagnostic episodes are excluded from
benchmark aggregates. Version argument chooses the exact qualified29f7ce8
1.0.1 source or isolatedv1.0 tag,keeping equivalent effective horizons.

First-observation NPZ/action digests and server RNG are retained. Video
encoding is disabled only for this capacity probe,so timing represents the
majority of the main sweep's episodes; the main evaluator default remains
selected video recording for its first two episodes per task. No observation
or policy transform changes. Simulator observation cameras still render on
the allocated GPU. Use measured controls/s,task mix,reset/startup overhead and
actual remaining budget before scheduling the1600-case composite sweep.
