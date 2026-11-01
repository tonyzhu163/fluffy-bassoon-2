# RoboTwin scheduled collection check

Read-only source metadata at06:08:54Shanghai and authenticated scheduler
status at06:10 confirm both allocations are live and their counters advanced.

| Learner | Complete source rounds | Complete K4 trials | Recorded source actions | Optimizer counters |
| --- | ---: | ---: | ---: | --- |
| PPO |96/100|576|928637|480 SGD steps|
| Trained reset |99/100|594|953771|495 SGD steps|
| SAC |88/100|528|885076|561760 per optimizer;1685280 total|

Paired checkpoint audits pass and frozen-motor records match. The closed
SAC child has no pending trials or unretained/uncertain source charge; its
wall-budget stop is a finite child boundary, not overall job completion.
Combined closed source actions:2767484, excluding active/evaluation and
historical-bank costs. Trial counts do not prove distinct physical states.
The final100-round comparisons are unfinished; no robot-quality or memory
claim follows from collection/SGD counters.

Owning chat retains all controls, budgets and recovery. No launch, model,
test, source mutation or additional resource commitment occurred. Next
scheduled check06:30Shanghai, or earlier upon meaningful completion/failure.
LIBERO source/readout readiness and the pending compute decision are unchanged.
