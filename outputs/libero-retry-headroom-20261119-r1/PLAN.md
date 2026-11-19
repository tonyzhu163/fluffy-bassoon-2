# Can an earlier attempt select a better LIBERO retry?

Run the proposed retained-versus-cleared comparison on the eight registered
LIBERO-Pro2.0 development tasks. This pilot measures the two execution
strategies available to a decision-pretrained selector. The official test
split stays reserved. The released OpenPI JAX π0.5 motor stays frozen.

For each task, pair support resets 0 and 3 with query resets 1 and 4,
respectively. Use BASE and S20 with the same BASE instruction, episode budget,
reset IDs and noise seeds across timings. Generate one support with the
released five-control cadence. The support record contains public issued
commands, end-effector positions and native feedback at control rate. Fit
nothing on these tasks. Use the lag estimator and the medians (0 and 13)
from the pinned fitting-task demonstrations. Choose default execution if the
lag is nearer 0 and paced execution if nearer 13. If the estimator abstains,
choose paced. The reader sees only support commands and responses and remains
frozen. It does not use reward.

Write the selected mode before executing the query. Run it first. Also run
the other mode on a fresh instance of the same query reset with the same
noise seed. Verify equality of the initial simulator fingerprint, public
observations and horizon. Share the initial π0.5 action proposal across both
branches. At subsequent decisions both modes use the same seeded sequence of
policy noise draws. Default uses up to five controls per inference. Pacing
executes up to ten and then holds arm displacement at zero and maintains the
last gripper command for 14 charged controls. Both stop at native termination
or the released horizon. Each trial has one physical support and two
physical query rollouts (32, 64, and 96 total). Both query branches use the
same within-attempt observations and inference logic for their chosen mode.

Report unconditional native query success for the retained-history rule,
always pacing, always default, and the selector given the true timing
condition. Also report the row-wise union of successful branches to bound
the two-mode menu and both-success and neither-success counts. Include
effect by timing, native completion at the charged horizon, paired changes,
reader abstentions and the successes on each task. Ties and failures remain
in the denominator. Both-mode failures leave no positive training label.
The support/query result is a development mechanism check with a fixed
reader, with no learner fit or official benchmark score. A full VLA adapter
or transformer will require successful continuations and a measured advantage
over the strongest fixed mode.

Run serially on one allocated Baige 4090 under the existing project device
lock. Charge up to 1.5 GPU-hours for this pilot and keep its outcomes separate
from the prior fitting collection.
