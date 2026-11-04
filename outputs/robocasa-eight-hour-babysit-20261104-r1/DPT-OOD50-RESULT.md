# Audited converged DPT: contracted OOD readout

The same validation-selected update16200 candidate (SHAbab8e0f9) used on ID was frozen before OOD deployment. The contracted tasks are CloseFridge H900 and TurnOnMicrowave H450,50paired cases/task,6actualepisodes,full and episode-reset modes. No OOD fitting or checkpoint selection occurred.

OOD transfer is weak, with no consistent advantage from retaining previous episodes. Full-history Fridge success is10/50 initially and9/50 after five support episodes; reset is11/50 initially and9/50 after five. Microwave full-history success is4/50 initially and3/50 after five; reset is3/50 and4/50. For context, the original/fixed130 native Fridge controls are33/31 initially and36/41 after five; neither native control has learned memory.

| Task / memory | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| CloseFridge / full | 10 | 11 | 10 | 7 | 9 | 9 |
| CloseFridge / episode | 11 | 11 | 11 | 10 | 8 | 9 |
| TurnOnMicrowave / full | 4 | 3 | 3 | 2 | 3 | 3 |
| TurnOnMicrowave / episode | 3 | 4 | 5 | 3 | 3 | 4 |

Entries are actual next-query successes out of50 paired cases after the indicated number of complete preceding episodes. The JSON includes all descriptive per-query paired differences, exact McNemar p-values and marginal95%bootstrap intervals. Cumulative ever-success is separate; no multiplicity-adjusted adaptation or method-ranking claim is made.

This is a frozen-VLM-feature direct physical-action head, with zero deployment SGD. It is separate from the conditional five-source learned-KV steering experiment through the frozen action expert. The native controls and direct-action head use different training data/objectives.

Independent CPU audit passed1200episodes,810000physical controls,16200chunks,3600videos and2433600decoded frames, including every actual command/draw/feedback, memory continuity and initial native state matching. DPT-OOD50-AUDIT.json binds the result and independent raw audit hashes. Scheduler job-bwzckp4hz18i released06:03:23 Shanghai,2.7133GPU-hours conservatively.
