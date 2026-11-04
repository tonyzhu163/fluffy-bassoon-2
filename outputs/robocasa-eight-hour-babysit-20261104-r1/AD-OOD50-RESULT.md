# Audited converged AD: contracted OOD readout

This uses exactly the same validation-selected update13800 candidate (SHA989e87fe) as the audited ID panel, frozen before OOD deployment. CloseFridge H900 and TurnOnMicrowave H450 were designated before this run; no OOD fitting or checkpoint selection occurred. The15-task five-source steering panel remains a later conditional stage.

Transfer is weak. Fridge full-history success is8/50 initially and13/50 after five actual support episodes, versus reset7/50 initially and10/50 after five. Microwave full-history counts are0/50 initially and1/50 after five; reset remains0/50. The matched frozen native controls score33/50 and31/50 initially on Fridge, and36/50 and41/50 after five, despite having no learned memory. Native-control fluctuations across queries are not adaptation.

These are descriptive single-fit comparisons. Direct-action learner and native controls have different training data/objectives. The small observed full-minus-reset differences do not establish an adaptation gain.

| Task / memory | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| CloseFridge / full | 8 | 12 | 10 | 11 | 11 | 13 |
| CloseFridge / episode | 7 | 8 | 8 | 8 | 8 | 10 |
| TurnOnMicrowave / full | 0 | 0 | 0 | 0 | 2 | 1 |
| TurnOnMicrowave / episode | 0 | 0 | 0 | 0 | 0 | 0 |

All entries are actual query successes out of50paired cases after the indicated number of complete preceding episodes. Paired contrasts and marginal95%bootstrap intervals are in AD-OOD50-RESULT.json. Cumulative ever-success is a separate measure.

The learned physical-action head uses frozen original VLM features and zero deployment SGD. It does not use learned KV steering through the frozen native expert.

Independent CPU audit passed1200episodes,810000physical controls,16200chunks,3600videos and2433600decoded frames, including actual commands/draws/feedback, memory continuity and initial native state matching. AD-OOD50-AUDIT.json binds the exact result and raw audit hashes. Scheduler job-8htog8yitbyq released05:16:37 Shanghai,2.69GPU-hours conservatively.
