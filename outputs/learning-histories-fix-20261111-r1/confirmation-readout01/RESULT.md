# Fresh-session confirmation

Three frozen fits per recipe; 8,192 sessions per split and fit. No additional updates.

| Recipe | Train k=6 | Held-out k=0 | Held-out k=6 | Same-history reference k=6 | Held-out seed range |
| --- | ---: | ---: | ---: | ---: | ---: |
| selected-corrections | 84.72% | 44.74% | 75.64% | 94.94% | 75.01–76.00% |
| original-corrections | 68.09% | 45.30% | 67.40% | 94.94% | 67.15–67.59% |
| attempts-both-lr | 64.50% | 44.56% | 60.50% | 87.39% | 56.37–62.88% |
| original-attempts | 58.08% | 45.31% | 57.34% | 86.70% | 53.94–63.38% |

Seed ranges describe three fits; they are not confidence intervals. Corrections have matched histories. Attempt histories depend on each learner’s own choices.
