# Learning-history recipe screen

Development panel only. Three seeds per recipe, 20,000 updates, batch 256; 8,192 evaluation sessions per split and seed. Fresh confirmation is pending.

| Recipe | Train k=6 | Held-out k=0 | Held-out k=6 | Reference on same histories k=6 | Seed range k=6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| original-corrections | 68.54% | 45.62% | 67.51% | 94.89% | 67.00–67.93% |
| original-attempts | 58.19% | 45.44% | 57.05% | 86.50% | 53.64–61.82% |
| both | 85.18% | 45.38% | 75.76% | 94.89% | 75.12–76.64% |
| base-lr | 69.09% | 44.94% | 67.90% | 94.89% | 64.78–72.64% |
| both-lr | 79.92% | 45.53% | 74.18% | 94.89% | 72.68–75.95% |
| wide | 79.05% | 45.64% | 74.18% | 94.89% | 71.47–77.23% |
| attempts-both-lr | 64.34% | 45.50% | 61.15% | 87.21% | 57.21–63.55% |

Selected corrections recipe for fresh confirmation: `both`.

The full-prior enumeration reader is a reference, not the split-specific optimum. Corrections use identical histories across recipes. Attempts use each learner's own choices, so their histories and enumeration reference can differ.

Dense changes attention and the training objective together; wide changes size and learning rate. This screen cannot isolate a supervision-density or capacity effect. Seed ranges describe these three fits and are not confidence intervals. No longer-duration fit was tested.
