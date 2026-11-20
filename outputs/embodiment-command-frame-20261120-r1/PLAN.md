# Source-only command-frame breadth comparison

Question: at equal practice cost, does a reader trained across four command mappings per arm predict the motion of two held-out mappings better than one trained on two mappings? This is the first registered screen in [the design factory](../embodiment-meta-factory-20261120-r1/SCALE.md). Neither a positive nor a null outcome is an unseen-arm task-success result.

The contracts and prior results start at `0fbc5e3167b362bc33b741e6bbfa20724aa93234`. The collector and reader implementation revisions will be pinned for execution. Four independent local CLI workers own native collection, reader fits, access qualification, and prior-art review. No physical robot or UR5e data are in scope for this screen.

Use 192 source and 64 held-out native simulator tapes of 16 commands each. Form two narrow banks and one nested broad bank, 96 tapes per fit. Use three paired optimizer seeds. Measure matching, empty, and within-arm, cross-setting swapped support on the same queries, and a linear calibration estimated from the same public practice. The registered 10% equal-cell mean-error margin in both held-out settings, same direction across fit seeds and within-family history controls determine progression. Report failures and actual counts. Stop bulk collection on reset, direction, or commanded inverse failure. Reuse the same frozen source-only response scale. No checkpoint or target setting may be chosen from UR5e.

This stage tests a training-data decision. The paper claim additionally requires native manipulation outcomes on held-out physical arm families and an oracle correction through the same interface. [Paper justification](JUSTIFICATION.md)
