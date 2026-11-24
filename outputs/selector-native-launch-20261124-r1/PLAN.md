# Frozen VLA selector native evaluation

Test whether choosing a motion correction or a drawer-state prompt from public observations can rescue failures of the frozen OpenPI policy on a held-out robot arm. Train the two selectors on matched Panda and UR5e branches, then compare their choices with untreated repeats, each fixed intervention and the one-choice hindsight ceiling on Kinova3. If the menu rescues no matched failures or selector gains track the untreated-repeat differences, this panel provides little evidence for deploying the selector.

Use the [registered seeds, boundaries, labels and case order](../frozen-vla-selector-20261124-r1/README.md). The source comprises 80 query episodes and the target 36, with one separate 15-command calibration per arm. The current-state visual reader must be frozen using source observations before native evaluation. Simulation state and terminal reward are audit or offline training data only. Record both bounded proposals at decisions and commands actually sent through termination.

Validate the pinned code on Baige Linux CPU, then run the registered evaluation through Baige Distributed Training on one RTX 4090 with a 120-minute limit. Store the exact SHA, device probe, support, paired case records, fitted selectors, terminal results and exit status in the shared attempt directory. Confirm an advancing native case after launch.
