# Can a public grasp observation select a better frozen-policy continuation?

Use the 16 frozen OpenPI π0.5 command tapes (8 development tasks, default and paced cadence) from the committed eight-task friction replay. The archive changes cadence from reset. Its whole-program outcomes cannot score a switch at grasp. Re-run the pinned BASE simulator with the recorded seeds, start 1, and ordinary or reduced pad friction. Hold the initial tape fixed in each comparison. These are paired tape interventions on one Panda, a proxy for transfer of contact handling across grippers.

At the first 12 consecutive issued close commands, expose only the completed commands, current and past public jaw positions and wrist poses. Twelve controls follow the previous regrasp study. This is a grasp-attempt trigger. Do not require a privileged contact event. With no trigger all programs follow the tape. No target calibration episodes or reward input. At this point compare three continuations:

1. Continue the same frozen tape.
2. Slow the first ten further closed-jaw commands by issuing each twice at half its six-axis motion, leaving its gripper command intact. Then resume the tape.
3. Open, retreat 10 cm and return to the pregrasp wrist pose using the earlier 30+30-control public-wrist servo. Resume at the command eight steps before closure onset. Use the actual shared-prefix states for the two waypoints.

All branches share the suite horizon. After recorded commands run out, issue zero motion and preserve the final gripper command until success or horizon. This gives the longer maneuvers time to complete. Also measure unmodified replay against the archived reward within its recorded length.

Hold out all materials and both cadences of one task per fold. The 7 other tasks provide 28 source cells with 3 paired labels each. For each triggered query find the closest triggered cell from each source task, then average the outcome vectors of the three nearest **distinct** tasks. Use Euclidean distance after source-only standardization. Ties in action value favor continuing, then slowing, then regrasping. Use no reward, task, material, cadence, clock, horizon, object pose, force, contact-pair IDs or future commands as features. The full gate adds jaw width, asymmetric closure, reopening and wrist motion over the 12-control prefix window to the mean and last translational command. A matched command-only gate removes the public response.

Score every held-out choice against its measured branch. Compare with fixed continuations, training-fold best fixed, and the per-cell hindsight maximum. Report paired wins and losses and resample whole tasks, keeping both materials and both cadences together. One start per task cannot estimate within-task reset variance. Task folds are development-only. Prior whole-tape results were known before proposing this test.

Evidence against this architecture is a tie or loss to continuing on held-out tasks, especially if the oracle also has little advantage. A routing gain needs headroom from executable continuations and held-out improvement over both fixed and command-only comparisons. Stop after one complete paired panel. No tuning on its results, new VLA calls, motor updates, GPU allocation or official evaluation tasks.
