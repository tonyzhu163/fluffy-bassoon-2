# Frozen PPO and GRPO phone-stand baselines

Evaluate the two existing Sidney PPO and GRPO actors saved after 40 Adam updates on November 6. Start after the corrected SFT baseline finishes. These actors trained with the earlier RLinf deployment. The experiment measures those saved policies under the same deployment as original Sidney SFT. It does not estimate what re-training with the new cameras would achieve.

Apply the pinned RLinf inverse layout transform to every actor tensor that differs from the original SFT after matching its saved dtype. Preserve all other original native tensors, including Sidney's distinct input embedding and output head. Drop only the eight PPO critic tensors during actor inference. Verify both qualified export hashes and reopen each native file, encode it forward again, and compare all 667 actor tensors against the saved RLinf export. Use the same original LeRobot processors, strictly loaded model and inference code as the SFT run.

Evaluate 100 frozen clean and 100 frozen randomized scenes per actor with both wrist near planes at 1 cm. Match actual scene instructions, K10 deterministic ODE integration with the same stochastic initial Gaussian per case and chunk, H50, batch size one, 400 delivered targets and one native controller call per target. The original SFT with corrected cameras is the common comparator. Verify all paired initial images, states and prompts and the published action hashes. Report discordant wins and losses and paired exact tests. The archived K5 Flow-SDE fast-controller scores are a separate protocol.

Use up to one Pro6K GPU for 370 minutes per actor, retry off. Store generated weights only on private job-local scratch. Keep both training checkpoints and qualified frozen exports intact.
