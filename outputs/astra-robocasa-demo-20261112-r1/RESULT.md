# Astra opened a native RoboCasa drawer

Selecting `gpt-6-astra` in the signed-in local Codex CLI and connecting the recorder over authenticated SSH completed **OpenDrawer / pretrain / seed 7** on Baige's allocated RTX 4090 development instance. The native instruction was “Open the right drawer.” The model controlled native actions from three RGB cameras and public robot state, with native rewards/success hidden throughout the episode.

- **Native success:** true, first reached at control 210 and still true at the final control. Native return 11.0; the agent independently claimed success.
- **Trajectory:** 220 controls at 20 Hz (11 simulated seconds), 15 decisions, 221 observations, 663 lossless camera images. Finished voluntarily within the 600-control budget.
- **Context:** each decision includes the pre-action subgoal, concise public decision summary, expected visible outcome and requested/executed actions. The trace includes approach, suspected contact, withdrawal/reorientation, handle alignment, grasp, pull and release. These are explicit training annotations, not hidden model reasoning.
- **Runtime:** episode wall time 221.72 seconds; complete client invocation 245.08 seconds. Conservative GPU window including setup and idle reasoning: 0.184 GPU-hours, below the initial 2 GPU-hour ceiling. The existing development instance remains running; the owned simulator has exited and no compute processes remained at the post-run check.
- **Training views:** 15 causal planner examples and 15 behavior-conditioned action-policy examples, plus a numeric action/state tape. Quality, mistake and verified subgoal-achievement labels remain unknown pending a sourced review. No VLA training was performed.

[Three-camera replay](preview/astra-open-drawer.mp4) · [Final views](preview/final-three-views.png) · [Decision contexts](decision-context.jsonl) · [Dataset archive](astra-open-drawer-data.tar.gz)

## Evidence and provenance

[Execution](EXECUTION.json), [GPU probe](gpu.json), [alignment verification](alignment.json), [dependencies](dependencies.json), [client result](client-result.json), [client usage](client-usage.json), [controller prompt](PROMPT.txt), [recorder tests](pytest.log), and [native smoke](native-smoke-r3.log).

Source: `c1cde5fa1ee328c541c7ab0fdea569a13a43be71`; native RoboCasa: `3443cdf3dd20f9ae8ed41adba442059b506891d0`. Codex CLI `0.167.0-alpha.3.2`, explicitly selected `gpt-6-astra`, medium reasoning effort. Model identity is the accepted CLI selection; no independent backend attestation was returned. The CLI reported 512,790 input tokens (488,960 cached), 4,544 output tokens and 1,051 reasoning-output tokens. Its JSONL is retained locally outside Git.

Eight Linux recorder tests passed. Native qualification initially exposed absent EGL and GL loader libraries in the development image; extracting the distribution packages into a private run directory fixed that without changing the shared environment. The third smoke completed two native controls. The actual Astra attempt succeeded on its first episode. All executed actions, consecutive control indices, simulation timestamps, causal histories and 663 PNG hashes passed export checks. No uncertain dispatches or interrupted episode were exported.

## Retained artifacts

The 35,385,299-byte archive contains the original session, private native snapshots, both training views, the numeric tape, decision contexts, alignment evidence and replay. SHA-256: `51e29c4a9f91a51e55c0ba1ed4c84eb94d3a39e82f79f585aea5f49b3733e74b`. Videos and raw data remain outside Git.

Authoritative remote run: `/mowenjie/fluffy-bassoon-2/outputs/astra-robocasa-demo-20261112-r1`. Remote archive: `/mowenjie/fluffy-bassoon-2/setups/astra-robocasa-demo-20261112-r1/astra-open-drawer-data.tar.gz`. A verified local copy and unpacked dataset are beside this report. The export manifest retains its original remote `session_root`; for the local copy, resolve image references against the adjacent `astra/` directory. Preserve that session with the export.

This single successful example establishes the live model-to-simulator/data path. It does not estimate success across seeds/tasks or establish a downstream training gain. The reusable launcher is in `benchmarks/agent_sim/run_codex_ssh.py`; its documented CLI configuration is the one used in this run, with post-refactor syntax and argument-help checks.
