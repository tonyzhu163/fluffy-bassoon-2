# RL² CUDA parity qualification r2

The immutable r1 job3g0m5zpcn4tb stopped at a fused-versus-stepwise FP32 gradient tolerance, maxabsolute3.2e-5. Frozen outputs and exactCUDAcheckpoint recovery passed;17tests passed,1skipped. Preserve the failed admission. The user accepted slightly looser gradient parity.

The replacement test checks BOTH FP32 paths against a CPUfloat64 oracle initialized from exactly represented FP32 weights/inputs. It separately verifies float64 fused/serial gradients, keeps tight output checks, and logs normalized-L2 and max-absolute gradient errors. No learner/optimizer changes. Re-run the fullsuite, then one50env,10epoch,8worker real update, no evaluations. OneRPBZZZ6/16CPU/64GiB,45min hard cap. Fresh immutable output/worktree; no active r1 worktrees modified.
