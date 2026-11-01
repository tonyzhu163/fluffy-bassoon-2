# Resume CPU retry

Test-only commit e825821 isolates the synthetic driver CUDA-admission gate from global PyTorch availability; all fixture learners remain real CPU models with CPU parameters and private generators. Production source code is unchanged from 90a234. The first zero-GPU run passed 112 existing tests, then stopped at the synthetic CUDA fixture. Its actual-data manifest gate did not run. This fresh identity reruns the whole focused suite plus all eight recovery cases and then validates the three real saved states without constructing a production model.

Latest main acca706d5e75d1da099ff7ce6fddd727dace0fba was fetched and reviewed at 22:56 Shanghai; it is unchanged from the prior compatibility review. Original DB scientific lineage is retained explicitly. No active worktree is modified.
