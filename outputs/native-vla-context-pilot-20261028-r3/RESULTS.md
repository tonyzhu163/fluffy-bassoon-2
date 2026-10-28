# Native context interface admitted; collection stopped

Baige `job-8kkjhb65px67`, source `b5d97b0`, exit 1 after 100.84 driver
seconds. [Gate receipt](evidence/gate.json) proves exact native empty-history
parity, twelve finite fitting steps (loss 0.306293 to 0.230301), nonzero public
feedback gradients, exact stochastic replay and unchanged frozen weights.
[GPU probe](evidence/gpu-probe.json) and
[seven passing policy tests](evidence/gpu-qualification.log) are retained.

The first prepared command was on CPU. The next public feedback record
therefore arrived on CPU while the history buffer was on CUDA, triggering its
device guard. [Failure receipt](evidence/failure.json). No full history,
warm-start training or PPO update completed. Gate fitting weights had already
been restored to initialization.

The fix moves the detached previous requested command to the policy device
before forming feedback, matching the existing shared bridge collector. The
new two-attempt integration regression also checks reward delay, boundary
fields, public unrotated commands, full history length and dtype/device. It
runs once on CPU and once on the actual allocated CUDA device before loading
the native model. This run remains immutable; continuation uses r4.
