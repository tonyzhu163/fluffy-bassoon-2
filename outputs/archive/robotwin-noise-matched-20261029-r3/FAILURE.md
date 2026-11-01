# Asset-complete runtime failure

Baige `job-okladnmlfk8l`, source `5f2d1c4fc7c1a5e93010aa530d9ab82234edd4fa`,
failed with exit139. Scheduler duration **2m49s**, terminal22:37:22 Shanghai.
The allocation probe and **37 focused Linux tests** passed (5.14 seconds).
The complete pinned object archive passed SHA/CRC verification; all12task
asset checks passed. Frozen motor loaded, real microphone reset succeeded,
and native/cached explicit-noise parity passed with **max error0.0**.

The process then segfaulted after saving initial learner/coordinator recovery,
as source collection began. No episode or update record exists. Only two
scene receipts exist, from initial setup/reset; no next task scene was created.
The upstream vector wrapper creates scenes during setup on the calling thread,
resets/steps via ThreadPoolExecutor, and closes on the calling thread. The
observed crash is consistent with destroying a scene on a different thread
from its creation. This is a diagnosis to validate, not a proven native stack
trace; r3 lacked Python faulthandler output. OIDN also logged unsupported CUDA
denoiser errors, despite successful rendered observations and sampler parity.

r4 keeps all single-environment simulator operations on one thread using a
Future-compatible inline executor installed before first reset. It adds
faulthandler output and a source-trial-start receipt. No action, reward, noise,
seed order, learner, or task-distribution setting changes. Initial r3 recoveries
are preserved, but contain no collected experience. r4 starts fresh with the
same learner seeds rather than migrating a changed runtime contract.

Remaining bounded retry:25minutes. r1 reserve2m + r2duration14m49s +
r3duration2m49s + r4cap25m = **44m38s**, within the45GPU-minute envelope.
