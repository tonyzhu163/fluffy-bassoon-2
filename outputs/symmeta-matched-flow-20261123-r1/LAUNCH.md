# Paired evaluation on Baige

Distributed Training accepted `job-sjubp8q0tg1x` on November 23 at 14:43:04 Shanghai. The allocation requests two Pro6K GPUs for at most 90 minutes (3 GPU-hours). At preparation the queue reported zero idle GPUs out of 416 and five queued jobs. Both estimator arms run afresh on 96 query resets, with the same policy noise and gain and gripper estimates. CPU preparation on Baige exited zero and produced all 96 comparison rows.

The detached evaluator revision is `08cf720efbb83231342fe22b285038065f254cda`. The input prediction SHA-256 is `c6828c9c848ccd0ce10bc89a5c91356b66c58619b08cbc07ba15d2af7608e191`. The CPU-verified panel SHA-256 is `276a0dce7685cefb4aad7898ccdb568d5e1fac46e19377bedc49b4e90b467cd2`. The shared job output is `baige-dev:/mowenjie/fluffy-bassoon-2/outputs/symmeta-matched-flow-20261123-r1/gpu01`. [Submitted job](evidence/submission.jpg) · [Yaw estimates](evidence/predictions.json)

At 14:54 Shanghai the two-GPU probe passed and both lanes had saved their first native pair. For indices 0 and 12, the two arms matched on the complete pre-action reset receipt and opening policy proposal, with NVIDIA EGL. Both used the submitted Pro6K pin.
