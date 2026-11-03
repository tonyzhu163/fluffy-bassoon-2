# Four-method pipeline qualification

Implemented AD, DPT, DIT and in-context IQL for each independent frozen-steering
source. PREPARE now emits four method views with matched train/validation
streams. DPT alone relabels query actions; the other methods use executed
steering actions. FIT routes the value methods to causal Q/V fitting followed
by frozen-value weighted actor extraction. All exports use the existing actor
format and physical-attempt readout, with frozen inference weights.

On Baige allocated development CPUs, source
`44f08c9964056b8bca04bfeb6d49bb19a49f0e69` passed **60 core tests in 8.88s**
and **all eight method/interface fit-export-readout combinations**. All four
DIT/IQL interrupted value-stage recoveries reproduced uninterrupted learner
weights exactly. Each exported candidate completed the synthetic five-attempt
protocol. The end-to-end checks took51.59s; tests plus fits took64.79s.
Runtime: Python3.11.14, PyTorch2.11.0+cu128, CUDA hidden in CPU children,
13CPU quota/112GiB memory container. No GPU job or real simulator episode was
launched for this qualification.

Tests protect complete-episode returns and TD masks, scalar-reference
expectile/weight/value objectives, actual-action targets, masked padding,
causal separation of return labels, both32/1152 producer serialization ports,
matched AD/DPT/DIT/IQL corpora, logical-loss weighting under execution batches,
next-Adam/target-Q recovery and inference export compatibility. Float64 checks
qualify mathematical batch equivalence; float32 gradient/parameter tolerances
account for reduction roundoff amplified by Adam near zero gradients. The
new learners enforce the analytically zero redundant attention-key-bias
gradient before Adam. Existing AD/DPT training behavior remains unchanged.

Initial attempts are retained: cpu-r1 had a runner string-substitution syntax
error before any tests; cpu-r2 exposed four overly strict float32 Adam checks;
cpu-r3 cleared the redundant-bias issue and exposed one remaining4.90e-6
float32 parameter discrepancy. cpu-r4 passes with strict float64 references
and explicit bounded float32 checks. No failed result is hidden or reused as
successful qualification.

The eight fits use deliberately short synthetic stopping criteria to exercise
pipeline transitions and recovery. They are not converged scientific students,
robot success results or evidence that source PPO improves. Scientific fits
still require completed admitted PPO source data and their actual execution
resources. Production settings retain independent validation convergence,
minimum1000updates/stage, full complete-episode windows and context300.
IC-IQL is the decoder/IQL adaptation, without S-ICQL's world-model prompt.

Evidence: [validation](cpu-r4/VALIDATION.json), [core log](cpu-r4/pytest.log),
[eight fits](cpu-r4/fits/RESULT.json). Original archive57297bytes, SHA256
`fc83115b3b5f6d83ce2268644e77ad95e89621bbea43949e74f6880fa62e5909`,
verified after authenticated browser download. Synthetic checkpoints and raw
qualification outputs remain on Baige under the same run identity.

The final merged integration at
`414ab5d9005e65f6c8a973bc8731c21e7fc7dfcb` also passed: **98 core tests in
45.83s**, including the current collector and residual PPO tests, **eight
fit/export/readout combinations**, and **four exact interrupted value-stage
recoveries**. This includes the concurrent source-owner collector and PPO
changes merged after the first numerical qualification. The fits took 51.25s;
the test-and-fit child processes took 127.39s in the same allocated CPU runtime.
No GPU computation or real simulator episode was performed. See the final
[validation](cpu-r5/VALIDATION.json), [test log](cpu-r5/pytest.log) and
[eight fit results](cpu-r5/fits/RESULT.json). The downloaded evidence archive is
54,335 bytes with verified SHA256
`53f801a95184747b9328af311200a214c1e4b19ae1cf896cea8168854506f8ff`.
