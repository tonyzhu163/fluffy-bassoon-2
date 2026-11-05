# Actor-only canonical replay before the next full qualification

This bounded probe creates no environment worker and requests no simulator
controls. It reuses qual02's immutable original-SFT native chronological batch
and pre-SGD RNG, restoring the original headless native step0 DCP/actor sidecar
before replay. No updated probe weights become a scientific initializer.

Installed Torch FSDP _flat_param.py lines2044–2110 assigns unsharded gradient
views to every original parameter during no_sync. Lines2183–2243 clears frozen
parameters' views on the synchronized sharded path. use_orig_params=True is
the actual configuration. qual02 incorrectly required None at the first
accumulation microbatch. Actual frozen values there were not recorded, so this
source finding is an explanation to test, not proof that the views were zero.

Record name/original names/requires_grad, parameter and gradient shape/numel/
dtype/device, finite/nonzero/maxabs, storage relation and optimizer membership
BEFORE assertions on all eight first-global-update microbatches. Require no
frozen optimizer members, finite exact-zero or empty intermediate views, None
at the synchronized boundary, finite policy gradients, no critic, exact
old-logprob replay and unchanged frozen hashes after one actual native update.
Save native actor/Adam/scheduler and actor RNG/scaler state, then start an
independent fresh process and require exact model/optimizer/scheduler/scaler/RNG
signatures on all four actors. This component test does not replace the full
grouped simulator and12-worker/case-ordinal recovery qualification.

Allocate4GPU/64CPU/256GiB/64GiBSHM/14min, no automatic retry. Reserve15min including
startup/cleanup:3600GPU-sec=1GPUh. About4–8min is expected for two actor-only
initializations, checkpoint I/O, one accumulated update and signature readback.
The separate full qualifier keeps a prospective3GPUh reservation and fresh
qual03 identity. Both scientific arms remain blocked.

Main PPO is closed with40updates/rank and all17final-file hashes verified.
Completed task cost is62,959scheduler GPU-sec including legacy (17.48861GPUh).
Completed conservative additional charge60,745 +probe3600 +fullqual10,800 =
75,145additionalsec<the existing23h envelope82,800. Including2561legacy gives
77,706sec=21.585h. A later pair adds82,080, giving159,786sec=44.385h<the human50h
cap. Proposed46additional-hour scientific-pair migration remains inactive.
The old41,040main reservation is replaced by actual27,532creation/release charge;
do not count both. No other campaign's budget is used.

Reserve16GiB new probe growth and40GiB future full-qualification growth, plus
286GiB other owners/floor. Main and qual02 are closed and already occupy PFS;
their future growth is zero. Keep qual02 canonical/step0 payloads while this
probe loads them. Reconcile/verify/archive eligible closed scratch before a
fresh536GiB scientific-pair gate. Do not count projected releases in advance.
