# Prospective wide action head

This separate variant is prepared from reviewed main
`0e487adf2783c9c7bdc01fadd018b297f7f157b2`. The original `make_heads`, source
protocols and all12 previously declared event runtime blobs remain byte-identical
to `be247676c1217d9e046091c52cac2b7d0f2d3014`. The05–11 suffix and04 mechanism
diagnosis inherit no change from this code. No runtime integration or physical
job is added here.

The actual inherited capacity head reads the original0.05rad bound and disables
gripper corrections. The completed13-test allocated support audit shows a
positive geometric imitation floor on four already successful TRAIN replays;
an explicit2rad counterfactual removes the recorded arm floor but leaves the
zero-edit gripper floor. Those four records are a support diagnostic, not an
admitted reduced training bank and not an explanation for the unbounded04
teacher replay failure.

`WideActionResidual` directly inherits the generic `ReactiveALOHAActionResidual`,
using its current public2062 + detached decoded native base14 + normalized slot
input, width64 tanh MLP and zero output initialization. This new version declares
**2rad arm corrections and1.1 raw absolute-gripper command corrections** in its
own CONFIG/contract. The c7a6 native interface supplies unchanged motor/norm/
reviewed URDF limits; it does not declare these new bounds.

All finite raw native14 commands must be preserved bitwise at zero output,
including signed zeros and gripper values outside[0,1]. A tiny exact-zero add
retains the ordinary delta gradient at zero, so this guarantee also holds under
autograd rather than suppressing initialization gradients. Edits clip only to
the12 actual arm URDF headroom. An out-of-range native arm is rejected rather
than silently repaired. Command dtypes that cannot exactly represent the source
arm limits are rejected, preventing outward rounding during native validation
or final clipped output. Approved +/-10 limits remain unchanged. Grippers receive editable deltas without a policy-level
absolute clamp; the unchanged native actuator later applies its existing clip.
Native motor/norm/executor/H32/TOPP/compression/gripper actuator code is unchanged.
The generic head checks all finite/shape/device inputs and detaches public/base
tensors; only new head parameters receive gradients.

CONFIG contains the versioned full constructor/configuration and source hashes.
PROFILE binds its file SHA and expected canonical head-contract SHA. The class
can reconstruct only this exact version from configuration and refuses legacy
or changed protected bound/headroom buffers on checkpoint load. Standard
`state_dict`/Torch checkpoint APIs are reused; no new checkpoint framework or
learner is introduced. Assignment-style checkpoint loading (`assign=True`) is
explicitly refused before mutation; a lossy comparison cannot install distinct
higher-precision protected buffers. Ordinary copying round trips and legacy
`strict=False` bound rejection remain covered. The caller still verifies actual motor/norm/URDF file
hashes and binds the configuration/contract/policy hashes before any runtime.
The1.1 bound is represented in the inherited float32 bound buffer; declared
physical units and requested-versus-applied clipping semantics stay explicit.

## Expectation and gates

The geometric expectation is that arm2rad/gripper1.1 should remove the captured
coordinatewise target-support floor **if actual captured base/target deltas fit
those intervals and the targets fit actual arm URDF headroom**. The bounded
readonly extension must measure this. Zero floor would establish only individual
coordinate representability; shared parameters, numerical rounding, optimization
and physical success can still fail. A residual floor should be investigated
from retained per-channel gaps and bounds, not repaired by silently enlarging
this configuration.

Fit remains gated on the original fixed12 teacher bank and a predeclared real
learning-curve profile. No reduced/easier bank or simple test fixture can replace
those gates. CPU synthetic tests qualify software only. PROFILE records all
actual qualification/admission fields false, and allocation/fit/output identities
null. No new GPU work, model inference or physical rollout is authorized by this
code-only preparation.

## Existing-owner qualification handoff

Run the focused CPU suite on allocated Linux only:

```sh
python -m pytest -q tests/test_robotwin_wide_action.py --junitxml=<fresh-owned-receipt>/wide-action.junit.xml
```

The parent may combine this gate with its existing bounded readonly support
extension in one actual residual CPU allocation. Reuse the established audit
and captured records; no separate simulator/model/physical job, allocator,
monitor or new analysis framework. The parent and finish_assembly retain all
allocation, deployment, qualification, resource admission and runtime ownership.
This worktree performs editing/AST/committed-byte hashing only on macOS.
