# Shared control and data interface for the targeted pilot

Scope: `SCOPE.json`. This records implementation contracts for the user's
explicit goal, not evidence that a steering policy or dataset has qualified.
The three control mappings use one four-dimensional coefficient action and
the exact same pinned Motus/RLinf motor. Old pipelines/defaults stay unchanged.

## Control mapping

Every mapping binds its kind, dimension, scale/units, selected cache layers,
bases and their hashes, active joint/gripper indices, temporal parameterization
and clipping rules. A coefficient has no meaning without this binding.
Noise, KV and decoded actions are distinct interventions. The first search
uses a declared restricted mapping; it does not bound the full policy class.

The control driver owns the actual motor/backend, sampler, raw corpus writer,
physical-action accounting and real frozen-motor checks. The collector calls
the selected mapping with `coefficients[4]`; it must not manipulate a private
backend cache or invent a second simulator loop. The driver exposes public
features, native reset/step, current mapping identity and complete recovery.

Keep H32/five flow steps/decoded14D ALOHA transforms, three cameras and the
source task's actual horizon. Validate rather than assume cache shapes or
physical joint units. The actor may not modify motor parameters. Resetting an
environment retains the current teacher/history state unless a new trial or
lineage is explicitly started.

## Persistent condition

The condition helper owns only a declared environment/reward parameterization
and its audit record. It cannot change the motor or expose condition IDs to
student features. Use the same initial public observation across conditions
and test scene transfer. A bounded hidden actuator-calibration bias is a
possible controlled diagnostic, not automatically an adopted condition or a
natural-benchmark improvement claim. Record the exact transformation order,
arm units, gripper exclusion and any clipping; an inverse control is only
called exact if actual execution verifies it.

If an environment-side hidden calibration changes requested commands, the
agent's public action history contains its pre-calibration request. The
post-calibration actuator command and hidden parameter are audit-only. Actual
sensor observations and observed rewards remain public. Otherwise recording
both commands as student inputs would disclose the latent without learning.

## Canonical decision record

Use existing raw corpus machinery; add compact typed decision metadata rather
than a new logger. Required associations are:

- lineage/trial/episode/chunk chronology and TRAIN/VAL role;
- current/next real observations and public frozen features;
- actual requested four-dimensional controller choice and mapping digest;
- native noise/base proposal, requested/applied correction, issued command,
  actual duration, reward, terminal/truncation and any clipping;
- private teacher version, motor identity, condition identity and RNG/recovery
  references, explicitly excluded from student inputs.

All physical actions, including settling, repeats, support, native controls,
failed or unresolved calls, count toward the single phase ledger. Complete
recoveries include SAC actor/critics/targets/temperature/optimizers/replay,
sampler and all RNGs; search includes basis, candidate and cursor state.

## Teacher and student semantics

Use the existing `NoiseSAC(observation_dim, action_dim=4)` as the coefficient
learner; this is a feature-SAC pilot, not the authors' PixelSAC reproduction.
The new profile can change warmup/batch/UTD explicitly without changing old
defaults. Save before and after real updates. Update only between complete
episodes and preserve exact episode chronology. Training clock alone does
not qualify learning; evaluate fixed source-validation cases and compare
stationary retries. Preserve non-improving lineages too.

Split whole lineages and cases before windows. AD predicts the actual actor
choice recorded before its update. DPT replaces only query labels using a
qualified final teacher for that condition and mapping; all earlier behavior
remains actual. Privileged condition information selects an oracle teacher
for label generation only. Neither private IDs nor future outcomes enter the
Transformer. Preserve source-only normalization/selection and source/target
costs. A compatible completed immutable prefix can be fitted while subsequent
collection continues; never mutate admitted data or relabel it silently.

The requested first student comparison uses full/static Transformers for both
AD and DPT, matching data, optimizer budget and seeds. Use a distributional
coefficient objective and the same bounded sampling convention at deployment.
All query weights stay frozen. Query support budgets0/1/2/4 mean actual support
episodes followed by a separate reset query, with complete memory/reward/retry
controls and all support charged. Actual robot outcomes, rather than imitation
loss or integrity checks, determine whether the approach works.

The earlier eight-demo supervised fit remains an optional capacity diagnosis
when search is reachable but RL fitting fails. It is not substituted for the
goal's actual RL learning histories and Transformer experiments.
