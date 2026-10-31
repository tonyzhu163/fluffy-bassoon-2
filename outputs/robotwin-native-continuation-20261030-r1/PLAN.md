# Source continuation supervisor: prepared, not launched

Execution/recovery candidate reviewed:`e825821a7c2231952bcf1042162f3470bd8ecdea`.
The production recovery patch is90a234d; e825821 only fixes CPU test admission. Root owns qualification/deployment/submission. Source implementation stays with
its existing owner. This launch artifact adds no learner, PPO, replay, simulator
or ledger implementation and performs no work outside an actual allocation.

## Admission without changing scientific lineage

The original immutable parent remains SHA256
`1f80c08310dfa4543ec7a748ca6e2bf9aad589cd55978a402210abb0d83bdbc0`, with scientific
source pin`db0184089fa88a5f96c153f9b80029bb91b77def`. The launcher verifies the
actual fresh worktree is the new reviewed executor pin, then writes a separate
`parent-admission.json` binding original parent/DB contract, original child
ledger header, audited trained checkpoint/selected-trial manifest, method and
new actual executor. It deliberately does not call old`admit-worker` with a false
source identity, edit a parent/header, add prior receipts or obtain new action
allowances. Existing caps remain PPO1,226,400/SAC1,194,400/reset1,168,800;
source safety cap1.1M each.

Each method's initial argument is its audited selected-trial manifest path+SHA.
The manifest identifies the protected trained backup under immutable exports.
The sole owner's`load_trial_continuation` validates it, including learner,
optimizer, private/global RNG, sampler, prefix episodes, pending PPO and physical
ledger state. Only complete retained SAC episodes and explicitly supported
reset-PPO prefixes are recovered. No new initialization or extra warmup occurs.
The old qualified scalar LR receipt821995... is reused; diagnostic weights are
never transferred.

## One independently supervised allocation per existing method

Resource request:1GPU/16CPU/128GiB/shm8;13h scheduler, automatic retriesoff.
The operational supervisor window stays at most12h within that envelope, not an
overall campaign cutoff. Remaining collection was estimated18.20GPUh across
three workers before reload/journal/evaluation overhead; root reviews this cost
before actual submission. Model reloads each round add overhead but bound the
native scene lifetime far below the observed roughly1000-reset failure regime.

The allocation probe runs once inside the actual GPU job. Focused CPU tests
are qualified separately by root; no duplicate test allocation is created.
Every fresh Python driver rechecks real native-noise parity and rendering.
All scientific args match the DB lineage:100rounds×6completeK4trials, native
task horizons, seed20261030, PPO5epochs/minibatch16/LR1e-6/KLguard.1, SAC500warmup/
batch256/UTD20, H32/1024D/bound1.5, K8target evaluation, sourcevalidationK4,
evaluation iterations10/25/50/100. The same method ledger/cap/campaignID remains.

Each child gets a fresh`slices/0001/experiment`path and passes:

```text
--source-commit NEW --lineage-source-commit DB
--resume AUDITED_CHECKPOINT
--resume-trial-manifest MANIFEST --resume-trial-manifest-sha256 SHA
--max-source-trials-per-process 6 --recycle-at-round-boundary
```

Six new complete source trials normally finish one global round. A retained
initial partial round may require up to five further trials before recycling.
A child may also finish one76-episode evaluation stage before its source round;
it remains far below the former native reset lifetime. The execution runID
is verified from the fresh resolved output path, never old`experiment` IDs.

A child has a one-hour internal wall guard and65-minute external timeout.
No new child starts in the last70minutes of an allocation, leaving room for a
round/evaluation and verified handoff. A normal time-bound stop with pending
PPO data pauses the supervisor with a complete next manifest rather than
recycling inside the allocation as though it were a round boundary. The next
allocation can recover those pending data through the same explicit mechanism.

## Handoff and failure rule

Continue only from`wall_budget_stop` with a hash-valid next manifest, full
checkpoint, unchanged scientific contract/motor receipt and zero live ledger
pending/uncertain/unresolved actions. The snapshot's hash is checked against
the actual journal head; the next driver replays/validates the complete chain.
Ledger checks are read-only and take its existing exclusive lock briefly.

Exit134/139 is accepted only after the fresh output's pre-close wallstop
summary/manifest/checkpoint prove a complete round with empty pending PPO.
The source implementation writes that handoff before`env.close`; the supervisor
records the cleanup fault and validates every bound artifact before continuing.
Missing receipts, other errors, timeouts, changed checkpoints, repeated source
episodes or unexpected ledger state halt and require audit. There is no blind
retry. Completion requires normal exit,100completed rounds/600K4trials,
empty pending PPO,100PPOupdates where applicable, and every10/25/50/100 stage
complete in both summary and checkpoint. `source_budget_match` alone never
ends the campaign: the final evaluation can require a new short-lived process.

All old and new corpus roots are indexed without copying RGB, replay or model
data. Hash-bound execution-migration ancestry is followed across allocations.
`case-provenance.json` retains every reset occurrence and its task/requested/
actual seeds, phase/trial identity, initial-observation hash, raw manifest and
physical counts. Repeated qualification/paired evaluation cases retain their
occurrences; a repeated executed source episode across roots halts. This is a
provenance index, not a relabeled/new AD/DPT dataset.

## Submission command template

Deploy qualified executor code through Git in each fresh method worktree.
Transfer both`job.sh`and`supervise.py`to the same project inputs directory,
leaving the executor worktree clean. Do not precreate each runtime output.
Root supplies the already bound initial manifest path+SHA for each method:

```bash
bash /mowenjie/fluffy-bassoon-2/inputs/robotwin-native-continuation-20261030-r1/job.sh \
  e825821a7c2231952bcf1042162f3470bd8ecdea \
  robotwin-native-continuation-20261030-ppo-r1 ppo \
  /mowenjie/fluffy-bassoon-2/inputs/robotwin-wholetrial-20261030-r1/parent-manifest-r2.json \
  AUDITED_PPO_MANIFEST ABSOLUTE_MANIFEST_SHA256 780
```

Repeat with distinct fresh SAC/reset run identities and their existing method
manifests. The parent file's actual deployment path may differ; its exact SHA
must match the original above. Any candidate fix needs root qualification and
new explicitly bound executor manifests; the parent/DB lineage stays unchanged.
No allocation or source computation was launched by this preparation.
