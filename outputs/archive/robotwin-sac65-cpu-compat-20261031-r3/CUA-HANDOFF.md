# Actual complete-detail proof handoff

The execution owner must bind the fresh observed form/list/detail controls
before submitting. Install the two pure functions from
[capture_complete_proof.js](capture_complete_proof.js) in its existing CUA
runtime **before** arming the bridge. Syntax validation alone does not execute
them or establish live UI readiness.

Deploy the reviewed exact pin in a fresh detached remote code directory;
preserve both old runtime outputs and code pins. Use unused runtime/input
basename `robotwin-sac65-cpu-compat-20261031-r3`. Stage the same exact three
ownership helpers in its `owned-primitives`. Scheduler form: 0GPU / 8CPU /
64GiB / 22 minutes / autofault OFF. CPU command (exactly five arguments):

```text
bash <fresh-code>/outputs/robotwin-sac65-cpu-compat-20261031-r3/job.sh \
  <fresh-code> <reviewed-full-pin> \
  /mowenjie/fluffy-bassoon-2/outputs/robotwin-sac65-cpu-compat-20261031-r3 \
  /mowenjie/fluffy-bassoon-2/inputs/robotwin-sac65-cpu-compat-20261031-r3/actual-bootstrap.json \
  /mowenjie/fluffy-bassoon-2/inputs/robotwin-sac65-cpu-compat-20261031-r3/owned-primitives
```

Start the local bridge only after deployment/form/helper preparation:

```text
python3 -I -B -S <local-code>/outputs/robotwin-sac65-cpu-compat-20261031-r3/dispatch_bootstrap.py \
  --inbox <fresh-local-inbox> --code-commit <reviewed-full-pin> \
  --ssh-target <configured-target> --ssh-port <configured-port>
```

Verify its live readiness, nonce, exact code/run, expiry and task name
`mwj_1031_robotwin_sac65_cpu_r3_<first-seven-pin-characters>` before Submit.
Use a hard outer CUA tool timeout of at most 60 seconds; Date checks cannot cancel
an individual browser/filesystem API await. In that one bounded invocation, set
`submittedAtMs=Date.now()` immediately before
the observed Submit action and `overallDeadlineMs=submittedAtMs+55000`. Navigate
through fresh visible UI links to the actual detail tab, then **await**:

```javascript
await publishCompleteCpuProof({detailTab, fs: firstFitFs, inbox,
  ready, submittedAtMs, overallDeadlineMs});
```

Do not return when the header first says Running. The helper waits until job ID,
actual `job-...-master-0` and complete five-column/one-Worker `1/0GPU/8/64` table
all render, then captures native bytes and rechecks complete actual detail.
Header-only resource tables and missing Worker cells/text keep waiting; complete
contradictory resources or extra roles are rejected. The proposed snapshots in
`capture-proposed-cases.json` describe those outcomes and have not been executed.
It writes `.image` bytes, full AX and only then the nonce marker. It shares the
original overall deadline; no fresh wait is created after navigation. Retain
the actual CUA invocation duration and result as evidence.

The pre-armed bridge sniffs PNG/JPEG, builds the actual resource receipt from
the complete AX, binds the actual gate intent and automatically publishes the
hash-checked ticket last. Raw image bytes are never recoded. No manual ticket
publication or subsequent CUA reads may be represented as one-call success.

If fields do not render within the deadline, publish no marker and retain the
failure. Stop/cancel only the owner's identified job through actual UI controls
when possible; otherwise its funded gate still refuses unproved work. Reconcile
actual scheduler cost. No automatic resubmission or extra budget follows.
