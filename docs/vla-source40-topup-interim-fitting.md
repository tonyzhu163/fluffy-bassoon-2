# Source40 interim fitting at 25/100 updates

This is an explicit intermediate source-size point on the existing CPU-produced,
immutable all40/fixed8 top-up manifest. It performs no collection, snapshot capture,
reassembly, target exposure or additional simulator actions. It is not completion
of the cumulative 1M-source-action, 500-update or robot-evaluation phase.

All four AD/DPT × full/static controllers use the existing subset trainer and the
same seed0, batch8, AdamW LR3e-4, width128, layers2, heads4, context300 and window128.
Each follows one optimizer trajectory through updates25 and100. The profile caps
each controller at300 seconds and cumulative fitting at1500 seconds. The existing
25/100/500 default is unchanged. Existing source collection, label clocks and final
source-teacher costs remain attached to the manifest/assembly.

Run from the exact clean deployed commit inside the already funded exclusive
Baige GPU allocation, with scheduler walltime at most1800 seconds:

```bash
bash scripts/diagnostics/vla_source40_topup_interim_fit.sh \
  EXACT_COMMIT FRESH_RUN /absolute/immutable/assembly/topup-inputs.json MANIFEST_SHA256
```

The launcher uses the actual allocation GPU probe, qualified OpenPI Python3.11
with the read-only Torch2.7.1+cu128 overlay, and only the focused interim profile
and clock tests. It consumes the supplied manifest through the existing strict
`milestone_command` admission. The CPU assembly gate is separate and must have
passed before this launch; the launcher does not repeat that gate or alter its
outputs.

`worker-time.json` reserves1740 seconds plus60 seconds scheduler allowance, inside
a fixed1800-second total including setup, failures and retries. Exact retries may
append `--resume`; they receive only the remaining original time. Unclosed attempts
remain fully charged. The same source, manifest, profile and output identity are
required, and the trainer's exact resume contract prevents silently extending this
point to500 or changing its dataset. A changed scientific profile needs a new
explicitly funded run.

The existing trainer keeps complete optimizer, sampler and Python/NumPy/Torch/CUDA
RNG recoveries, with immutable policy/full-recovery archives at both milestones.
`interim-result.json` appears only after all eight policy/recovery exports and the
two four-controller matrices pass checksum and exact-update checks. Partial work
retains its actual progress/recoveries; it is never labeled completed. Inspect
`experiment/scales/update-000100/index.json` for the fixed100-update candidate set.
The result explicitly records `full_phase_complete=false`; loss/exports alone make
no robot-performance or adaptation claim.
