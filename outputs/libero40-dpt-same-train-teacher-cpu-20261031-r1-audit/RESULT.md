# Same-TRAIN-teacher validation preparation

Completed exit0 on Baige, job `job-z9tjwlkovscv`, submitted 2026-10-31
05:36:05 Shanghai. Source: `850dd96f6e8ff9b96076c7da4618f175dc473688`,
after reviewing main `acca706d5e75d1da099ff7ce6fddd727dace0fba`.
One worker, zero GPUs, four CPUs, 16 GiB, two GiB shared memory,
20-minute scheduler limit. Automatic retries are disabled.

The actual allocation attested zero visible GPUs before any CPU-only mask;
its CPU and memory cgroups match the request. All 49 focused writer,
adapter and legacy tests passed in 16.49 seconds with no skips. The
OpenPI/DSRL actor process selected the CPU backend. These are execution
checks, not a robot result. The derived binding subsequently completed all
eight goals and 8202 validation decisions. The combined assembly round trip
also completed, verifying all40 TRAIN AD/DPT histories and all8 VAL AD
histories byte-for-byte. Only the VAL DPT targets changed. Source charge
remains457851. Conservative cost348CPU-worker seconds, zero GPU seconds;
the scheduler reports4min51s. Terminal clock, completed index and exit0 agree.

Completed binding SHA256:
`9da01d662b45fe3bdc584cceb1dd55732527f9babe9ff0ef0db93d1cae1a1b23`.
Derived assembly file SHA256:
`1d7a93c8c0ec64a92687003058dac372bf6c19ab7c871aa1e9d218e624b4c29e`.
Derived assembly content identity:
`05cb614a21805e2dee390fadc36a278f05704169f1c8c764942fc107f0a4bf0b`.

The job reads the immutable all40/fixed8 source snapshot, admission SHA
`26303151bc26bc6809348b7b3e3351205619efbfb21f34da4b70229a33198fe2`.
It uses each selected TRAIN actor to label the matching validation
observations in a fresh derived generation. Actual histories, AD labels,
all TRAIN data and legacy publications remain unchanged. The completed
round trip checked all eight validation goals and source charge 457851,
counted once. Original files and complete source recoveries are preserved.

This optional job adds no simulator actions, optimizer updates or VLA
inferences. It does not gate the ready source collection or fixed100
robot readout. Their existing resource request remains pending; this job
does not grant GPU time. The proposed residual-noise diagnostic was held
at static findings without a launch or reservation.

Actual target disagreement on the same recorded validation observations:

| Goal | Decisions | Mean absolute label change | RMS label change |
| --- | ---: | ---: | ---: |
| spatial00 |1018|0.604584|0.696556|
| spatial01 |1010|0.140328|0.189078|
| object00 |1030|0.602228|0.686116|
| object01 |1014|0.165801|0.216943|
| goal00 |1006|0.222580|0.289950|
| goal01 |984|0.156623|0.196597|
| long00 |1047|0.284297|0.353321|
| long01 |1093|0.378705|0.466550|

For each loaded TRAIN actor, one recorded TRAIN label was replayed on CPU.
The maximum absolute replay delta across those eight checks was
3.2782554626464844e-7, below the declared1e-5 tolerance. This is a sampled
backend check, not exhaustive CPU/GPU equality. Pure actor parameter digests
matched the saved TRAIN checkpoint and remained unchanged during inference.

The old and new label functions differ on every goal. This directly confirms
teacher disagreement in the old validation objective, beyond the earlier
checkpoint-provenance finding. It does not establish that either teacher is
better, explain the entire25-to100 NLL increase, or show improved robot
success. Preserve legacy metrics; use the explicitly bound new metric for
heldout-state imitation in the next generation. Robot performance still
requires the prepared target comparison.

[Independent terminal/file-hash verification](../libero40-dpt-same-train-teacher-audit-20261031-r1/INDEPENDENT.md)
rehashed34 files and matched all25 supplied reference hashes plus the assembly
content identity. It inspected saved evidence without rerunning models or tests.
