Verified2026-10-31: Baige CPU job `job-k404j8e0104x`, run `libero40-source-pipeline-cpu-20261031-r2`, source `8b226a69b080380992bcff12e7e2eca4148f6ec0`, terminal exit0.

- Actual pre-mask allocation:0 CUDA devices, no NVIDIA device nodes, uninitialized CUDA. Cgroups measure4 CPU quota cores and16GiB memory. The qualified binary is Python3.12 /`cpython-312`, Torch2.7.1+cu128. The earlier3.10 assumption was corrected under a fresh source/run; the failed predecessor remains preserved.
- All34 allocated metadata tests passed, including the six competence tests. JUnit has zero failures/errors/skips.
- Real capture retained all40 source training streams and the fixed8 independent validation streams, selecting one complete prefix per stream. Actual combined data:33,729 train rows and8,202 validation rows for each AD/DPT objective.
- Latest bridge prefixes are included: Spatial55 cases /1,927 rows /28,620 teacher updates; Object33 cases /1,987 rows /29,940 updates. Cumulative source cost457,851 actions is retained once, within the1M source envelope. Capture/assembly adds zero simulator actions.
- Capture took514.16 seconds; assembly113.31 seconds. The final conservative CPU-worker charge is710 seconds, GPU charge0. No model construction, forward inference, optimizer update or robot evaluation occurred.
- All25 saved-result checks passed, including actual manifest/data file hashes, all40/fixed8 coverage, cost accounting, source-only admission, latest bridge exposure, runtime attestation and JUnit.

Ready fitting input: `/mowenjie/fluffy-bassoon-2/outputs/libero40-source-pipeline-cpu-20261031-r2/assembly/topup-inputs.json`, SHA `6563d1fc76ba506cf9f44cb648c30455313a33602ab1154e30b37cc54bc54883`.

Assembly file SHA `89d28ad329b612ba52a4bcf108150a335de378a0e0b209e73b7bd5f47a347160`; admission SHA `26303151bc26bc6809348b7b3e3351205619efbfb21f34da4b70229a33198fe2`.

`receipt.json` retains exact allocation/artifact evidence; `audit_saved.py` only reads terminal metadata and hashes. This makes immutable data ready for fitting. It establishes no trained controller,500-update completion, adaptation benefit or robot-performance gain.
