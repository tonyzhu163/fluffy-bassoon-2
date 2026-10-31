# Phone data stage0 — actual allocated CPU result

Baige job `job-rh3ldkdapfv9` succeeded on exact source `9e1953e6ee2ea4c91fdc494355fce2ab55c99c9d`: 11/11 focused tests pass, no errors/failures/skips. Actual pre-mask CUDA count is zero, CUDA is uninitialized, and cgroups attest 2 CPUs/16GiB. The single attempt charged 86 conservative CPU-worker-seconds, giving 417/1,800 phase CPU seconds used and 1,383 remaining; 214 seconds are released from the same 300-second reservation. GPU/action costs are zero. The canonical action snapshot is unchanged.

The exact phone archive was copied and its first12 whole HDF5 episodes inventoried. Episodes0–7/8–11 remain fixed TRAIN/VAL **candidates**, bound to actual seeds, scene metadata and each first published seen instruction. All12 explicit success flags are absent. Source code establishes contemporaneous drive-target data; the inventory does not qualify an action temporal shift or simulator replay. Finite physical14D arrays, per-file hashes and byte-accurate metadata are saved. Dataset conversion, normalization/decoder round-trip, teacher success and capacity are unqualified.

Saved A1 commands contain 17,760 issued slots and 17,591 actual native-valid slots. Gripper channels6/13 have zero nonfinite values. Valid minima are `[-0.0101426,-0.0069030]`, maxima `[1.0090805,1.0086253]`, with `[2362,5913]` values outside `[0,1]`. Issued-slot outside counts are `[2384,5913]`. These are pre-actuator decoded native commands; the existing RoboTwin actuator clipping is retained, and no command was corrected or replayed during this read.

The data-stage storage snapshot is 840,132,600 bytes including the external 255,542,392-byte archive as well as its in-phase copy; final JSON/log metadata adds a small amount. No weights, model, simulator, training, new robot actions or capacity GPU run occurred. R1/R2 remain closed and immutable. The control owner can now use the exact inventory for its separately reviewed alignment/capacity plan.

![Actual0GPU CPU job succeeded](screenshots/phone-stage0-cpu-terminal.jpg)
