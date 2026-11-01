# Allocated CPU software gate

Baige `job-q6dhvcewwwx1` succeeded: **67 RoboTwin continuation/paired-launcher tests and 43 LIBERO scalar-cache CPU fake tests passed**, with zero failures, errors or skipped cases. The three real owned-process launch/termination regressions passed for TERM, INT and durable callback failure, including normal exec-child signal masks and unrelated-session survival.

Actual allocation receipt independently confirms Linux, hostname matching the scheduler job, zero visible GPUs and uninitialized CUDA, cgroup 4 CPU cores and 16 GiB RAM. No real robot actions, VLA queries or source process controls occurred. Scheduler running time was 131 CPU-allocation seconds, from 02:18:33 to 02:20:44 Shanghai; GPU cost is zero.

Exact clean source pins before and after: RoboTwin continuation `0c16c2b794f8de5b78603206917e72195083c265`; LIBERO CPU fakes `26b55eae2a5ce6545cdc97b3a875cc6e3a7b46fa`; gate launcher `5bbddf7eaff3d8ff58c50e5fb359c18fbbb5f1d8`. Each phase used its own Python process and source path. Raw pytest logs are retained verbatim. Independent JUnit readback matches both saved hashes and all 110 passing cases.

This qualifies the selected software behavior. Actual continuous production still requires current trained checkpoint/ownership admission, real allocated CUDA probe and native frozen-motor parity. LIBERO actual cached-motor GPU performance remains unqualified by these CPU fakes.

Receipts: [summary](receipts/summary.json), [independent readback](receipts/independent-junit-readback.json), [actual CPU allocation](receipts/actual-cpu-allocation.json), [scheduler terminal](receipts/scheduler-terminal.json).
