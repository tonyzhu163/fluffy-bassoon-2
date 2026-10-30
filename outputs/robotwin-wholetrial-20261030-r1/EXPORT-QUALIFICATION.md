# Native source export qualification preparation

**Update, 2026-10-30 17:19 Shanghai:** qualification is complete at corrected
pin `9c27e027ffed1f66a5d56c79e142e7a55faba717`, CPU job `job-2j83ht020h15`.
All 34 focused tests and the real SAC AD/DPT export passed. See
[results](../robotwin-native-export-20261030-r2/RESULTS.md). The unsubmitted-form
notes below are historical and superseded: tab 89096665 now displays the failed
r1 allocation, not a pending form. Do not resubmit that old command. The sole
exporter owner continues native trainer/deployment work.

2026-10-30 17:04 Shanghai. No qualification job has been submitted.

The sole exporter owner is preparing a correction to candidate
`9ee7fc27a31d1c8a7d2f5fa8f26f07962264bb32`. Static review of the actual
`db01840` corpus found that the full runtime comparison rejects
`actual_render_pci`: the corpus retains it, while the resumable source contract
intentionally removes this allocation identity. The correction must exclude
only this key from scientific equality and retain the raw file hash.

The reviewer also requested independent acting-weight tensor-digest validation
for AD policy changes and reconciliation of selected physical actions with
the frozen source/task counters. Causal teacher replay, physical bound 1.5,
complete-trial cursor selection and next-feature retention otherwise match the
actual saved schema. The owner has all findings and owns the corrections;
there is no duplicate converter or student implementation here.

One Baige allocation form is prepared in Chrome tab `89096665`: one worker,
zero GPUs, four CPUs, 16 GiB RAM and five minutes maximum. It is not submitted.
It still contains the copied production command, which MUST be replaced by
the reviewed saved-data qualification command before any submission. Do not
accidentally launch the copied command. The tab is marked for continuation.

After the corrected pin is reviewed, deploy it through Git to a fresh remote
worktree, verify its clean commit, and use the qualified Python environment.
Run the native exporter/PPO/SAC focused tests with CUDA visibility empty, then
the saved SAC AD/DPT smoke from the immutable prefix in
[source-prefix-manifest.json](source-prefix-manifest.json). The frozen SAC
recovery SHA is `f2fb39571cc3d0e809dfdc454f7f6df7f48e585fc972fd4c2fb8c83d36e63a42`.
Source and teacher are the same immutable source-only recovery; labels are
candidate deterministic means, not a teacher-competence claim. Bind the CLI to
the actual allocated hostname and clean exporter commit.

This check requires zero robot actions, zero VLA queries and zero SGD. All
tests and teacher computation must occur inside that allocation, never on
development SSH or macOS. Preserve test/export receipts and exact source,
teacher and input hashes. Native student training/deployment belongs to the
exporter owner and remains separate pending work.
