# Additional failures separate wrong drawer, insufficient opening and later closing

All4 retained-action replays qualify on Baige's allocated development CPU.
Source `0f930abe89665daa24f2b8bf2066111e186e0a15`. All10 saved metadata fields
per episode and all60 saved normalized public16D chunk boundaries match
exactly, maximum error0. All4 terminal native outcomes reproduce.3000 existing
physical controls were replayed; no new action sampling, motor/model inference,
optimizer steps or GPU calls. Elapsed366.97s. All7 exported JSON artifact hashes
and the archive hash match the Linux receipt/output.

| Case | Target's peak normalized opening | Peak outward distance / fixture depth | Ever native success | Diagnosis |
| --- | ---: | ---: | --- | --- |
|376|0.13789|7.58%|No|Another drawer opens widely while the target remains far short|
|256|0.91310|50.22%|No|Correct target opens only about halfway, just below native threshold|
|336|0.61752|33.96%|No|Correct target opens about one third of its depth|
|216|1.82282|About100%|Yes, control212 /10.6s|Target later closes to11.12% depth; terminal failure|

Native success is normalized opening>=0.95, equivalent to outward slide>=
0.5225×fixture depth. Case256 stops at0.502203×depth,2.03 percentage points of
depth below the requirement. Therefore 'roughly half' is not a sufficient
binary-success definition: it includes a real retained simulator failure.
This establishes a wording/specification gap, not the internal reason Flora
assigned that video a5 under earlier rubrics. Case336 shows a larger extent
error, while376 is a different-drawer error. The reward repair must cover both
identity and sufficient displacement without requiring hand release or every
other drawer to remain closed.

Case216 is not a false-positive episode under stop-at-first-success semantics:
it succeeds at212 before later closing. Its complete-video score alone cannot
establish whether Flora would first trigger too early; causal prefixes are
still needed. Its original terminal failure label must not be used as a
never-success label. These4 selected examples do not estimate prevalence among
all183 terminal failures.

The next small panel can retain the existing6 trajectories/35 prefixes and
add10 prefixes here:700/750 for each of the3 never-success cases, plus200/250/
300/750 for216. This gives10 trajectories/45 prefixes, with4 never-success and
6 ever-success trajectories. The new negative cases are sampled late, not on
a complete early grid. [Public video references](PUBLIC-ADDITIONS.json) contain
no labels; [native reference sidecar](PREFIX-ADDITIONS.json) stays outside model
inputs. Actual Linux video hashes/frame clocks and new target geometry still
need qualification before scoring.

Two untested next task texts remove the countertop and conditional-failure
phrases while explicitly defining the52.25% boundary. They are recorded in
[NEXT-RUBRICS.json](NEXT-RUBRICS.json), not promoted. Full-success detection
should require complete-success output; the previously fitted Score>=4 screen
remains a diagnostic and must not silently become native success. Freeze the
new screen before inference. No production reward or threshold is changed.

No full MuJoCo initial state was saved. Replay identity is qualified by full
scene metadata and all retained state boundaries, not by restoring a complete
saved simulator state. Only the known unordered semantic-tag list is
canonicalized. Original pending checkpoints remain untouched; recovered binary
traces stay on Linux and are not committed. No process remains running.

Evidence: [per-case readout](READOUT.json), [full replay summary](physics-replay/SUMMARY.json),
[CPU receipt](CPU-RECEIPT.json), [trace provenance](SOURCE-SURVEY.json),
[transfer checks](COPIED-HASHES.json).
