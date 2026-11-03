# CPU qualification passed; framing scores pending

The40-grade framing comparison is prepared and its inputs verified on Baige.
No GPU allocation, model weight load, inference, simulator episode, policy
action, or optimizer step ran. Source43d9ca5d3852df3810a8e44fb4c2fad89e1a8996
was deployed as a clean detached checkout. The CPU process exited0 in426.152s;
PID1981266 is gone.

The preparation decoded19 distinct64-frame inputs from8 retained videos
(7 candidate videos plus one reference-only video), producing1,216 raw RGB
hashes and19 actual HF processor feature references. All four previous
processor-audit inputs match exactly at RGB, float32 features, and bfloat16
features. All40 official message structures and their criterion hashes verify.
Native replay clocks and the24 underlying view-video hashes match. Four cached
reference anchors are112@300=1,64@550=5,128@400=1,144@250=4.

The downloaded JSON archive is57,406bytes, SHA256
`8bc405651cd0a3d8a2df04c3cd9b53ced593823bd5f3f6d9cd5e9ee60c64ca33`.
Every manifest entry and the four earlier processor references were checked
independently using VERIFY-CPU.py. Raw CPU log remains on Linux; its path and
hash are retained in TRANSFER.json.

The proposed comparison is18 candidate prefixes,10 native-negative and8
post-first-success endpoints, scored under two criteria plus4 reference
repeats:40 grades,7 batches. Both criteria name cabinet-fixture depth and
first-success semantics. The only between-arm difference is the explicit
frame-edge evidence clause. This is a diagnostic development screen and
cannot qualify the reward or establish population precision.

Incremental GPU proposal: one96GiB RPBZZZ6,32CPU/128GiB/16GiB SHM, at most
900GPU-seconds, no automatic retry. Actual live vLLM processor outputs must
match all19 CPU bfloat16 references during scoring, in addition to40 raw RGB
and prompt audits. A mismatch stops the comparison. Queue capacity was checked
read-only; no resources are reserved. Refresh source/capacity before launch.

The cumulative resource boundary remains separate from CPU qualification.
AUTHORITY.md distinguishes the human's small-sample scope from an agent-set
per-allocation cap. No unused balance from other runs or chats is claimed.
