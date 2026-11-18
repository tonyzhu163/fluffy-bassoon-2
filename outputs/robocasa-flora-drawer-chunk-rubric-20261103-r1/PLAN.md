# Fixed explicit drawer rubric on recent causal chunks

The last system-message comparison reintroduced wrong-drawer failures and
did not reliably recover128. Its CPU-only confidence/confirmation checks still
miss128. Test the visual context presented with the explicit1.6 rubric, holding
its wording and the official grading messages fixed. This is a video-protocol
experiment with that rubric, not a new wording-only comparison.

Use the same6 retained trajectories and35 sampled endpoints. Compare the
last50 and100 controls, inclusive of each window's start and endpoint, against
35 cached full-prefix1.6 grades. At20Hz these windows span2.5 and5 seconds;
the reset-inclusive51/101 video frames have official metadata durations2.55
and5.05 seconds. Clamp starts at0. Preserve the three synchronized views,
the official duration-aware sampler, model/processor/tokenizer, score grammar,
parser, temperature0 and seed0. The sampler sees only the selected window;
its relative frame indices map to explicitly audited absolute controls.
All source videos remain unchanged. No new trajectories, simulator controls,
motor calls, training or production-code/cache changes.

Generate35 grades per window plus4 unchanged full-prefix1.6 repeats:74 new
outputs,105 candidate/reference grades and4 separate repeats. Keep the raw
Score5 false-prefix/timing readout. The predeclared primary development screen
requires all4 anchor grades to reproduce, no false first acceptance, and all5
ever-successful cases accepted after their native first success and within100
controls, using two Score5 probes exactly50 controls apart. Accept at the
second probe and latch success thereafter. This is two model judgments on
different clips, not consecutive-frame persistence. Missing or irregular
probes cannot confirm a success. Only112 has a full early50-control grid;
passing this selected sparse panel still requires denser clocks and a new
small validation panel before any production qualification.

On Linux CPU, verify frozen hashes/source identity and source-video hashes,
all74 official structured messages, and absolute sampled-frame bounds. Decode
both windows at50/100/400/750 endpoints and compare their pixels exactly with
direct reads of the mapped source frames. Keep native truth outside scoring.
On GPU, audit the actual renderer messages and every frame-read request,
retaining original worker output plus annotated window records. The production
worker source stays byte-identical; the run-local reader adapter is restored
after each batch.

Use reviewed latest main in a new immutable Git checkout. One Baige NVIDIA
RPBZZZ6 GPU,32CPU/128GiB RAM/16GiB SHM,15minutes, auto retry off. Probe the
actual allocated GPU; TP1/batch8/.78 memory, finite780/810/860-second internal
deadlines and900 GPU-second cap. Prior2546 prefix-attempt worker GPU-seconds
remain separate. No automatic retry, full634 scoring, or PPO launch.
