# Frame-edge clause rejected on the retained failure sample

The added instruction about a cropped drawer front does not repair Flora.
Both fresh rubrics give full marks to six of ten native-negative prefixes
and falsely confirm three never-success episodes. The clause recovers one
earlier positive endpoint on64, but128 stays missed and216's later-closed
prefix is still scored1. Neither rubric passes the frozen development rule.

| Fixed metric | Fixture-depth/first-success base | Same text plus frame-edge clause |
| --- | ---: | ---: |
| False score5 on native-negative prefixes | 6/10 | 6/10 |
| Score5 after native first success | 4/8 | 5/8 |
| Correctly timed two-probe confirmations | 1/3 | 2/3 |
| Falsely confirmed never-success cases | 3/4 | 3/4 |
| Later-closed prefixes still scored5 | 1/2 | 1/2 |
| Repeated revision1.6 anchors | 4/4 match | Shared same4 repeats |

Confirmation means the second of two score5 probes exactly50 controls apart;
it must occur after native first success and within100 controls. This is a
detector with latency, not a claim that success held on every intervening
frame. The sample is reused and enriched for errors; these fractions are not
population false-positive rates or independent qualification.

## Every candidate score

Scores are listed in the order of the controls column.

| Case | Controls | Native first-success control | Base | Frame-edge |
| --- | --- | --- | --- | --- |
|112|300,350|Never|1,1|1,1|
|64|500,550,600,750|542|1,1,5,5|1,5,5,5|
|128|350,400,450|384|1,1,1|1,1,1|
|216|250,300,750|212|5,5,1|5,5,1|
|256|700,750|Never|5,5|5,5|
|336|700,750|Never|5,5|5,5|
|376|700,750|Never|5,5|5,5|

The same remaining errors concern different native predicates:376 opens a
lower drawer while its target stays far short;256 and336 open insufficiently;
128 crosses the native threshold but is missed. The target of256 reaches
50.2203% of fixture depth, below52.25%;336 reaches33.9638%. The observed target
peak of376 is7.5839%. These clocks/geometry come from the retained native
replays, not estimates made by Flora. All36 fresh candidate grades are1 or5;
there is no intermediate-grade threshold that separates these errors.

## Actual input path and evidence verification

All40 grading calls have raw RGB and rendered-message audit receipts. Actual
vLLM HF preprocessing ran on19 distinct inputs, and all19 bfloat16 feature
hashes match the CPU references after the same conversion. All live outputs
are already bfloat16. Their expansion back to float32 does not match the
unrounded CPU float32 hashes (0/19), which is a different comparison, not
evidence of a bfloat16 input mismatch. CPU reconstruction had already shown
pixel preservation on the four diagnostic inputs. This closes the raw and
processor-input check for this run; it does not inspect learned encoder/LLM
states or prove semantic recognition.

The downloaded archive contains54 manifest files. VERIFY-SCORES.py independently
checks every file hash, all40 raw-input/message audits, seven batch receipts,
the19 live processor references, four anchors, all36 candidate rows, paired
metrics and the cost receipt. See VERIFICATION.json and TRANSFER-MANIFEST.json.

Archive:199,153bytes, SHA256
`b60060a5607d319062b7ba78a67194305679fe4c36ea7def1e49016cd1a91cdb`.

## First-success readout without more model calls

FIRST-SUCCESS-READOUT.json replays the saved scores with an external latch.
Once an acceptance occurs, later closing cannot undo it. A single-score5
latch detects64 and216 within100 controls in both arms and retains both
later-closed successes. Requiring two probes loses64 in the base and retains
both in the frame-edge arm. Both detector choices still falsely accept256,
336 and376. Thus external latching can preserve a recognized success; two
probes do not remove these persistent recognition errors. This offline readout
does not promote a different acceptance rule or rescue either frozen gate.

## Run closure and next scientific boundary

Baige `job-pb53j1e4jom1`, source
`1f29377b04cab278ce07137fc417f29c18f5cccd`, completed40 grades in seven batches.
One allocated RPBZZZ6 passed its actual CUDA/UUID probe. Worker charge575
GPU-seconds, scheduler runtime590s, creation-to-success605s; successful and
released at13:45:32 Shanghai on2026-11-03. The prior7,350 workerGPU-second
prefix/chunk/clause/wording subtotal becomes7,925. That subtotal excludes
earlier terminal/video replays and is not whole-campaign usage. No policy
actions, simulator trajectories, optimizer steps or automatic retry ran.

Reject this framing hypothesis and do not tune the cutoff or continue lexical
fitting on the same panel. The new substantive diagnostic is to separate
recognition of the current native drawer state from retrospective first-success
judgment, with first-success memory maintained explicitly by the evaluator.
An initial-frame/current-frame comparison could test whether simplifying the
visual evidence helps target/depth recognition while retaining initial target
context; the earlier recent-chunk test did not retain that full context.
That would be a prespecified input ablation with matched current-state labels,
not evidence that another wording has repaired the reward. No such new GPU
comparison is launched by this result. Any candidate still needs unused
retained-case precision/recall/timing validation before PPO use.
