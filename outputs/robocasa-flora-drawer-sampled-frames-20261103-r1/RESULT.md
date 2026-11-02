# The missed success contains sampled opening evidence

The Baige development CPU reconstructed256 protocol-selected RGB collages
across4 retained causal prefixes in19.590s. All32 lossless pages were visually
inspected, and all33 exported artifact hashes verify. Source
`382ec480955e61844cba38bc512a8e6ae14f15bf`. There were0 model, simulator,
policy, motor or optimizer calls and no new GPU allocation.

The unchanged _PrefixCollage, pinned official sampler and frozen videos
select64 frames per prefix. All12 video hashes,751-frame/20Hz clocks and
recorded scoring frame counts match. Each original960×320 RGB collage is
pasted without scaling onto a1920×1400 PNG page; the header is outside its
pixels. The in-memory pasted RGB hashes match all256 source frame hashes.
These are reconstructed source frames, not saved post-processor GPU tensors;
no direct processed-tensor byte comparison is claimed. Human pages never
enter scoring.

| Case and prefix | What the sampled sequence shows | Consequence for rubric iteration |
| --- | --- | --- |
|128@450, native first success384|Samples54–63, controls386–450, include visible opening/interior of the shallow under-counter drawer in the left view. The wrist view is often a close-up or occluded.|Omission of all post-success opening frames by the source sampler cannot explain this selected miss. Target identification/type wording remains worth testing.|
|112@350, never succeeds|The upper under-counter front remains flush-looking while a lower drawer protrudes; clear examples are samples54–63, controls300–350.|There is useful evidence for an initial-frame target identity and same-front tracking rule. Movement of a lower drawer is insufficient.|
|376@750, never succeeds|In samples44–63, a lower drawer protrudes in the right view while the requested upper front stays near its closed position.|This supports tracking the requested front instead of accepting any drawer motion. Do not claim zero target movement: the replay's target peak is7.58% depth.|
|256@750, never succeeds|Opening is visible around samples17–23, near the lower edge of the fixed views. The wrist view later points at other surfaces and the box's full-depth reference is unclear.|Visible interior alone is insufficient for full success. These views do not support a reliable human measurement separating50.22% from52.25%; that is not proof of universal impossibility.|

At128@450, reference1.6 and both preceding system-rubric candidates all scored1.
The reconstruction therefore narrows the previous visibility concern: image
framing is imperfect, but missing source opening frames is not a sufficient
account of this miss. It does not identify Flora's internal cause, prove what
post-processing preserves, or establish the exact opening fraction from pixels.
Native identities, first-success times and opening extents remain grounded in
the separately qualified retained-action replays, which lack full MuJoCo states.

The next controlled wording comparison is frozen in [NEXT-RUBRICS.json](NEXT-RUBRICS.json):
start with verbatim1.6 and change only its approximate half-depth sentence to
the exact native52.25% boundary. Compare that base against one added initial-
frame identity-tracking sentence and, separately, one conditional sentence
clarifying that the target can be a shallow drawer above cabinet doors. Keep
the exclusion rule, holding permission, official messages and model fixed.
These3 candidates are untested and unqualified. No rule, cutoff or PPO reward
has been promoted by this audit.

This is a selected4-case diagnosis with one prefix per case, not a population
visibility estimate or independent validation. A successful wording screen
would still require the broader small panel and new retained cases before a
production claim. No full634-video scoring was performed.

Representative evidence figures: [128 opening](case-000128-prefix-450-page-07.png),
[112 lower-drawer error](case-000112-prefix-350-page-07.png),
[256 partial opening](case-000256-prefix-750-page-03.png),
[376 lower-drawer error](case-000376-prefix-750-page-06.png).
The full50,521,580-byte reconstruction stays outside Git under the
[retention record](RETENTION.json), with every page's hash and cluster path.
Other evidence: [receipt](RECEIPT.json), [visual review](VISUAL-REVIEW.json),
[transfer verification](COPIED-HASHES.json), [all artifact hashes](MANIFEST.json).
