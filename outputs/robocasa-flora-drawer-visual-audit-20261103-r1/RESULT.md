# Added drawer failures have limited visibility in selected frames

The allocated Baige development CPU decoded14 retained three-view collages
for4 cases in2.223s. All12 source-video hashes and751-frame/20Hz clocks match;
all5 exported artifact hashes verify. Source
`071fa411347af55a947938525c5957dd8d01de3d`. There were0 model, simulator,
policy, motor or optimizer calls. The running grade-rubric experiment was
already frozen and submitted; these sheets never entered its model inputs.

Each source collage is960×320, with left/wrist/right views horizontally
arranged. The sheets add a30-pixel header above each unmodified collage for
human inspection. No target boxes, simulator answers or generated visual
details are added. The recorded scene extents and success times still come
from the separately qualified retained-action replay, not pixel measurements.

The selected stills show a concrete observability concern:

- In256 at700/750, the fixed views mostly show the countertop and upper
  cabinetry; the relevant under-counter region is near the lower edge. The
  wrist view points away. These stills do not support a reliable visual
  measurement separating50.22% from the native52.25% depth threshold.
- In336 at700/750, the left view shows part of the pulled drawer at its lower
  edge. The wrist view looks at another cabinet face and the right view
  concentrates on the sink/counter. The drawer's complete box/depth reference
  is not clearly shown in these stills.
- In376 at700/750, the under-counter drawer front and part of a lower opening
  are near the bottom of the fixed views; the wrist view points at the wall
  and counter. This is consistent with the replay's wrong-drawer diagnosis,
  but the replay remains the authoritative target-identity evidence.
- In216, the left view is largely blocked by a nearby surface. The right
  view shows a drawer extending into the bottom of the frame at250/300 and
  a more closed state at750. The wrist view changes between cabinet faces
  and floor. The replay establishes native first success212 and later closing.

This is not proof that the target is invisible throughout each full video,
that all64 model-sampled frames lack evidence, or that visibility caused a
particular Flora score. There are only14 selected stills. A causal claim
would require inspection of the actual sampled sequences and a controlled
input change. It does show why a rubric that assumes a clearly visible full
drawer box can demand evidence these selected views do not provide.

No change to the current rubric, images, camera, score cutoff or production
reward follows from this audit. If the grade-rubric comparison still fails,
the next cheap diagnosis should inspect the actual sampled frames before
another wording search. Do not conclude that numerical wording alone can
recover precise geometry from poorly framed or occluded views.

Contact sheets: [256](drawer-000256.jpg), [336](drawer-000336.jpg),
[376](drawer-000376.jpg), [216](drawer-000216.jpg).
Evidence: [CPU receipt](RECEIPT.json), [hash verification](COPIED-HASHES.json).
