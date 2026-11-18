# CPU audit of the official processor representation

The preceding goal turn made progress: the 82-grade target-hints screen
completed with verified actual RGB reads, and rejected all three candidates.
Case 128 remains score 1 after simulator success despite opening evidence in
the selected source frames. Check the processor before another rewrite.

Use the same four retained inputs as the earlier full sampled-frame audit:
112@350, 256@750, 376@750, and 128@450. They contrast a wrong lower drawer,
a near-boundary insufficient opening, another wrong drawer on the other
side, and the missed success. Decode the unchanged official 64-frame sampler
and require all 256 RGB hashes to match both the visual audit and the latest
scoring input references. No new trajectories, model inference or GPU work.

The downloaded source survey pins the actual installed official engine, HF
processor, vLLM processor and model configs by hash. Confirm that Qwen3.5
uses the Qwen3VL multimodal processor, and inspect its nested pre-sampled
video/metadata call, request size, and disabled second frame sampling.
Instantiate only the official processor on allocated Linux CPU. Do not load
model weights. Use the frozen native-boundary messages and reproduce the
previous rendered prompt hashes for all four inputs.

Run both the full HF processor path and its direct video processor, requiring
equal video features, grids, and unchanged source-frame indices. Record the
actual shape, grid, config, source/package versions, and feature hashes.
Invert the documented spatial/temporal patch permutation and normalization;
compare every reconstructed RGB byte with source. Record errors rather than
assuming that temporal patching averages frames. Separately reconstruct the
bfloat16-cast features used by the visual model's input path and measure
numerical differences. Save four original/round-trip comparison sheets for
human inspection; they never enter scoring.

The source implementation predicts a [32,20,60] video grid: 64 original
320×960 frames packed in pairs, with no spatial resizing, giving 38,400
patch rows and 9,600 merged vision tokens. This is a prediction until the
CPU audit runs. An exact round trip would rule out missing pixel information
at this reconstructed preprocessing stage for these four inputs. It would
not prove semantic retention through learned visual embeddings or language
grounding. Previous GPU processor/encoder tensors were not saved; never
claim a direct byte comparison to them.

Deploy latest main via a Git bundle into a fresh immutable detached checkout;
do not modify the completed scoring checkout. Check shared storage capacity,
write access and development-container CPU/RAM limits. Set CUDA visibility
empty, use four CPU threads and a finite 240-second process deadline. Package
the receipt, configs and images with SHA-256 hashes, download and verify them,
then consolidate the completed rubric evidence. The audit cannot itself
qualify a reward. Propose further scoring only when this diagnosis supports
a concrete bounded comparison with independent held-out validation reserved.
