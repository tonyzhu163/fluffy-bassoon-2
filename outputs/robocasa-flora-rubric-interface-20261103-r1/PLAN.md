# Bound the official per-step rubric interface before scoring

Owner: chat01a1a006. The current-state2×2 diagnostic rejected all candidates;
the authorized goal remains fixing reward recognition before reward mapping.

This is a **CPU input qualification**, not a GPU launch or successful fix.
Use the same pinned model/processor sources, on allocated Baige Linux CPUs,
without loading weights or initializing CUDA. Keep production and old runs
unchanged. A failure preserves this identity; diagnose before a successor.

The official model offers rubric prompts with camera descriptions, per-step
point values, thinking, and an `ANSWER:` array. Its CLI defaults are TP2,
98,304 context,9 sampled answers and20,480 output tokens. Those defaults are
outside the unchanged terminal recipe. This audit uses official prompt,
template and parser functions with explicitly bounded **custom causal arrays**;
it does not claim to reproduce the rubric CLI's video reader or sampling.

Freeze four diagnostic endpoints before this interface is scored:112@350
(wrong drawer),128@450 (true opening near image edge),216@300 (true opening),
376@750 (insufficient opening). They are reused development cases, not an
unused validation split. Each receives the same two binary half-point steps:
target identity and current native target opening. Success requires both
steps equal0.5; total points and malformed arrays cannot substitute for that
conjunction. Simulator truth is for a later evaluator only.

Cross64-frame official causal collage evidence with32 causal frames for each
separate camera, in original left/wrist/right order. Both include reset and
the queried endpoint and exclude all later frames. This changes spatial
representation, frame count and timestamps together; any effect cannot be
attributed to just one factor. Preserve all source video hashes, frame
indices, RGB signatures, rendered prompts, processor tensors and token counts.
Require prompt tokens plus2,048 output tokens to fit16,384 for all8 inputs.
The CPU audit also checks official parser behavior. Parsing must later enforce
allowed values strictly; the official parser's clamping alone is insufficient.

Before any subsequent inference, review the latest main, commit the actual
worker and freeze its12-output contract (8 rubric probes plus4 old reference
repeats), source and input audits. A bounded single-GPU trial may use at most
1,200 additional GPU-seconds, retries off, greedy one-sample generation with
2,048 output tokens, both rubric steps required, and explicit malformed or
truncated-output counts. This is a proposed finite diagnostic within the
user-approved under-eight-hour task scope; there is currently no GPU job or
runtime qualification for this new interface. No PPO or simulator episodes.

A small diagnostic pass only admits a predeclared unused-case validation.
No wording search, threshold tuning, production reward promotion, or claims
that generated reasoning proves the model's internal cause follow from it.
