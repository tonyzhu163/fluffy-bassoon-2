# CMM structure on compact in-context RL benchmarks

Status: plan only, not approved for launch. It is gated on the CMM calibration
(thu2 job 48876) passing its final 10,000-episode test within the paper's spread.
That job qualifies the supervised recurrence only; this plan asks whether the
CMM *memory structure* does something on RL trials that a plain recurrent core
cannot.

Every claim below is a contrast inside CMM or against matched baselines, never
"CMM obtains reward" alone.

## 1. Tasks whose required history exceeds the short-term trace

The CMM short-term trace holds 10 steps by default (32 in the screen). A task
solvable inside that window never exercises long-term memory. The DPT pilot
already shows this: on the deterministic bandit the memory-wiped score equals
normal (.92188 both).

- Bernoulli bandit, K64/K128 pulls, six arms: pull counts must be integrated
  far beyond 32 steps.
- Dark Room 9x9, H20, 2/4/8 attempts (40-160 steps): the goal location must
  survive attempt boundaries.
- Key-to-Door 9x9, H50: two ordered hidden facts. Second stage.

## 2. Structural ablation ladder, parameter-matched

| Condition | Isolates |
| --- | --- |
| `--core cmm`, 32 LTM tokens | the claim |
| `--core cmm --ltm-tokens 0` (CMM-STM) | long-term store |
| `--core ctm` (zero memory blocks) | memory Transformer |
| CMM T1 vs T4, decay once per observation | internal ticks, without the horizon confound |
| GRU, Transformer-XL | external references |

The structural signature: full CMM holds its score as trial length grows past
the short-term trace, while `--ltm-tokens 0` falls off near that length. Sweep
trial length (Dark Room attempts 2/4/8; bandit 16/64/128 pulls) and report
score against length per condition. That curve is the primary result.

Parameter matching uses the screen's rule (largest width within the cap, at
least 85% of it). Ticks, wall time, FLOPs and retained state bytes are separate
reported axes, not matched.

## 3. Representation versus trainability

The pilot's Dark Room result (normal .03125, memory-wiped .76563 after 112
updates) looks like an optimization failure, not an architecture verdict. Run
each task under two objectives:

1. Supervised in-context training on histories (Algorithm Distillation on
   improving learning histories, and the existing DPT objective). Optimization
   is easy here, so this measures what the structure can hold. Anchor with
   Transformer-XL: AD solves Dark Room with a causal transformer. If XL does
   not reach the published level here, the pipeline is broken and CMM numbers
   are uninterpretable.
2. Meta-PPO (RL²-style) with the existing full-trial PPO. This measures whether
   the structure is trainable from reward alone.

## 4. Lesions and probes on frozen checkpoints

Greedy and sampled, paired evaluation seeds.

- Existing: normal, memory-wiped, reward-blind.
- CMM-specific: zero LTM tokens at each attempt boundary; truncate the
  short-term trace to the current step; reset only the synchronization
  accumulators. Whichever lesion hurts locates the information.
- Linear probes after each attempt: decode the Dark Room goal cell (or bandit
  best arm) from LTM tokens and from synchronization features, fitted on
  training trials and scored on held-out trials. Decodability should rise with
  attempts, and only in the components the lesions implicate.
- Length extrapolation: train at 4 attempts, evaluate at 8 and 16. CMM's
  fixed-size memory should degrade gracefully; Transformer-XL with a finite
  memory length should fall off. This is the property the paper claims and the
  most direct structural test.

## 5. Meta-World ML10: deferred

Not in the repository (no Meta-World dependency). It mostly tests task
identification and continuous control rather than memory, and published
held-out success is low and noisy (ECET about .35). If wanted as a comparison
with ECET, TrMRL and AMAGO, start with ML1 Reach/Push using the compact CMM and
their protocol, only after sections 2-4 give a clear answer.

## Predeclared gates

- At least three training seeds, paired evaluation seeds, parameter match
  within 85%.
- Normal beats memory-wiped with a confidence interval excluding zero on Dark
  Room and K64 bandit.
- Full CMM beats `--ltm-tokens 0` whenever trial length exceeds the short-term
  trace, and not when it is shorter.
- AD with Transformer-XL reaches the published Dark Room level before any CMM
  comparison counts.

## Code needed before launch

Checked against `main` at `1a21c86`:

- `warmstart_train.py` accepts only `persistent_ctm`, `transformer_xl`, `gru`;
  add `cmm` and `ctm`.
- Its objectives are `dpt` and `behavior_cloning`; add an AD objective over
  logged improving learning histories (for example single-task PPO or UCB
  runs with failures kept).
- `screen.py` covers only the Bernoulli bandit; extend the sealed matrix to
  Dark Room and Key-to-Door with the attempt sweep.
- `evaluate.py` has only normal, memory-wiped and reward-blind; add the three
  CMM lesions, an attempt-count override for extrapolation, and probe-feature
  export.

Run CPU tests for each change on an allocated Linux node, then a small
allocated qualification, then ask before the full matrix.
