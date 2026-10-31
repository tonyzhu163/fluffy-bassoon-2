# Review of the steering-interface criteria proposal

Read 2026-10-31 Asia/Shanghai at the user's request. Source:
[PROPOSAL.md](../steering-interface-criteria-20261031-r1/PROPOSAL.md), reviewed
at main `d0853cbe7fbdd0b646f9053c6c2796816257cd76`, blob
`f070c2b71abc4b8568b2dee2ea6de7ff3c31ad1c`.
This is an adoption recommendation, not a changed GPU contract or resource grant.
The optional-hook synthetic CPU qualification can proceed independently.

## Recommendation

Incorporate the three core requirements. The initial eight-demo supervised
fit is a useful capacity probe but does not separate controllability from
fitting quality, establish a reason to use memory, or create an AD dataset.

| Requirement | What it adds | Recommended interpretation |
| --- | --- | --- |
| A1 search reachability | Some intervention can solve a native failure before fitting is judged | Achieved reachability within a declared restricted search and action budget |
| A2 condition dependence and scene transfer | The useful intervention changes with a persistent latent, and evidence transfers to fresh scenes | A controller-by-condition cross-evaluation plus the best single context-blind control |
| A3 reuse teacher learning | Per-condition learning histories and final qualified teachers feed AD/DPT | Reuse actual trajectories and charge their collection once; qualify improvement and teacher competence separately |

The first in-context interface should satisfy all three requirements relevant
to its data route. A large reactive gain alone remains a useful ordinary-control
result. KV and physical-action hooks are alternatives to screen, not winners
already selected by the implementation work.

## Changes needed before adoption

1. **Bound A1 rather than treating search as an oracle.** Declare the searched
   parameters, bases, number of proposals, resets, repeats and all-in actions
   before execution. Fixed random KV bases restrict the class; failed search
   means no candidate found at this budget, not an interface impossibility.
   Rejecting it from this pilot is a triage decision. Two successes in three
   repeats are a development heuristic, not a precise reliability estimate.
   Confirm selected settings on fresh execution draws/cases before claims.

2. **Keep intervention classes and costs explicit.** A per-chunk action vector,
   a low-rank KV coefficient sequence and a full initial-noise tensor have very
   different dimensions and priors. Define the temporal parameterization and
   matched physical-action/computation budgets; equal CEM iterations alone are
   not equal interaction budgets. A1/A2 costs must not be hidden inside the
   existing supervised-fit estimate or borrowed from closed grants.

3. **Make A2 a causal task test.** Hold the reward-relevant condition fixed
   within a trial and keep it hidden from the student's initial observation.
   Cross-evaluate condition-specific controllers on fresh scenes and include a
   single trained context-blind controller. Require evidence that feedback can
   identify which behavior to use. Controller identity alone is insufficient:
   several controllers may share a useful strategy, and a reactive policy may
   infer the condition from current observations. A matched-versus-best-mismatch
   margin is a sufficient development gate when well estimated, not the only
   possible proof of history value.

4. **Do not promise A3 improvement by construction.** Online training may stay
   flat. Preserve failed lineages and require fixed-case improvement beyond
   stationary retry and clock-only behavior. If A2 uses search or supervised
   fitting, online multi-lineage collection in A3 is additional work, even
   though it uses the same conceptual data route. Declare its budget explicitly.
   DPT can proceed from qualified expert query labels and informative behavior
   contexts without an improving context-generating teacher; AD has the stronger
   ordered-behavior requirement.

5. **Resolve the selection/confirmation split.** A4 chooses an interface using
   the original 32-case locked comparison, whereas the governing plan forbids
   exploratory selection on that panel. Either select using separate development
   cases and retain it for confirmation, or explicitly call the 32 cases an
   interface-selection panel and reserve a fresh confirmation bank afterward.
   Do not reuse the selected maximum as an unbiased final performance estimate.

## Recommended sequence

1. Finish synthetic CPU hook qualification. It does not spend A1/A2 robot budget.
2. Bind a source-only motor/task and verify native failures, successful compatible
   teacher trajectories, zero-intervention native parity and the real KV gradient.
3. Run bounded A1 search on development cases; use successful traces to diagnose
   fitting failures. Keep all development cases/costs, including unsuccessful search.
4. Run the reactive fit/control comparison. Define a candidate persistent-condition
   distribution before A2; screen crossover and transfer without opening the final
   confirmation bank. A cheap A2 construction test can precede expensive full fits
   if those fits would otherwise be wasted on a history-irrelevant task.
5. Collect only the required condition-specific learning lineages, retaining
   chronology, and qualify final teachers. Reuse these data for AD/DPT with the
   label spaces below. Then test one GRU with reset/reward/retry controls.
6. Confirm on fresh locked cases before architecture comparisons or broad claims.

## Labels depend on the chosen intervention

| Student interface | AD-style executed target | DPT-style query target |
| --- | --- | --- |
| Noise | Actual executed steering noise, plus the exact frozen decoder contract | Qualified same-motor noise teacher evaluated at the saved query observation |
| Decoded-action residual | Actual applied correction with its base proposal and clipping/valid-duration records; alternatively a declared physical-action objective | Teacher command minus that same base proposal, with bound/reachability handling explicitly defined |
| KV residual | Executed physical action chunks through a supervised flow-matching objective and the frozen expert; no need to imitate nonidentifiable KV edits | Qualified physical teacher actions through the same normalized-action/flow objective |

Historical inputs remain actual observations, actions and rewards. Teacher-only
condition information may produce query labels but never enters student context.
Physical targets require verified transforms, active action-coordinate masks and
valid chunk masks. A changed teacher action expert does not invalidate its physical
action targets automatically; it does invalidate an assumed correspondence between
its RNG seed and the original motor's noise-steering target. Record the chosen
objective and label space before fitting.

No A1/A2/A3 jobs were launched by this review. The proposal remains pending the
user's decision; the governing plan and current runtime budgets are unchanged.
