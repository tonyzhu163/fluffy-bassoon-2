# Collecting one logged retry and a paired audit

The existing archive answers the first cost comparison with zero new robot episodes. It contains one default support per case and both physical retries. At training time, reveal one or both of the two retry labels. More than a single revealed candidate in the paired arm adds physical information even though the fitting objective and readout can stay identical. Use the native simulator outcome and save incomplete or unsuccessful attempts.

## Unit of collection

A trial is one task and one persistent hidden condition, one support on an independently sampled initial pose, and one retry on a fresh initial pose. Choose the condition and reset IDs before collection. Use fixed, versioned candidate policies capable of producing different outcomes on fitting tasks. Start with exactly two candidates. Generate the support using a stationary policy or a prespecified randomized probe with its propensity recorded. In this archive the support always used default cadence, and the query candidates were default and original pacing.

Before executing the retry, draw its candidate by fair coin and record the propensity (0.5). Select a balanced random subset of trials in advance for a second physical query at the same reset and policy noise using the opposite candidate. Choose audit membership independently of rewards or inferred failures. Use a different noise draw for the query from the support. Assign fresh seeds per independent trial and preserve the grouping when the same reset is reused across hidden conditions. The first candidate supplies a valid logged dataset on every trial. Paired-audit cases additionally supply a direct head-to-head; one outcome per branch measures one draw and does not settle a noisy ranking.

## Minimal record

| Data | Fields |
| --- | --- |
| Trial identity and split | Stable ID, task and condition family, fit/dev/test, separate support and query reset IDs, RNG seeds, frozen motor revision, available candidate definitions |
| Public support, chronological | Instruction, observations or cached frozen features, issued commands, observed wrist/gripper motion, rewards, episode boundary, controls and terminal reason |
| Public retry input | Fresh initial observation and instruction, permitted current prefix if applicable, candidate identities or proposals and compute budget |
| Logged evidence | Sampled action and propensity, audit inclusion and its probability, native success, terminal reason, actual charged controls; separately keyed branch outcome if audited |
| Evaluator-only | Hidden condition, simulator state/fingerprint, true predicates, counterfactual branch labels and test outcomes. Keep these out of the deployment input |

Split by task before creating histories. Keep all siblings of the same physical rollout in one split. Store a trajectory once and reference it by ID. Verified simulator predicates may permit hindsight reward relabeling in a separate experiment; it adds neither new physical starts nor independent target measurements. Train the one-label head only on the sampled actions. Train the paired head on all revealed outcomes, including zero/zero and one/one pairs. Keep input, model, initialization schedule and update budget fixed. If only an audit subset is fully labeled, compare one versus two labels on exactly that subset first. Then compare using the remaining logged cases at the same acquisition budget.

## How much to collect

With N histories, K practice episodes per history, one logged query and a second query on A audit histories, the physical count is N(K+1)+A. Score and store the actual control and inference counts too. As a budgeting illustration, eight fitting tasks × two conditions × eight independently sampled support/query pairs gives 128 histories. With one support per history and 32 randomly audited queries, collection costs 288 simulator episodes (128 supports, 128 logged queries, 32 alternates). With two independently executed supports per history it costs 416. These are example fitting costs only. Fresh development/test queries, independent repeat draws and any new motor qualification add to them. Existing compatible rollouts can offset a tranche only after validating task splits, reset identities, motor settings and outcome semantics.

For the next purchase of simulator time, estimate the potential improvement above the strongest same-budget fixed/current/lag rule from development branch disagreements. Examine those cases by task and verify the earlier history could have distinguished them before the decision. If all candidates fail or a simple public rule already reaches the ceiling, improving the menu or changing the mechanism is more informative than repeating the same starts. Where there is headroom, compare more tasks versus more starts at fixed total trials using the registered nested fitting splits. Reserve fresh, fully paired evaluation after freezing the recipe. An inspected development panel cannot provide a confirmatory sample-size guarantee.
