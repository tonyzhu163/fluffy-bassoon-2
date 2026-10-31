# Proposed amendment: choosing the steering interface

Drafted 2026-10-31 Asia/Shanghai against the governing
[frozen pi0.5 steering screen](../frozen-pi05-steering-screen-20261031-r1/PLAN.md)
at `caff7b7`. **Proposal only.** It does not change that plan's contract, its
owner's row or any resource ceiling. Adopting it is the user's decision, and
the plan owner then records the change.

## What the amendment addresses

The plan compares native, a KV residual and a physical-action residual, each
fitted by supervised imitation of about eight teacher trajectories, and passes
an arm with a 15-point paired gain on 32 locked cases. That answers whether an
arm can be *fitted* from demonstrations. Three questions stay open, and the
answers decide whether an arm is useful for in-context RL, not just for
reactive control:

1. If a fitted arm fails, was the interface unable to express a success, or
   did the small fit fail? The plan cannot yet distinguish these.
2. Does the best intervention differ between latent conditions? If one setting
   is best for every condition, a static controller suffices and history has
   nothing to add, however large the reactive gain.
3. Which histories feed AD and DPT? The plan requires real teacher learning
   for AD and same-interface teachers for DPT, but does not say where they
   come from.

## A1. Reachability by search, before fitting

On the development bank (never the locked panel), for each native-failure
case and each arm, search directly over the arm's intervention from the same
reset: CEM or a similar derivative-free search over a low-dimensional
parameterization, with three repeats of each candidate setting. The plan's
eight development resets per task may contain few native failures; if so, add
a separate predeclared reachability bank rather than using the locked panel.
Report, per arm:

- cases where some searched setting succeeds in at least two of three repeats;
- the search budget used, charged like any other physical actions.

Search is an achieved lower bound on what the interface can express, not a
ceiling. An arm with no searched success on any native failure is rejected for
this task before its supervised fit is interpreted. A fit failure on cases that
search solved is reported as a fitting or data problem, not an interface limit.

For the KV arm, search over the adapter's low-rank output coefficients with
the bases fixed at a declared random draw; report that this restricts the
search. For the action arm, search over a per-chunk correction within the
declared joint and gripper bounds. Optionally include per-chunk initial noise
as a reference arm, because it is DSRL's interface and cheap to search; its
inclusion is fixed before the development bank is opened.

## A2. Condition dependence, before collecting histories

When the latent-condition distribution is declared, fit or search one
controller per condition on development cases and evaluate each on every
condition's held-out development cases. The arm qualifies for the history stage
only if, for at least two conditions, the matched controller beats the best
mismatched controller by a predeclared margin on paired cases.

Without this, a reward-relevant latent may exist while the same intervention
remains best everywhere; then a trained static controller matches any learner,
and a null adaptation result says nothing about memory.

Within each condition, also report transfer across fresh scenes: fit on some
scenes, evaluate on others. Evidence from one attempt can only inform the next
if the right intervention stays right across scenes.

## A3. Reuse per-condition training as AD and DPT data

Train at least one A2 per-condition controller per condition online, with
complete chronology: early, intermediate and late episodes, failures and
unsuccessful updates, in interaction order, from more than one independent
lineage. These runs:

- are AD histories that contain improvement by construction, subject to the
  plan's existing check that improvement holds on fixed separate source cases
  and beyond a stationary retry policy;
- leave final per-condition policies on the same frozen motor and interface,
  which can label DPT queries once each is qualified on its own condition.

This adds no new data route. It satisfies the plan's existing AD and DPT
requirements with runs A2 already needs.

## A4. Selection rule

Choose the interface for the first adaptation test in this order:

1. Pass A1 reachability on the selected task.
2. Pass A2 condition dependence on the declared condition distribution.
3. Among arms passing both, prefer the larger paired gain from the plan's
   locked evaluation, then fewer losses on native-success cases, then lower
   cost per decision.

An arm with a large reactive gain that fails A2 is a reactive-control result
and a DSRL-style baseline. It is not the interface for the in-context test.

## Cost and ownership

A1 and A2 add physical actions and GPU time. They belong inside the execution
owner's single all-in phase ceiling, measured before launch like every other
part of the phase, and add no separate allowance. Order within the phase: A1,
then the plan's supervised fits and locked evaluation, then A2 and A3 once the
latent distribution is declared.
