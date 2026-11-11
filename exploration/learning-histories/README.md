# Learning histories: can a Transformer learn from attempts?

Exploratory diagnostic, not part of the paper plan. The completed three-seed grid is recorded in the [run readout](../../outputs/learning-histories-20261111-r1/RESULT.md).

The RoboCasa and RoboTwin students showed no consistent gain from retaining history ([findings](../../docs/FINDINGS.md)). This folder asks the prior question in the cheapest setting available: when history *does* carry information, does a plain Transformer learn to use it, and how close to optimal does it get?

## What an environment needs

1. **A hidden variable the query cannot reveal.** The correct action must depend on something that only the history identifies. Otherwise a history model has nothing to learn.
2. **A motor that executes every alternative.** A failure must mean a wrong choice. If the motor cannot do the task either way, no history can help.
3. **Outcomes that separate hypotheses.** A failed attempt must rule something out.
4. **Fresh scenes between attempts and query.** Copying a past action must fail, so the lesson has to transfer.
5. **Many rules, with some held out.** With a handful of rules the learner memorizes them instead of learning how to update.
6. **A computable reference.** Exact rule enumeration measures how much the recorded history can reveal. A formal optimum additionally requires matching the evaluator's rule prior and specifying the attempt policy.

The existing robot failure tapes mostly fail requirements 1 and 2. Outcomes repeat at a fixed scene (ICC 1.00 for CloseDrawer, 0.885 for OpenDrawer), and no tested correction rescued a CloseDrawer failure, where a per-scene oracle matched the native 10/16. Those are execution failures with no available alternative. A history model can learn to predict them, but not to avoid them.

So the first collection should use an actor that is **competent at execution and ignorant of the hidden variable**. A scripted or already-qualified motor supplies the competence. The ignorance supplies informative mistakes. An improving PPO teacher is the wrong source: its mistakes are execution noise.

## The toy

`sessions.py` builds hidden-rule selection sessions. A rule is "pick the object whose attribute *d* is closest to *v*", over one attribute or two. Scenes hold five objects with three continuous attributes, so every rule names one object in every scene. There are 126 rules; 27 two-attribute rules are held out of training. A session is one teacher demonstration, K attempts in fresh scenes, and a query in another fresh scene. Execution is perfect.

The reference enumerates all 126 rules with a uniform prior and chooses greedily. Each evaluation split actually samples a subset of those rules, so this is not the split-specific Bayes optimum or an information-optimal exploration policy. Its query accuracy, from `python sessions.py --K 6 --n 20000` on training rules (held-out rules agree within 0.02):

| Attempt scenes shown as | k=0 | k=2 | k=4 | k=6 |
| --- | ---: | ---: | ---: | ---: |
| nothing, or the choice without its outcome | 0.488 | 0.488 | 0.488 | 0.488 |
| the choice and its outcome | 0.488 | 0.615 | 0.779 | 0.908 |
| the teacher's choice | 0.488 | 0.745 | 0.888 | 0.952 |
| choice, outcome and teacher's choice | 0.488 | 0.745 | 0.888 | 0.952 |

Chance is 0.20. Two things follow before any model is trained:

- Success/failure feedback alone is worth 42 points here, with no correction teacher.
- When a correction shows the full right answer, the linked failed attempt adds nothing. Testing whether attempt–correction linkage matters needs *partial* corrections that are ambiguous without the attempt.

## The learner

`train.py` fits a four-layer Transformer over object tokens, one fit per context condition. Each session is trained at a random prefix length, so the fit covers the whole learning curve. Training attempts come from posterior sampling with 20% random choices. Evaluation uses the learner's **own** greedy attempts with all weights fixed, and reports three curves: the learner, the enumeration reference reading the learner's own histories, and the reference making its own attempts.

Later training-actor choices depend on earlier rewards, so the blind condition can indirectly observe feedback through those choices. Its evaluation policy has no such reward access. Comparisons between separately trained conditions therefore need a follow-up intervention on the same frozen learner and histories (mask or shuffle feedback) before claiming a causal feedback benefit. The query imitation objective also does not train information-seeking exploration.

```bash
for c in demo blind attempts corrections linked; do
  python train.py --cond $c --K 6 --out ../../outputs/<run-id>/$c-s0.json
done
```

The completed five-condition, three-seed grid used K=6 and 20,000 CUDA updates per fit. On held-out rules, attempts improved from 45.44% to 57.05% at k=6, versus 86.50% for the reference reading the same histories. A similarly large gap remained on training rules. All seeds failed the original near-reference screen; see the [curves and qualification evidence](../../outputs/learning-histories-20261111-r1/RESULT.md).

The [paired feedback follow-up](../../outputs/learning-histories-feedback-20261111-r1/RESULT.md) reproduced the original curves exactly. On held-out random-attempt histories, the same checkpoints scored 54.65% with real feedback, 43.94% with outcomes masked and 38.83% with outcomes shuffled. All three seeds showed a benefit. Training-style histories explained only 1.77 points of the held-out deficit; a large reading gap persisted even on training rules.

## Decision rule

Read each fit on held-out rules, three seeds per condition. These original screening rules use "Bayes" to mean the full-family enumeration reference above; they are diagnostic thresholds, not proofs of optimality or causal feedback use.

| Outcome | Reading | Next step |
| --- | --- | --- |
| `attempts` within 5 points of Bayes at every k, and `blind` no better than `demo` | A plain Transformer learns the update from outcomes. | Move to continuous actions and partial corrections, then a physics environment with a qualified motor. |
| Close to Bayes on training rules, more than 10 points short on held-out rules | It memorized the rule set. | Enlarge the rule family before changing the model. |
| More than 10 points short on training rules after 20,000 steps | The update is not being learned. | Fix the objective or architecture here. Do not move to robots. |
| `blind` beats `demo` | The learner is reading the actor's competence from its choices, not the outcomes. | Change the attempt source before interpreting any gain. |
