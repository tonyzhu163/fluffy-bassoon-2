# Learning histories: can a Transformer learn from attempts?

This synthetic experiment measures how much information a history reader extracts when execution is perfect. The enumeration reference and the frozen learner can be evaluated on exactly the same attempts. The project uses LIBERO-Pro2.0 to test transfer in manipulation. [Benchmark design](../../docs/PAPER.md)

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

Training uses supervised correct-choice labels, not online policy-gradient updates. At evaluation, the visible history is supplied again to each forward pass; the implementation does not carry a separate recurrent hidden state between calls. A feedback intervention can establish useful input dependence, but does not by itself identify a belief representation or Bayesian update mechanism inside the Transformer.

Later training-actor choices depend on earlier rewards, so the blind condition can indirectly observe feedback through those choices. Its evaluation policy has no such reward access. Comparisons between separately trained conditions therefore need a follow-up intervention on the same frozen learner and histories (mask or shuffle feedback) before claiming a causal feedback benefit. The query imitation objective also does not train information-seeking exploration.

```bash
for c in demo blind attempts corrections linked; do
  python train.py --cond $c --K 6 --out ../../outputs/<run-id>/$c-s0.json
done
```

The completed five-condition, three-seed grid used K=6 and 20,000 CUDA updates per fit. On held-out rules, attempts improved from 45.44% to 57.05% at k=6, versus 86.50% for the reference reading the same histories. A similarly large gap remained on training rules. All seeds failed the original near-reference screen; see the [curves and qualification evidence](../../outputs/learning-histories-20261111-r1/RESULT.md).

The [paired feedback follow-up](../../outputs/learning-histories-feedback-20261111-r1/RESULT.md) reproduced the original curves exactly. On held-out random-attempt histories, the same checkpoints scored 54.65% with real feedback, 43.94% with outcomes masked and 38.83% with outcomes shuffled. All three seeds showed a benefit. Training-style histories explained only 1.77 points of the held-out deficit; a large reading gap persisted even on training rules.

Substantial gaps also appear on training rules and in the `corrections` condition, which involves no attempt policy. The completed five-recipe, three-seed screen tested denser supervision, Fourier attribute features, learning rate and a larger model. Fresh confirmation of the selected dense + Fourier corrections recipe reached 75.64%, versus 67.40% originally and a 94.94% same-history enumeration reference. Attempts reached 60.50% versus 57.34%, with one seed slightly worse. All fits still failed the near-reference screen. Dense training changes attention and the objective together; these recipe comparisons do not isolate a mechanism or substitute for the same-checkpoint feedback intervention. No longer-duration fit was tested. See the [confirmed results and limitations](../../outputs/learning-histories-fix-20261111-r1/RESULT.md).
