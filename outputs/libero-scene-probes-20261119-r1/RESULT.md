# What did the frozen robot learn from practice?

The visible layout reversed which target the frozen π0.5 could reach. Reading the reward from the first attempt substantially improved the retry. Choosing the first target from images and proposed actions added one native success on the prespecified pairs and tied a fitted fixed target across all familiar pairs.

## One practice attempt, then a scored retry

We executed 64 fitting and 96 fresh LIBERO simulator branches. Each layout and pose had a common physical reset across candidate instructions. A trial held the hidden goal and layout fixed, used a different pose for the retry, and allowed 300 controls per attempt. Every branch recorded all candidate native predicates throughout execution. The scene selector was frozen before fresh collection. Its three extra π0.5 proposals per option cost inference but used no simulator controls. The fixed baseline picked a first target per pair on fitting poses. Both repeated their first target on reward and switched on failure.

| Fresh development panel | Scene selector | Fitted fixed target | Balanced first target | Best first target per start in hindsight | Query motor ceiling with known goal |
| --- | ---: | ---: | ---: | ---: | ---: |
| Two prespecified cross-object pairs | 53/64 | 52/64 | 51.5/64 | 54/64 | 54/64 |
| All six familiar pairs, including the two above | 157/192 | 157/192 | 154.5/192 | 162/192 | 162/192 |
| Drawer versus bowl-to-stove, both excluded from fitting | 21/32 | — | 21/32 | 22/32 | 22/32 |

All six familiar pairs had a fresh start where only the first candidate reached its own target and another where only the second did. In the 28 cases where exactly one succeeded, the scene selector picked it 22 times versus 17.5 expected for the fitted fixed target. On the primary endpoint the entire one-success advantage came from one of the four fresh pose IDs. Substituting another layout's selector inputs gave 52/64 primary and 157/192 familiar. These are paired branch replays. Pairs, hidden goals and cyclically linked poses reuse trajectories. Fractions are expectations under tied or randomized decisions. [Per-pair evidence](scene02-evidence.json) · [Scenes and outcomes](scene02-panel.png)

The feedback rule fitted on the two same-object training pairs was repeat after success and switch after failure. With the scene selector held fixed, it earned 157/192 familiar retries versus 81/192 for a fitted reward-blind rule. The same comparison gave 53/64 versus 27/64 on the prespecified pairs and 21/32 versus 11/32 on the unseen pair. An explanatory check put every query in the base layout while keeping varied support layouts. The familiar score became 185/192 versus 181/192 for a fixed target refitted on analogous source trials, with a per-start ceiling of 191/192. Query execution can hide differences in what the first attempt reveals. The persistent-layout endpoint above is the primary comparison.

## Does the first reward help choose the second practice attempt?

After reading the pair results, we replayed all four triples of familiar goals. Each trial kept its layout and hidden goal across two practice attempts and a query on three distinct poses, with 300 controls each. We fitted a separate action schedule and reward-to-query table for each triple on the source poses. We compared all nine fixed two-probe schedules with all 27 schedules that could choose their second probe from the first reward. Both decoders saw both rewards. We preferred a fixed schedule in a fitting tie, averaged remaining tied choices, and used the source marginal query choice for an unseen reward sequence. The one-reward and blind controls ignored the second and both rewards respectively under the same physical budget.

| Information and choice allowed | Fitting successes / 192 | Fresh successes / 192 |
| --- | ---: | ---: |
| Ignore both rewards | 64 | 58 |
| Read the first reward | 116 | 105 |
| Read both rewards, fixed probe order | 151 | 154 |
| Read both rewards, allow adaptive second probe | 151 | 154 |

On every triple a fixed schedule attained the maximum fitting score, so the tie rule selected the same policies in both two-reward arms. The 49-success gain over reading one reward shows the second observation carried information. This comparison supplies no evidence that choosing the second probe adaptively helped. Even with the hidden goal revealed, some queries failed. The best available query action could succeed on 162/192 assignments. The four fresh pose IDs and reused branches make this a development diagnostic. [Fitted policies and scores for each triple](three-goals.json) · [Replay implementation](three_goals.py)

## Paper decision and run record

Use the fixed two-probe schedule with a fitted reward decoder as the comparator for learned exploration. On fitting tasks, measure whether seeing the same sparse reward after a wrong placement versus a failed grasp calls for different executable corrections. Then test a frozen reader of public action and motion history on new tasks and poses with native rewards and equal control budgets. Report goal identification and native query success separately. The official eight-task test split was untouched.

Baige development RTX 4090, about 0.60 GPU-hours across collection workers (945 and 1,206 seconds). Worker `83282136` completed the fit and six fresh base-layout branches before encountering an unpublished drawer S02 label file. Worker `79204b7e` reused those branches and the frozen bank, scored the BASE native predicates in each shared world, and completed collection. We checked BASE/S02 prompt and predicate equality for all four fitting goals and verified four distinct initial simulator states and camera views per pose. Readout `193d824b` recomputed the native hits and fresh selector scores. The three-goal analysis at `fdb708e3` used stored trajectories on CPU. [Run plan and migration](PLAN.md) · [Final worker exit](scene02-exit.txt)
