# Native-context pilot completed: no observed adaptation benefit

Baige **`job-lwpvhzys2vtc` succeeded at 2026-10-28 20:00:45 Shanghai**,
exit 0. Runtime source `0d6dee3d2a792dbede86fe754425bdd361e3cdde` stayed
clean and immutable. The driver finished in 1,899.94 seconds, within the
original cap. Completion audit was performed at 21:49–21:50 Shanghai on the
existing allocated CPU machine, without executing a model or launching a job.

The native parity/gradient/replay gate passed, five complete K4 labeled
histories were collected, and the paired +45-degree oracle execution gate
scored **4/4 versus 1/4** for the unadapted base collector. Twenty supervised
updates completed. Held-out validation flow loss fell from **0.68696 to
0.51411** (the same six fixed queries/noise samples). That is an offline loss
change, not live adaptation evidence.

Final deterministic smoke evaluation used two K4 held-out trials per
condition, at -30 and +30 degrees:

| History condition | Successes / attempts | Successes by attempt across the two trials |
| --- | --- | --- |
| Normal | **2/8 (25%)** | 2, 0, 0, 0 |
| Wiped between attempts | **5/8 (62.5%)** | 2, 1, 1, 1 |
| Reward blind | **2/8 (25%)** | 2, 0, 0, 0 |

There is no observed context or reward benefit in this tiny pilot. Normal
and reward-blind arrays match exactly; wiping history scored higher. Two
trials cannot establish reliable harm, generalization, or an architecture
ranking. Evaluation source fixes the same scene/noise seeds across conditions;
only per-trial outcomes were saved for this panel, not full evaluation traces.
There is no matched frozen-base or warm-start-only live evaluation panel.

**PPO did not demonstrate RL learning.** Its one K4 rollout had four failures,
zero rewards and zero old values, hence zero advantages. All 64 chunks
replayed exactly, but the saved loss and gradient norm were both **zero**.
The optimizer executed one step; its warm-start Adam momentum can still move
parameters even with zero new gradient. The `ppo_smoke: true` flag means that
the code path completed, not that it received an informative RL update.

The completion audit independently checked all five histories' public
command fields, delayed terminal rewards, boundary/clocks and checkpoint
hashes; reconstructed all six evaluation JSONs against the summary; and
verified the recovery checkpoint has 20 warm-start updates, PPO completion,
finite trainable weights, optimizer state, sampler state and Python/NumPy/
Torch/CUDA RNG state. Optimizer counters are 21 for used warm-start tensors
and 1 for the critic. Frozen original weights passed the driver's final
fingerprint assertion before its successful summary. Total task rollout cost
was **4,160 simulator steps** across 13 K4 trials, excluding reset/settling
and the observation-only gate.

The complete checkpoint is retained on PFS:

`/mowenjie/fluffy-bassoon-2/outputs/native-vla-context-pilot-20261028-r4/execution/latest.pt`

SHA256 `2ea6b5644f8c91673f807116f1bde3d7cde51dcb7ba50b05943e106df1f6b4fa`.
The five history hashes in it match the actual files. Raw histories and PPO
rollout remain retained on PFS; small evidence is saved beside this report:
[summary](evidence/summary.json), [PPO](evidence/ppo-smoke.json),
[oracle gate](evidence/label-competence.json), and
[independent completion audit](evidence/completion-audit.json).

The bounded pilot is closed and its 20-minute monitor deleted. No follow-on
job was launched. Before a larger run, the missing prerequisites are an
informative, nonzero PPO learning signal, a deliberate optimizer-state
transition between warm start and PPO, and matched base/warm-start controls.
