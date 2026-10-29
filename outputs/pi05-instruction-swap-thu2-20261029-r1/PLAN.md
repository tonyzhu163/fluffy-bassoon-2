# Instruction-swap check for frozen pi0.5 on ManiSkill (2026-10-29)

Question: does pi0.5's language instruction choose which object it manipulates?
This decides whether a hidden-target in-context RL task (vague instruction,
reward reveals the correct object) is viable on this motor. Nothing earlier in
the repository tested instruction following; the existing scenes hold one
object and a plate, so the RL-finetuned motor could ignore language.

No training, no PPO. Frozen weights, RLinf eval-mode sampling, per-batch seeded
noise. Secondary school cluster (thu2), RTX3090 partition, one GPU, 60-minute
scheduler cap per job, at most two jobs: RL checkpoint first, then SFT.

## What RLinf e1a82eb supports (inspected on thu2)

- `PutOnPlateInScene25Main-v3` places exactly one object (`select_carrot_ids`)
  and one plate; all other objects are parked off-table. The instruction is
  `f"put {model_db name} on {plate name}"`, e.g. `put carrot on plate`.
  `use_multiple_plates=True` only varies the plate (17 plates); it adds no
  distractor object. `obj_set` selects objects 0-15 (`train`), 16-24 (`test`)
  or all 25.
- RLinf ships `PutOnPlateInScene25MultiCarrot-v1` (variants package): the named
  object, one extra object from the same set (episode-id encoded, never the
  same object) and the plate, with 0.15 m minimum object separation. Its
  `evaluate()` and instruction concern only the named slot-0 object.
- `PutOnPlateInScene25Instruct-v1` paraphrases the template (21 templates);
  not used here.

## What is built (no RLinf patch)

`src/fluffy_bassoon/rl/maniskill_instruction_swap.py` registers
`PutOnPlateInScene25InstructionSwap-v1`, a subclass of RLinf's MultiCarrot
scene that (1) accepts an instruction override and (2) reports, every step,
grasp, RLinf's own on-plate success test and object-to-plate distance for
both objects. Episode ids are encoded/decoded with RLinf's arithmetic and
checked after every reset. The slot-0 condition uses RLinf's native prompt
and asserts it equals the template used for the other conditions. The RL
checkpoint loads through the existing flow-noise key patch copy
(`patches/rlinf-e1a82ebf-flow-noise-keys.patch`), as on Baige.

Driver: `scripts/diagnostics/maniskill_pi05_instruction_swap.py`;
launcher: [launch.sbatch](launch.sbatch).

## Design (fixed before launch)

Objects 0 carrot, 9 toy bear, 12 banana, 13 hamburger (training range; each
placed 2/2 by the SFT motor in the thu2 smoke, distinct shapes and names).
Pairs: carrot/toy bear, banana/hamburger, carrot/hamburger, toy bear/banana.

| Condition | Scenes | Episodes per checkpoint |
| --- | --- | ---: |
| a. Single object, Main-v3, native prompt | 16 per object | 64 |
| b. Multi, prompt names slot-0 object | 24 per pair (12 per slot order) | 96 |
| b. Same scene/reset/noise, prompt names slot-1 object | same 96 scenes | 96 |
| c. Same scene/reset/noise, `put object on plate` | same 96 scenes | 96 |

Batches of 32 environments. Each block of 32 scenes mixes all four pairs and
both slot orders, and runs the three prompts with identical episode ids,
reset seed and sampler seed. An adaptive guard starts a block only if it is
expected to finish before about 54 minutes into the allocation, so partial
runs keep complete matched blocks. Total: 352 episodes per checkpoint.

Chosen object per episode: the object placed on the plate (RLinf success
test applied to that object) if any, else the first object grasped, else none.

## Metrics and provisional criteria (reported, not tuned)

- Explicit success per object: named object on the plate at any step
  (also the clean version: only the named object placed). Criterion: >= ~60%.
- Swap-follow: fraction of paired scenes whose chosen object matches the named
  object under both prompts (also: chosen object changes with the prompt),
  among all pairs and among pairs where either attempt grasped something.
  Criterion: >= ~80%.
- Vague distribution of the chosen object, by pair and by slot. Criterion:
  near chance (0.5) among acted episodes.
- Gap = explicit success - vague success for the same object over the same
  scenes. Criterion: >= 30 points.
- Single-object success per object for the same four objects (sanity).

## Validation before launch

macOS: `py_compile`, ruff F/E9 on new files, `bash -n` on the launcher.
Linux: each job runs the new CPU tests plus the existing ManiSkill and
checkpoint-compat tests inside the allocation before any model work, then the
GPU probe inside the step (`ms.prepare_gpu`).

## Execution record

Filled in after submission: commit, job IDs, nodes, outcomes. Raw outputs:
`/cluster/nvme5/zhujiayang/fluffy-bassoon-2/outputs/pi05-instruction-swap-thu2-20261029-r1/<ckpt>-job<id>/`.
