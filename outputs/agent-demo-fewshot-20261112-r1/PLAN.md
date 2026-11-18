# Few-shot agent demonstrations for a frozen π0.5

Not launched. Drafted 2026-11-12 from main `f9e6cd59`. A proposal for review: the benchmark, the agent and the budget below are open decisions, not approvals. It changes direction from in-context reinforcement learning (learning from own attempts and rewards) to in-context imitation (learning from a few demonstrations), and does not alter any active run.

## Question

A frontier model solves a new task in simulation through an agent harness, a few times in different scenes. Given those few demonstrations and no weight updates, does a frozen π0.5 succeed more often on fresh scenes of that task than it does without them?

## Design in one picture

| Stage | Who | Output |
| --- | --- | --- |
| Solve | Frontier model in a harness, in sim | K successful demonstrations of one task, each in a different scene |
| Compile | Fixed code, later optionally a meta-learned reader | A lesson: per step, a subtask sentence and an object-relative end-effector path |
| Execute | Frozen π0.5 | Actions sampled under the sentence as prompt and steered toward the re-posed path |

A **shot** is one successful agent demonstration of the same task in a different scene. Evaluation scenes are never among the shots. K runs over 0, 1, 2, 4 and 8.

## Pieces

### 1. Harness

Robot-level tools only, so the same interface exists on a real robot:

- **Look:** frames from the cameras π0.5 uses, plus proprioception.
- **Move:** an end-effector target or short segment, and the gripper.
- **Done:** the agent's own claim that the task is finished.

Sim pauses between calls. Each commanded segment is recorded at control rate with π0.5's observation and action conventions. The native success predicate is evaluated after the episode for scoring and is never shown to the agent during it. Record per task: solve rate, attempts before the first success, wall time and model cost.

### 2. Lesson

From K recorded demonstrations:

- **Steps and sentences** from the agent's own narration of each segment.
- **Object-relative paths:** the end-effector path in the frame of the object it acts on. Object poses come from sim state after the fact.
- **Agreement:** where the K paths agree in that frame, guidance is tight; where they differ, it is loose. With K = 1 the weight is uniform.
- **Re-posing:** at run time, pick the demonstration whose scene is nearest the current one and map its path onto the current object poses, then convert it to π0.5's action convention.

### 3. Absorption by frozen π0.5

Two training-free mechanisms, both leaving every π0.5 weight fixed:

- **Select:** sample several action chunks and keep the one nearest the re-posed path.
- **Guide:** add the gradient of distance-to-path during flow sampling, as in retrieved-guidance work ([RAGu](https://arxiv.org/abs/2611.10894)).

### 4. Meta-learned reader (later, conditional)

A transformer reads K demonstrations plus the current observation and predicts the target path for the current scene, replacing the hand-coded re-posing in piece 2. It is trained across many agent-solved tasks: K demonstrations in context, a held-out one as the label, no backpropagation through π0.5. Soft-prompt and residual outputs are later variants. This is the DPT arrangement in [Methods](../../docs/METHODS.md) with a demonstration set in place of the attempt history.

## Order of work

Each step states what it decides. None of the decision rules is a numeric admission threshold; they are read from the measured curves.

| Step | What | Compute | Decides |
| --- | --- | --- | --- |
| A | **Channel ceiling.** On tasks where π0.5 sometimes fails, steer it toward a known-successful trajectory recorded in the *same* scene, with select and with guide | GPU, small | Whether steering can change π0.5's outcome at all. If the exact-scene target does not help, no demonstration reader can, and the frozen route stops here for that task |
| B | **Harness.** Build the three tools on the chosen benchmark; run the agent on a handful of tasks | Sim plus model calls | Whether the agent produces successful demonstrations, and at what cost per shot |
| C | **Lesson, offline.** Compile K − 1 demonstrations and predict the path of the held-out one | CPU | Whether hand-coded re-posing is accurate enough, and how the error falls with K |
| D | **Few-shot evaluation.** Frozen π0.5 with lessons from K = 0, 1, 2, 4, 8 on fresh scenes | GPU | The main result |
| E | **Meta-learned reader.** Only if A shows steering works, C shows re-posing is the weak link, and B can supply many distinct tasks | GPU training | Whether a learned reader beats hand-coded re-posing on held-out tasks |

A needs no agent and can run on existing rollouts, so it comes first.

## Step A on RoboCasa365: what existing evidence already says

From [verified findings](../../docs/FINDINGS.md#robocasa365), read 2026-11-12:

| Evidence | Consequence for step A |
| --- | --- |
| Four native retries from matched starts gave scene ICCs of 1.00 for CloseDrawer and 0.885 for OpenDrawer, against 0.222 for CloseFridge and −0.067 for TurnOnMicrowave | **Select** has little to choose from on the drawer tasks, where a new noise draw repeats the outcome. It has outcome variation to work with on Fridge and Microwave. The drawer tasks need **guide** |
| On CloseDrawer, no base movement or wait rescued a native failure, and a per-scene oracle matched the native 10/16 | Those corrections moved the base, not the arm path. Path-level steering is untested there, and one recurrent failure never made gripper contact |
| A held −2 cm lateral correction opened two of six natural OpenDrawer failures | A small path change can flip an outcome on this task: the most direct existing sign that the channel has some capacity |
| On compact-bowl scenes frozen π0.5 grasped 8/8 but released right whatever side was requested; a scripted carry released on the requested side in 6/8 continuations; LoRA students reached 3/32 to 7/32 | A same-scene successful path exists (the scripted carry) for a behaviour π0.5 does not produce from language. This is the cleanest step A case |
| Fixed PPO teachers beat SFT on paired cases (OpenDrawer 331 → 369 of 512, Toaster 168 → 212, Faucet 204 → 242) | Cases the SFT policy failed and the teacher solved would give exact-scene successful action targets, if the teacher's executed actions were saved per case (not yet checked) |

Proposed step A cases, in order: compact-bowl carry (target: the scripted carry), then OpenDrawer natural failures (target: the −2 cm rescued tapes, which replayed exactly), then teacher-solved cases if their action tapes exist.

Code state: [`FrozenRoboCasaMotor.infer_batch`](../../src/fluffy_bassoon/robocasa/residual.py) already draws several chunks from one prefix pass with explicit noise seeds, which is what select needs. Guide needs a hook inside the denoising loop of RLinf's π0.5 evaluator; none exists here yet.

### Step A implementation (written 2026-11-12, not yet run)

- [Guided and select sampler](../../src/fluffy_bassoon/robocasa/guided_sampling.py). The compact-bowl stack is native JAX OpenPI, so the sampler is a copy of the pinned `Pi0.sample_actions` loop with one added line. At each denoising step, guide moves the clean-action estimate a fraction `w` of the way to the target on the steered coordinates (arm translation and gripper); `w = 0` is the native sampler and `w = 1` lands on the target for those coordinates, leaving rotation, base and mode to π0.5.
- [Rollout script](step_a.py) and [job script](run_step_a.sh). Per scene: one frozen pickup under a neutral prompt up to the first grasp, then every arm replays that exact prefix. Arms per requested side: scripted carry (its executed commands become the target tape), native, guide at `w` = 0.25, 0.5 and 1, and select among 8 samples. Policy arms share one noise stream. The tape is indexed open-loop by controls since the grasp.
- Checks made in the job, not as repository tests: decoded actions at `w = 0` must match `policy.infer` on the same noise to 1e-4 (the copied loop is otherwise unqualified), and the output transform must be the assumed affine map on the steered coordinates.
- Checked on the Mac only: syntax, and the guidance arithmetic with a stand-in velocity field (`w = 0` reproduces the plain loop exactly; `w = 1` reaches the target to 4e-16). Nothing has been imported against OpenPI or run in the simulator.

Launch inside a one-GPU Baige job from a clean checkout of the commit that contains these files:

```bash
outputs/agent-demo-fewshot-20261112-r1/run_step_a.sh <commit> stepa01 --seeds 961001 961002 961004
```

Those are the three development scenes where the scripted carry reached both bowls. Read from `rows.json`: for each scene and side, which bowl each arm released into, and how the gap to the tape and the outcome change with `w`. This session had no route to Baige.

## Comparisons at step D

All on the same fresh scenes, paired by scene:

| Arm | What it isolates |
| --- | --- |
| π0.5 alone | Floor |
| Sentence only, no path | What language carries without steering |
| Lesson from a different task (donor) | Whether the particular demonstrations matter, as in [the research question](../../docs/PAPER.md) |
| Open-loop replay of the re-posed path | Whether π0.5 adds anything over following the path |
| Fine-tuning π0.5 on the same K demonstrations | The SFT route of [HarnessPAI](https://arxiv.org/abs/2609.29166), at matched K |
| The agent solving the evaluation scene itself | Ceiling, with its time and cost |

Report native success against K for every arm, first-attempt only, with uncertainty paired by scene, plus cost per new task (agent time and model spend, GPU time). Orchestration-only steering as in [Harness VLA](https://arxiv.org/abs/2607.08448) is a literature comparison on shared benchmarks unless reproducing it proves cheap.

## Open decisions

1. **Agent and harness.** Which frontier model, and whether it runs inside an existing coding-agent harness with the sim exposed as tools.
2. **Whether fitted soft prompts count.** Proposed: a baseline, since it is gradient training, not part of the method.

Decided 2026-11-12 by the user: **RoboCasa365 atomic tasks** for steps A to D. They have a qualified π0.5 (Human300) with known partial failures and published agent baselines. The generated LIBERO scenes have no qualified motor (Behavior Prompting scored 0/8 on moved layouts; π0.5 has not been run there) and are kept only for a later rule-inference variant.

## Not yet verified

- That Baige job containers can reach the model provider, or that the agent can run elsewhere against a sim on Baige.
- That a frontier model solves these tasks through robot-level tools; published harness results use different tools and benchmarks.
- That an end-effector path converts cleanly to the Human300 checkpoint's action convention.
- That guidance can be added to the existing PyTorch π0.5 sampler without changing unguided outputs. Check numerical parity with guidance weight zero.
- Literature claims here come from abstracts, not full papers.

## Budget

Step A: at most 10 GPU-hours. Steps B and C: simulator time and model spend, both unmeasured; record them from the first task. Size step D from A to C and ask before launch if the cumulative total would pass 50 GPU-hours. Step E is a separate request.
