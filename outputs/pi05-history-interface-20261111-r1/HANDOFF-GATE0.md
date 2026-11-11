# Gate 0 handoff: released π0.5 on LIBERO-Gen

For the agent that owns the [LIBERO-Gen history pilot](../libero-gen-history-pilot-20261111-r1/RESULT.md). The human assigned this gate to you on 2026-11-11 because the planning session has no Baige route. It is a motor check only. It does not block or change your custom-layout stage; run it when that stage leaves a free slot.

## Question

Does the released language-conditioned π0.5 LIBERO-Gen checkpoint execute the candidate tasks reliably enough to serve as the motor under a trained history input? Your Behavior Prompting result (20/20 on both bowl-to-plate choices) is the reference.

## What to run

1. **Source.** The Behavior Prompting README points to an openpi fork with a `liberogen` branch and released π0.5 LoRA checkpoints. Record the fork URL, commit, checkpoint name and hash. These were not verified from the planning session; if the checkpoint is gated, missing or needs a different config, stop and record that.
2. **Runtime.** Use the fork's own serving and evaluation path (JAX) in a separate environment. Do not modify the qualified BPP runtime or its shared caches.
3. **Tasks.** Use your paired native initializations.
   - The two qualified tasks: center bowl to plate and near-plate bowl to plate, 20 initializations each.
   - Four more from the largest shared-scene groups in [RELEASE-AUDIT.json](../libero-gen-history-pilot-20261111-r1/RELEASE-AUDIT.json), chosen before any rollout: two with a different pick object and two with a different placement target, 10 initializations each.
4. **Prompts.** Each task's released instruction. Then repeat the two bowl tasks with one generic instruction ("do the demonstrated task") as a floor: the history interface will train with a generic instruction, so we need to know which bowl the model takes when language says nothing.

That is 80 language episodes and 40 generic ones.

## Record

Per episode: task, initialization, instruction, native success, steps, and for the generic runs which object was grasped. Also inference seconds per action chunk and peak GPU memory, which size the later training.

## Pass condition

At least 85% native success on each language-prompted task (17/20 or 9/10). Report all failures. The generic runs have no threshold.

Save one batch of raw observations with the JAX actions the server returned for them (at least 16 observations, fixed noise seed if the server exposes one). Gate 2 compares a PyTorch conversion against these.

## Budget and reporting

At most 4 GPU-hours on one GPU, counted against this run's 10 GPU-hour cap for gates 0 to 2. Use a fresh output identity under `outputs/pi05-history-interface-20261111-r1/gate0-<n>/`, write `GATE0-RESULT.md` beside this file, and update your own row on the coordination board.
