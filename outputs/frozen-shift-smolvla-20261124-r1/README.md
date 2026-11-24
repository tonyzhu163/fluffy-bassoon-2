# Frozen SmolVLA native CPU harness

[Plan](PLAN.md) · [Result](RESULT.md)

One Python environment holds the pinned LIBERO-Pro2.0 simulator and LeRobot 0.4.4. No GPU is used. Arms are swapped by [`ArmEnv`](../libero-crossarm-20261120-r1/arm_env.py), imported unchanged.

```sh
bash setup.sh "$HOME/smolvla-scratch"                        # once, about 6 GB and a few minutes
bash run.sh "$HOME/smolvla-scratch" raw Kinova3 14           # one native episode -> evidence/raw/Kinova3-goal3-layout14.json
HOOKS=adapters:LiberoPrompt bash run.sh "$HOME/smolvla-scratch" libero-prompt Panda 14 15
TASK=4 NOISE_OFFSET=1000003 bash run.sh "$HOME/smolvla-scratch" noise2-task4 Panda 0 1
uv run --no-sync pytest -q -W error .                        # CPU fixture, no simulator or weights
```

`run.sh SCRATCH TAG ARM LAYOUT...` loads the policy once and runs the layouts in order. An existing episode file is skipped. Reset seed is `42710300 + layout`. Policy noise for the whole episode is `numpy.random.default_rng(reset seed + NOISE_OFFSET)`, one `(1, 50, 32)` standard-normal draw per proposal, so two arms with the same layout and offset receive the same noise sequence. One 300-control episode takes about 2 to 2.5 minutes on two threads. Several `run.sh` processes can run side by side.

## Adapters

Subclass `harness.Hooks` in a module on `PYTHONPATH` (this directory is on it) and pass `HOOKS=module:Class`.

| Hook | Receives | Returns |
| --- | --- | --- |
| `observation(obs, context)` | public images, joint and tool pose, fingers | the observation the motor preprocesses |
| `state(state8, obs, context)` | unnormalized policy state after LeRobot's LIBERO conversion | 8D state |
| `prompt(prompt, obs, context)` | released instruction | instruction for this replan |
| `noise(draw, context)` | next draw of the episode's noise sequence | noise of the same shape |
| `choose(proposals, obs, context)` | `Hooks.proposals` chunks of 50×7 commands | index to execute |

`Hooks.replan` sets how many controls of the chosen chunk are executed. `context` has `step`, `replan` and the issued `commands`. Hooks never see reward, success, contacts, joint velocity or object state. Each episode file records success, control count, the prompt, state and image digests, noise and proposal digests per replan, the consumed commands, scene fingerprints and a parameter readback before and after.
