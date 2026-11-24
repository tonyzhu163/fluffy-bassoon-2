# Shared harness for the proposal-shift lanes

Lane plan, written before any harness result. Program plan: [PLAN.md](PLAN.md).

## Question

Can one episode driver apply every lane's adapter to the frozen π₀.₅ loop so that an untreated branch and an adapted branch differ only in what the adapter changed, and can a panel compare them on paired resets with native reward?

## Prediction

An adapter with every hook at its default sends the simulator the same commands, and gives the policy the same prompt, inputs and noise, as `ur5e_motor.run_episode` without the harness. Each hook then changes one recorded quantity: `observation` the input digests, `prompt` the prompt, `noises` the candidate noise digests and count, `choose` the chosen index, `command` the consumed window.

## What counts against it

On the CPU fixture: any command, prompt, input digest, noise digest or policy-call difference between the identity adapter and the plain motor loop; a hook that moves a quantity it does not own; an adapter that receives reward, fingerprints or non-public observation keys. These are software checks. They say nothing about task success.

## Prior work checked

[Selector rollout and driver](../frozen-vla-selector-20261124-r1/rollout.py) wrap the same motor loop for a three-way menu at one boundary and record decisions at the simulator handoff. This harness keeps its conventions (`sim.call('create'|'chunk')`, `policy.infer(item, noise=...)`, `ur5e_motor.run_episode`, the layout 11 calibration, reset and opening-proposal matching, paired layout bootstrap) and replaces the menu with the seven-hook `Adapter`.

## Protocol

1. `harness.py`: `Adapter`, `collect_support`, `rollout(sim, policy, support, case, adapter)`. Adapters act from the first replan of the query episode. Hooks receive copies of public observations only. The untreated opening proposal is always computed and recorded so branches can be matched even when an adapter changes the first replan.
2. `panel.py`: registered starts × discovered adapters plus `raw` and `repeat`, reset and opening-proposal verification, rescues, losses, control counts and paired layout bootstrap intervals. Locked starts require named adapters.
3. `fixture.py`, `test_harness.py`: deterministic simulator, policy and packer. Rewards in it are synthetic.
4. `run.sh`: pin check, GPU probe, one RTX 4090, `panel.py`. Not run here: this VM has no GPU and no Baige session.

Evidence delivered by this lane is CPU fixture only. No native reward.
