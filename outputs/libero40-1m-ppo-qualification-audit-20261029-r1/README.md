# Fresh PPO single-task runtime

Baige `job-rg68oj4uy6mv`, source `0f35d17a3a604553fb56612fe4eb10ebca8313de`,
exited0. Eight K4 trials/32 episodes, all successful;289 query chunks,
2,777 primitive+330 settling=3,107 actual actions. Eight PPO iterations,
32 optimizer steps, and behavior-policy likelihood replay error0 throughout.
198 focused Linux tests passed/two probe-environment JAX skips.

Raw data independently inspected: current/next public observations,
4104D frozen features, actual noise/decoded actions/rewards/durations and
explicit terminal/timeout boundaries. Pre-tanh samples, behavior likelihoods,
values and recurrent state accompany every episode. Eight immutable behavior
policies and the full policy/optimizer/sampler/motor RNG recovery are saved.
See [trace receipt](trace-receipt.json), [metrics](metrics.jsonl) and
[final index](index.json).

Initial learning rate3e-4/four epochs caused first-update approximate KL13.76
and clip fraction.743; final KL.230/clip.614. Finite losses and valid saved
data establish execution, not stable learning or better robot performance.
This single-task check is not shared40 meta-training. Its immutable contract
was retained; a new shared pilot separately qualified1e-4/two epochs.
