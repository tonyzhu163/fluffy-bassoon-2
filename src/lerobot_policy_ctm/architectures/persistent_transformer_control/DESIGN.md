# Looped Transformer action expert

Only `action_expert.py` was carried over from fluffy-bassoon, where this
package is a full LeRobot policy. Here it is the looped (house) Transformer
learner used as a matched comparison in `fluffy_bassoon.meta_rl.ppo`. It builds
on `recurrent_transformer_control/action_expert.py` and
`shared/transformer_expert.py`. The policy wrappers, configurations and their
evaluation records stayed in fluffy-bassoon (`0df7d026cb72d1d047bbb261ef95dc1f5d2da9fa`).
