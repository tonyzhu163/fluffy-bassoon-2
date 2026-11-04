# Critic warmup reference check — Nov 4, 2026

The checked released recipes use **zero critic-only optimizer steps**:

| Recipe | `actor.optim.critic_warmup_steps` |
| --- | ---: |
| StructRL pi0.5 RoboCasa365 Composite-Seen shared PPO YAML | 0 |
| RLinf pi0.5 LIBERO-10 PPO | 0 |
| RLinf pi0 RoboCasa365 OpenDrawer PPO | 0 |

The StructRL sparse PPO launcher selects that shared YAML and does not
override critic warmup. Its one-epoch setting and VLM-prefix value head do
not require a critic-only phase in the released recipe. This is evidence
of their configuration, not an ablation proving warmup cannot help ours.

Sources checked directly:

- [StructRL shared PPO config, commit6caba1f9](https://github.com/amazon-science/StructRL/blob/6caba1f9538167e0752a7338663e45aac8df965d/configs/robocasa365_ppo_openpi_pi05_composite_seen.yaml)
- [StructRL sparse PPO launcher at the same commit](https://github.com/amazon-science/StructRL/blob/6caba1f9538167e0752a7338663e45aac8df965d/launch/train_pi05_sparse_rl_ppo.sbatch)
- [RLinf pi0.5 LIBERO-10 PPO](https://github.com/RLinf/RLinf/blob/main/examples/embodiment/config/libero_10_ppo_openpi_pi05.yaml)
- [RLinf pi0 OpenDrawer PPO](https://github.com/RLinf/RLinf/blob/main/examples/embodiment/config/robocasa365_opendrawer_ppo_openpi.yaml)

Both our historical PPO and completed Stage1 PPO also resolved warmup to0.
The optional warmup transition therefore cannot explain their results.
The proposed100-step warmup was unvalidated and was not taken from these
reference configs. It should not be presented as a required repair.

The local untested optimizer-rewrite draft was removed after this check;
it was never deployed or used by a GPU run. Keep default warmup0 for the
reference-informed recipe implementation. Any warmup treatment is a separate
explicit ablation requiring numerical/checkpoint qualification first.
