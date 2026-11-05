# SFT initializer review before RoboTwin PPO

2026-11-05. User asks whether a better SFT checkpoint exists than Motus.
No alternative has been independently evaluated here. Published numbers below
are source claims, not matched comparisons or locally reproduced results.

## Same pi0.5 family

| Candidate | Evidence | Integration / interpretation |
| --- | --- | --- |
| [SidneyXie/pi05_robotwin](https://huggingface.co/SidneyXie/pi05_robotwin) | Independent LeRobot/PyTorch SFT from pi05_base, 50k updates, native H50, three views, 14-D ALOHA joint actions. Card reports 100 Easy episodes each: handover_mic 23%, pick_diverse_bottles 59%, place_a2b_left 49%, place_object_stand 90%, place_phone_stand 66%, scan_object 18%. Six-source mean 50.83% is our arithmetic. | First independent candidate to screen. LeRobot mean/std normalization and absolute-action convention require explicit RLinf conversion/adaptation and parity checks. No claim that it beats Motus on matched cases. |
| [madokalif/pi05-robotwin2-clean50-sft](https://huggingface.co/madokalif/pi05-robotwin2-clean50-sft) | Further full SFT from Motus on 50 clean tasks, 20k steps, global batch64; OpenPI-format 812-key safetensors and own normalization. Card reports training loss, without a task-success table. | A close implementation candidate. More SFT and lower flow loss do not establish better robot success. Native horizon still needs verification from metadata. |
| [Myungkyu/pi0_5_jax_robotwin_vanilla_b128_60k](https://huggingface.co/Myungkyu/pi0_5_jax_robotwin_vanilla_b128_60k) | Independent H50 OpenPI/JAX SFT, 50 tasks x50 clean demos, 60k steps, EMA .99, own quantile normalization. | Clear H50 training recipe, but needs JAX-to-PyTorch numerical conversion qualification for this RLinf runtime. No success table verified in the card. |
| [Motus pi0.5](https://huggingface.co/motus-robotics/pi0.5_robotwin2) | Common 50-task motor; 2,500 clean plus25,000 randomized demonstrations; card specifies `action_horizon=32`. Restored exact effc6e1a weights and normalization. | Retain as the native-H32 reference. Our prepared H50 setting is a migration, not unchanged native behavior. |

## Other published options

- [FlashVLA pi0.5](https://huggingface.co/z-lab/flashvla-pi05-robotwin)
  reports about90% clean/random success. Its
  [repository](https://github.com/z-lab/flashvla) uses streaming decoding and
  executes16 actions per RoboTwin segment. The repository and model-card
  aggregate tables differ slightly. This is a different inference protocol;
  do not rank it directly against Motus or assume RLinf-PPO compatibility.
- [LingBot-VLA RoboTwin SFT](https://huggingface.co/robbyant/lingbot-vla-4b-posttrain-robotwin)
  has [official RLinf support](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/lingbotvla.html),
  including a pinned SFT revision and GRPO recipes. RLinf reports high SFT
  success on click_bell/place_shoe, which are not the six VLRLA source tasks.
  A different backbone is a separate comparison choice, not a proven common
  six-task replacement for pi0.5.

## Recommendation

Screen SidneyXie and the clean50 continuation against native-H32 Motus on
the same source-task scenes, physical horizons and success checker, using
each checkpoint's own transforms and native inference settings. A small
16-case/task panel is a proposed development screen, not a completed result
or a statistically decisive ranking. Three source tasks still require
qualified scene banks. Expand only a promising comparison on independent
source-development cases. Do not select using the six held-out tasks.

Prefer a common motor with source-task competence and recoverable failures
that leaves useful PPO headroom. Then bind the selected checkpoint's native
chunk length, normalization and sampler before the first learning run. Keep
the current PPO launch on hold while resolving this choice.
