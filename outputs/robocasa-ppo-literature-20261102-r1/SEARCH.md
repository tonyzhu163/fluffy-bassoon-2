# RoboCasa PPO literature search record

November 2–3, 2026. Deliverable:
[literature and recipe review](../../docs/literature/robocasa-ppo-review.md).
Scope: online PPO on RoboCasa, demonstrated performance, released recipes,
and implications for this repository's π0.5 positive control and frozen
velocity/KV interface. Other RoboCasa RL algorithms are comparison evidence;
other simulators and reward/video-model-only results are kept separate.

## Discovery and screening

Web queries combined RoboCasa / RoboCasa365 with PPO, GRPO, reinforcement
learning, fine-tuning, flow matching, π0 / π0.5, GR00T, residual policy,
critic, reward shaping, action chunks, and released GitHub recipes.
Additional queries targeted individual candidate titles, author projects,
model cards and RLinf issues. English queries included `RoboCasa PPO
training success rate`, `RoboCasa365 PPO pi0.5`, and repository-scoped
recipe searches; Chinese discovery included RoboCasa with `PPO`, `训练`
and `成功率`. This is a record of query families and screened sources,
not a verbatim transcript of every web query or a preregistered protocol.

Two arXiv API metadata searches were saved:

1. `all:robocasa AND (all:reinforcement OR all:PPO OR all:GRPO)`:
   eight candidates; [query URL and parsed metadata](arxiv-search.json).
2. `(all:robocasa OR all:robocasa365) AND (all:reinforcement OR all:PPO OR all:GRPO OR all:RL)`:
   fifteen candidates; [query URL and metadata](arxiv-search-expanded.json).

These searches inspect metadata, not the entire paper corpus. They missed
relevant papers such as SEES and Simple Recipe Works; web discovery,
full-text inspection and reference chaining supplied those. No claim of
complete database coverage, independent double screening or PRISMA compliance
is made. Search-result summaries were leads, not quantitative evidence.
The saved metadata omits abstracts; full source texts are not needed to
retain the candidate list and exact query URLs.

## Candidates from the expanded metadata query

| arXiv source | Disposition |
| --- | --- |
| [RynnUnify 2610.32965v1](https://arxiv.org/abs/2610.32965v1) | Agentic RL/world-action extensions; no ordinary RoboCasa PPO recipe established. |
| [VaDar 2610.26629v1](https://arxiv.org/html/2610.26629v1) | Include as direct off-policy OGPO/critic evidence; non-VLA BC initialization and fixed initial-state conditions. PPO-style policy extraction is not conventional on-policy PPO. |
| [LoopWAM 2610.18873v2](https://arxiv.org/html/2610.18873v2) | Deployment-rollout self-training; not an inspected PPO recipe. |
| [Fewer Tokens, Better Action 2610.01939v2](https://arxiv.org/abs/2610.01939v2) | Agent/code execution; GRPO training uses LIBERO-PRO rather than a RoboCasa VLA PPO experiment. |
| [FutureWorlds 2610.01019v1](https://arxiv.org/abs/2610.01019v1) | RL for world-model prediction; exclude from robot-policy success evidence. |
| [GroundingPI 2609.39601v2](https://arxiv.org/abs/2609.39601v2) | Grounding/SFT/GRPO discovery; no direct RoboCasa robot PPO recipe established. |
| [StructRL 2609.36352v1](https://arxiv.org/html/2609.36352v1) | Include direct PPO, reported sparse/structured comparisons and released launchers/patches. |
| [PolicyWeave 2609.33125v1](https://arxiv.org/html/2609.33125v1) | Include direct expert-LoRA PPO and post-merge performance. |
| [MiDAS 2608.11363v3](https://arxiv.org/html/2608.11363v3) | Include frozen-base residual RL; off-policy rather than PPO. Use latest revised results/title, not only cached v1. |
| [OopsieVerse 2606.31993v1](https://arxiv.org/html/2606.31993v1) | Adjacent safety PPO/latent control; modified rewards/tasks and multiple simulation backends. |
| [Z-1 2606.31846v1](https://arxiv.org/html/2606.31846v1) | Include direct π0.5 RoboCasa GRPO; distinguish scene-specific SFT and unequal evaluation counts. |
| [PearlVLA 2606.17924v2](https://arxiv.org/html/2606.17924v2) | Planning/refinement rather than an inspected online PPO recipe. |
| [LLM-Guided Future Hypotheses 2605.29864v2](https://arxiv.org/html/2605.29864v2) | RoboCasa BC+RL results; no identifiable PPO implementation recipe established. |
| [Adaptive Q-Chunking 2605.05544v1](https://arxiv.org/html/2605.05544v1) | Include direct offline-to-online GR1 adaptation/timing evidence; different embodiment and algorithm. |
| [DEAS 2510.07730v1](https://arxiv.org/abs/2510.07730v1) | Offline action sampling; not online RoboCasa PPO. |

## Additional primary sources

| Source | Role in the review |
| --- | --- |
| [SEES 2609.32698v1](https://arxiv.org/html/2609.32698v1) | Direct RoboCasa hierarchical bottleneck-skill PPO, sparse predicates, failure-state resets. |
| [Simple Recipe Works 2603.11653](https://arxiv.org/html/2603.11653) | Direct RoboCasa continual RL; appendix identifies GRPO, released tree lacks a RoboCasa training config. |
| [DRILL 2610.30814v1](https://arxiv.org/html/2610.30814v1) | Direct Sim-PPO comparison with recent-episode replay; limited standalone recipe comparability. |
| [πRL 2510.25889](https://arxiv.org/html/2510.25889) | Flow-SDE and π0.5 critic design; other-simulator results. |
| [RLinf-VLA 2510.06710](https://arxiv.org/abs/2510.06710) | Framework/practice evidence, not proof of a RoboCasa performance recipe. |
| [EBRL 2609.27547v1](https://arxiv.org/html/2609.27547v1) | RoboCasa execution throughput; displayed success convergence comparison is a different benchmark. |
| [ReinFlow 2505.22094](https://arxiv.org/abs/2505.22094) | Stochastic flow-policy likelihood and RL; original visual benchmark RoboMimic. |
| [PolicyTrim 2606.22540v1](https://arxiv.org/html/2606.22540v1) | Execution cadence and GRPO comparison mechanisms; original evaluation is not RoboCasa. |
| [Robo-Dopamine 2512.23703](https://arxiv.org/html/2512.23703) | Reward/process supervision; distinguish RoboCasa reward data from policy-learning benchmarks. |
| [VLRLA 2610.19517v1](https://arxiv.org/html/2610.19517v1) | Teacher PPO and frozen KV bridge; RoboTwin and supervised bridge fitting. |
| [Frozen π0.5 soft prompts 2609.11310](https://arxiv.org/abs/2609.11310) | RoboCasa placement and overfitting evidence; demonstration fitting, not PPO. |
| [RoboCasa-RL](https://github.com/micmia/RoboCasa-RL) | State MLP PPO with shaped rewards; integration/code evidence with a different learning problem. |
| [Official RoboCasa benchmark docs](https://robocasa.ai/docs/build/html/benchmarking/benchmarking_overview.html) | Version, task/split/evaluator and embodiment distinctions. |
| [RLinf legacy recipe](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robocasa.html), [RoboCasa365 recipe](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robocasa365.html) | Framework configs and performance-evidence limits. |
| [RLinf #1044](https://github.com/RLinf/RLinf/issues/1044), [#1034](https://github.com/RLinf/RLinf/issues/1034) | User failure reports; generic untested suggestions are not successful recipes. |

## Released-code inspection

[REPOSITORIES.json](REPOSITORIES.json) records exact Git commits and dates
for RLinf, StructRL, continual-vla-rl, PolicyTrim and RoboCasa-RL.
Shallow no-checkout clones and selected blobs were inspected read-only.
No external training program or downloaded checkpoint was executed.

For StructRL, inspection covered `docs/INSTALL.md`, `docs/REPRODUCE.md`,
the π0.5 model and PPO configs, sparse/structured launchers and
`patches/rlinf_core.patch`. Material findings: launch-time overrides of
LR/critic/chunk/flow settings; paper H10 vs launcher H50; explicit upstream
pins; target-SFT requirement; unseeded evaluation resets; done-shift patch
depending on a different collector convention. None was silently adopted.

The SEES [author-linked model card](https://huggingface.co/huangtianyu0815/robocasa-n15-subtask-termhead/blob/main/README.md)
points to `MuQoe/vla-self-evolution`, branch `tianyu`. Anonymous Git fetching
failed for that link. This limits this audit's reproducibility evidence;
it does not establish that no release exists. PolicyWeave, Z-1 and MiDAS
primary pages did not establish a verified runnable public recipe in this
inspection. Search snippets/forks were not substituted for author releases.

[fetched-papers.json](fetched-papers.json) records hashes of the selected
HTML downloads cached outside the repository. Other papers were inspected
with the web reader, so that file is not a complete bibliography. In
particular, MiDAS v3 was checked on the web after its v1 download. Full paper
texts and external model binaries are not retained in this evidence folder.

## Numerical follow-up and limits

The termination claim was checked using the deployed native functions on
allocated Linux CPUs with CUDA hidden. [GAE-CHECK.json](GAE-CHECK.json)
records exact agreement with 512 saved returns from one existing behavior
batch, backward success credit and no post-terminal reward leakage in a
successful example. It checks this collector/batch, not every timeout mode
or the full completed training run. No active code or run contract changed.

Reported paper scores remain author results. There was no external training
reproduction, author contact, independent second review, or matched statistical
comparison across papers. This search supports concrete recipe hypotheses
and a reconstruction target; it does not qualify our frozen steering map.
