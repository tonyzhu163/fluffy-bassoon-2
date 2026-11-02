# Match the reward to native first success

The goal remains a rubric repair on small retained-video samples. The short
revision1.6 rejects diagnosed wrong-drawer112 but loses terminal full-score
controls. This does not justify promotion. Test whether that tradeoff persists
at first success, where stopping would avoid later occlusion or undoing.

Keep exactly the same six retained trajectories:112,64 and success controls
128,72,264,144. No new motor inference or sampled trajectories. Prior CPU
replays qualified112/64/144 against all saved scene metadata and all15 saved
public16D robot-state boundaries (zero error). Recover saved actions and replay
only128/72/264 now, on allocated Baige dev CPU with CUDA hidden. Limit750
retained controls per trajectory. These are six distinct retained trajectories
in total, within the original diagnosis ceiling;128 had one earlier constructor
attempt, excluded before any controls because of semantic-tag ordering.

The new qualification explicitly canonicalizes only the unordered
`groups_containing_sampled_obj` tag list before metadata comparison. It keeps
all other metadata, all15 state boundaries, terminal labels and1e-4 numerical
bound unchanged. Do not modify or relabel the earlier excluded replay.
Retain the lack of saved full MuJoCo initial state as a limitation.

If the three new replays qualify, freeze a causal-prefix bank before inference:
112 at every50-control boundary through750; the other five trajectories at
floor(first_success/50)*50, ceil(first_success/50)*50 the next50-control boundary after the first post-crossing probe, and750, deduplicated and
clipped to positive controls. For exact-grid crossings use the preceding50
boundary as the negative candidate. Use actual current and ever-success
predicates offline, including any before-threshold success on a previous visit.
Maximum35 prefixes on six old videos. No future frame may enter a prefix.

Compare verbatim1.1 and short1.6, at most70 official graded outputs. All native
outcomes, first-success times and fixture identities stay out of prompts.
At most one96GiB GPU,32CPU/128GiB RAM/16GiB SHM,900GPU-s; no production reward
or PPO change. Require a reduction in pre-success/wrong-target full scores
without losing first-success acceptance. This is development evidence, not
independent precision or timing qualification. No full634-video bank rescore.

Before inference, add one post-crossing50-control probe per successful
trajectory so a crossing just before a chunk boundary does not falsely appear
to be a missed success because no perceptual delay was allowed. The CPU
replay contract and six-trajectory limit are unchanged. This selection change
precedes all prefix inference; GPU source will record the updated contract.
