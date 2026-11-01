# Learned steering on LIBERO and RoboTwin

Updated **2026-10-29 23:20 Asia/Shanghai**. The user requested ongoing
monitoring and continuation of both campaigns. Their existing execution
owners and20-minute monitors remain active; no duplicate allocation is
submitted from this chat. Meta-World remains stopped and its monitor paused.

## Current budget

The human confirmed **1,000,000 executed simulator actions per benchmark**,
shared across SAC and fresh PPO collection, training/validation, settling and
retries. This is up to2M across LIBERO and RoboTwin, not1M per learner.
Older10.24M-per-learner and larger collection proposals are superseded for
future launches; original run contracts and evidence remain unchanged.
All attempts and qualifications must be charged to the shared ledger.
Material GPU-hour expansion still needs a measured resource proposal under
AGENTS.md. The action allowance does not establish data sufficiency.

The human authorized coordinating this cap and the fresh PPO requirement
with `Clarify DSLR online RL post-training`. A confirmation was sent to that
owner, and both monitors now carry the1M cap. PPO needs its own current-policy
rollouts with behavior log probabilities, values, rewards and episode/trial
boundaries; SAC histories cannot simply be relabeled PPO data.

## LIBERO

`Plan VLA in-context learning` owns the source40 campaign. The first capped
stream is now actually collecting on Baige **job-p5we8pk45d8i**, output
`libero40-1m-sac-20261029-s00-train-r1`, exact source
`8ada0c0a59ea562716b251dc3cd7d71dc1ba3b18`. Its hard limit is9,500 actions
including settling, within a reserved task slot in the shared campaign ledger.

The [23:20 saved-artifact receipt](libero-capped-stream-2320.json) confirms
**40 episodes /10 complete cases,3,658 primitive+410 settling actions,383
query chunks/replay rows**, with a saved recovery. Success is39/40. Optimizer
updates remain0 because this new independent teacher is still below its
500-chunk warmup. This is real collection progress, not yet learned behavior
from this particular stream. Its allocation preflight passed; the owner
reports191 focused Linux tests passed before startup.

Earlier training and independent validation qualifications exited0 with
60/68episodes and2,220/2,400SAC updates respectively. Their near-ceiling
success did not establish useful recovery or adaptation. The r3 history
integration check also exited0, assembled paired train/validation AD/DPT data,
restored SAC state/RNG and trained four full/static AD/DPT plumbing controls
for25updates each. These are integration checks, not live robot evaluation or
meta-learning results. The next GPU update matched numerically within the
recorded tolerance, without a bitwise continuation claim.

Next: verify postwarmup optimizer updates and final accounting/exports for
the capped stream, then continue all40 source coverage and genuine PPO
collection under the shared ledger. Training uses nominal states0–39;
independent local validation uses40–49. Full90/Plus/Pro remain outside source
pretraining; older semantic filtering is archival. The official40-task OpenPI
motor stays separate from the old RLinf130 campaign.

## RoboTwin

`Clarify DSLR online RL post-training` owns the six-source/six-controller-held-out
campaign. Missing assets were repaired and all12task checks passed. The
single-thread scene lifecycle fix then allowed **r4 job-tgjz72b93hes** to end
cleanly at22:58:56, with32episodes/11,475executed source actions. All optimizer
counters are0: SAC has139replay rows and PPO still has five pending K4 trials
across its persistent/reset arms. The frozen motor and four complete recovery
payloads were audited and mirrored by the owner. R4 did not save raw RGB, so
it is not a complete reusable offline corpus.

The capture optimization gate **job-lspkxidy3haj**, source
`8114dd23e872d673ade6ea391b973c05a0dc6eb1`, exited0. Independent static
comparison of its closed JSON traces confirms both reference repetitions and
the candidate match exactly in each of two tested seeds, including camera,
state/feature hashes, noises, actions, rewards, counts and terminal flags.
One episode timed out at400actions and one succeeded at117. The six episodes
cost **1,551 actions**, charged to the budget but carrying no learning credit.
The candidate is **2.1785× faster** than the warmed reference in these two
cases of `place_phone_stand`; this is not an all-task throughput guarantee.
[Independent audit and input hashes](checks/20261029-2316-capture/independent-audit.json).

Next: integrate and validate the action ledger and raw observation/action
recording, explicitly document the smaller PPO batch needed for the1M pilot,
then propose the measured continuation compute before a larger run. Preserve
all authentic pending rollouts and recoveries through any documented contract
migration. The frozen Motus motor has seen all50RoboTwin tasks, so the transfer
claim concerns the controller. No learned-steering or memory benefit is yet
established by RoboTwin.

## Monitoring

`continue-frozen-pi05-libero-poc` and `robotwin-matched-steering-monitor` are
active every20minutes in their execution-owner chats. They check actual
rollouts and optimizer progress, losses, budget consumption, checkpoint/RNG
recovery and terminal artifacts, and continue authorized work between checks.
Notify meaningful results, failure, completion or a concrete decision; stay
quiet for unchanged/healthy progress. `monitor-metaworld-pi05-noise-screen`
remains PAUSED.
