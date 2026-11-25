# Repository guidance

This repository studies few-shot adaptation of frozen robot policies. LIBERO-Pro2.0
is the primary transfer benchmark. Use native rewards in simulation. Start with
the [documentation index](docs/README.md). Keep LycoReco work and runs pinned to
`fluffy-bassoon` commits in that repository. [Provenance](PROVENANCE.md)

## Research factory

- Give each independent research question or hypothesis its own workstream.
  Check the [coordination board](outputs/coordination.md) for overlapping work
  and shared compute before assigning workers. Review prior work before claiming novelty.
- Prioritize falsifiable hypotheses whose answers would change a research decision.
  State what observation would count against each hypothesis. Start with the
  least expensive experiment that distinguishes competing explanations.
- Gate work on the research question: is the concept sound, is the proposed
  method a reasonable answer to it, and does controlled evidence show it is
  effective? Do not invent numeric pass/fail thresholds as go/no-go gates; a
  number chosen without a derivation carries no burden of proof. Use a number
  as a gate only when it follows from a validity constraint or from the
  comparison that decides the question (a matched baseline, an oracle ceiling,
  a noise floor), and say which. Report measured effects with their
  uncertainty and let the decision follow from what they show.
- For confirmatory policy evaluations, register the primary paired native outcome,
  independent sampling unit, uncertainty method, and stopping rule before scoring.
  Continue on fresh units until the registered test resolves benefit or harm, or
  the authorized budget or eligible population is exhausted. Account for repeated
  looks and model comparisons. Report an unresolved result with its interval.
- As orchestrator, delegate investigations, implementation, experiments and critical
  reviews through the existing worker CLI (locally, `codex exec`). Parallelize
  independent tasks. Give each worker the question, context, deliverable, resource
  budget and stop condition, then let it work independently within that scope.
- Track hypotheses, owners, progress and open questions on the coordination board.
  Before a worker exits, save its evidence and conclusions in the run directory.
  Synthesize results across workers. For gains and null results alike, explain
  what the controls establish and which alternative explanations remain.
- Propose follow-up hypotheses from the evidence. Pursue, reprioritize or stop
  work according to what it can teach for the time and compute it costs.

## Evidence and records

- Give each run a fresh `outputs/<run-id>/`. Keep its question, plan, exact code
  revision, configs, seeds, metrics, logs and result together. An exploratory
  run needs only a short plan and result note alongside its evidence. Reserve
  full audits and hash readbacks for confirmatory results.
- Record verified findings, including negative results, with evidence links in
  [FINDINGS.md](docs/FINDINGS.md). Keep `docs/` to its five core documents,
  `literature/` and `robotwin/`. Keep this guide short and durable. Put dated
  material in run directories and apply the [retention policy](README.md#artifact-retention)
  when work closes.
- Keep tests for core behavior and numerical agreement with a reference. Check
  one-run plans, launch scripts and bookkeeping in the run itself. Tie each
  added test or pass/fail threshold to a concrete failure, validity constraint
  or decision.

## Development and deployment

- Use `uv` for Python environments. Edit and run static checks on macOS. Run
  tests and model computation on allocated Linux resources.
- Commit routine changes directly to `main` in the main clone. Isolate potentially
  conflicting worker edits in separate worktrees. Use feature branches only for
  major changes. Rebase on current `main` before starting work, merge promptly
  after tests pass and delete merged branches locally and on `origin`. Push
  coordination updates directly to `main`, editing only your own row.
- Review and validate current `main` for new runs. Deploy the verified commit
  through Git and check it on each host. Use detached worktrees for pinned runs.
  Keep active worktrees and contracts immutable and record any later migration.
  Preserve local changes when reconciling divergent checkouts.
- Use Git directly where available. On Macs requiring a proxy, run `proxy_on`
  or `proxy_on_clash` in the same shell as Git. If remote Git access fails,
  transfer a Git bundle via `scp`. Never use browser/WebIDE file upload to sync
  with servers; keep code in Git and sync artifacts directly over SSH (`rsync`
  or `scp`) using the local `~/.ssh/config` host aliases (e.g. `baige-dev` for
  `mwj-dangerous-dogs`).

## Compute and storage

- Default to Baige for compute, launches, availability and job status, and use the installed official CLIs directly instead of browser consoles: **`aihc`** (`/opt/homebrew/bin/aihc`, `~/.aihc/config` mode `600`, pool `aihc-serverless`, queue `aihcq-rh0wwk9muei4`, region `cn-bj`) for Baige, and **`pai`** (`/opt/homebrew/bin/pai`, `~/.pai/` mode `600`, region `cn-wulanchabu`, workspace `268463`, quota `DM05` / `quota1wyoh6y93pj`) for Ali PAI Wulanchabu PPUs. Keep browser consoles as fallback only when a CLI operation is unavailable or the user requests it.
- **CLI workflow:**
  - Baige (`aihc`): inspect with `aihc job list`, `aihc job get <job-id>`, `aihc job logs <job-id> --podname <job-id>-master-0`, and `aihc queue list`; submit via `aihc job export <template-job-id>` → edit `name`, `jobSpec.command`, `jobSpec.resources` (clear `faultTolerance: false` and `faultToleranceArgs: ""` on exported JSON and remove the temporary `workspace/aihc/` export dir) → `aihc job create -f <job.json>`; stop only your own jobs with `aihc job stop <job-id>`.
  - Ali PAI (`pai`): check live capacity with `pai resource quota get quota1wyoh6y93pj -o json` (`DM05` `PPU-ZW810E` specs: `1/2/4/8/16GPU` up to `16GPU/184C/1800GM`); inspect with `pai dlc job list/get/logs/wait`; submit via `pai dlc job resubmit <template-job-id> --export > /tmp/job.json` → edit `DisplayName`/`JobSpecs`/`UserCommand` → `pai dlc job submit --body-json @/tmp/job.json`; stop only your own jobs with `pai dlc job stop <job-id>`.
- Check queue capacity, limits and actual accelerator type before scheduling. Keep
  GPU isolation at the exclusive scheduler container. Avoid nested device
  namespaces and bubblewrap so the job retains scheduler GPU visibility.
- Submit full GPU training, rollout collection and benchmark evaluation on
  RTX 4090 or RTX PRO 6000D through Baige **分布式训练** (Distributed
  Training, via `aihc job create`), including single-GPU runs. Use **开发机** (development machine)
  GPUs exclusively for short smoke tests and runtime qualification.
- Launch verified, in-scope work once resources are ready. Allocations or
  incremental budgets up to 50 GPU-hours need no further approval. Track
  cumulative use across workers. Ask before an unapproved budget above 50
  GPU-hours, a scope expansion or exceeding an explicit user cap. CPU tests
  and static checks need no approval.
- Use project-specific shared NVMe for code, environments, caches and outputs.
  Check capacity and quota and verify a write before setup. Protect data,
  environments, caches and checkpoints that active jobs depend on.
- Keep SSH configuration, credentials and secret-bearing logs out of Git.
  Prefer direct downloads. Keep private credentials off public mirrors and use
  the user's Hugging Face authentication for gated weights.

## Accelerator jobs

- Run `scripts/[train]gpu_probe.py` inside the GPU job step with `--expected-count`
  and `--output`. For evaluation alone on an unchanged, qualified Baige runtime,
  a GPU-count/CUDA check and one real simulator episode suffice. Bind inference
  and rendering to allocated GPUs. Set job-local `NCCL_P2P_DISABLE=1` for RTX
  6000D distributed work.
- On Ali PPU, preserve vendor libraries and qualify framework support and core
  numerical behavior on the allocated PPU before model work.
- Measure batch capacity and update time. When tuning execution batches,
  preserve the saved global batch, loss weighting and optimizer schedule.
- Verify advancing updates or completed artifacts after launch, then check every
  20 minutes. Report cluster, job ID, progress, last check and next action.
  Diagnose failures before retrying under a fresh run ID.
- Before stopping or moving work, verify ownership and the exact job step.
  Stop only your own jobs. Preserve a resumable checkpoint with model,
  optimizer, sampler and RNG state.
