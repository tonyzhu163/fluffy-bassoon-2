import asyncio

REPO = "tonyzhu163/fluffy-bassoon-2"
PIN = "757a44cd4374833cad790852df8b87ddf412e638"
LANES = {
    "map": "Can an SVD-bounded residual on a similarity transform generalize to shear and reflections? Compare identity, Procrustes and source-tuned ridge using matched held-out operator or commanded-direction error.",
    "excitation": "Compare fixed-budget one-direction, passive, and two-direction support designs for full-map estimation across weak and strong excitation. Separate physically recorded from constructed supports; match duration, command magnitude and noise.",
    "delay": "Can a causal finite impulse response or joint lag/spatial estimator separate latency from command gains? Use temporally resolved issued commands and realized motion, a zero-delay control, held-out tasks and native vs injected lag labels.",
    "camera": "Test a calibrated or scene-referenced orientation correction on camera roll or orbit, compared with image-only yaw on matched supports. Keep known extrinsics, estimated extrinsics and unknown-view claims distinct. Archived camera perturbation evidence is available.",
    "fusion": "Compare visual-only, kinematic-only, and a confidence-weighted circular fusion on matched real supports under camera perturbations. Qualify identities, define confidence without truth, show when the added sensor is valuable or redundant.",
    "so3": "Does a gain-aware constrained SO(3) action head recover rotations on compound axes more reliably than raw polar and unconstrained solves when rank is poor? Fit controls on source, test fresh draws, use recorded 3D motion if possible.",
    "contact": "Test a causal contact-conditioned expert for transfer after grasp. Use archived cross-arm or friction branch evidence, held-out task evaluation, fixed-program and within-panel oracle comparisons. Avoid postdecision/privileged state in router inputs.",
    "gripper": "Is jaw closure a signed gain, an asymmetric dead zone or a hysteretic response? Fit competing maps from issued gripper command and public finger motion and assess independent held-out episode/time blocks, against sign-only and persistence.",
    "morphology": "Test a morphology-conditioned Jacobian or low-rank response adapter versus shared and arm-specific maps on genuinely different arms, e.g. Panda, Sawyer, UR5e. Verify data splits and total calibration cost. Quantify the limits if traces cannot identify the map.",
    "timing": "Test whether phase-aware chunk execution improves upon fixed default or constant pacing. Use logged frozen-policy alternative branches with measured outcomes; score only actions for which complete counterfactuals were recorded, causal selection, held-out tasks, fixed and oracle baselines.",
    "uncertainty": "Can a calibrated selective reader improve native success by falling back to an already evaluated matched frozen-motor alternative on ambiguous supports? Use paired per-query archival outcomes, inference-time diagnostics and held-out tasks, audit no query leakage; measure coverage and success.",
    "native": "Own live native rollout qualification. Inspect live Baige reachability via legitimate configured route, active board/queue, budget and pins. Launch one independent paired LIBERO-Pro2.0 frozen-motor native comparison advancing an open question (e.g. fresh interface draws or moderate camera orbit, with applicable adapters and oracle), max 12 cumulative GPU-hours, distributed training only. Do not claim historical jobs. If inaccessible, implement and exercise an exact paired rollout adapter/launcher on CPU fixtures with a real qualification error for missing prerequisites; report the precise route or asset needed.",
}
SCHEMA = {"type": "object", "properties": {k: {"type": "string"} for k in
          ("scope", "comparison", "measurements", "limitations", "branch_or_artifact", "jobs")},
          "required": ["scope", "comparison", "measurements", "limitations", "branch_or_artifact", "jobs"]}

async def run_lane(lane):
    resources = "CPU only, 0 GPUh." if lane != "native" else "Exclusive native lane for this workflow. Budget as below."
    prompt = f"""Tony's request: "We want to solve cross embodiment transfer, i need you to start as many agents or jobs as possible in a software factory to start targeted experiments to find the right model architecture to solve this with a FROZEN vla, this is for fluffy-bassoon-2"
Work in {REPO}, baseline {PIN}. Own the {lane} lane. {LANES[lane]}
Read repo AGENTS, coordination board, findings and existing runs first. Run the experiment. Start with evidence committed in the repo, including compressed JSON and NPZ, and use its frozen pi0/pi0.5 provenance. Do not update any motor parameter; adapters only. State accessible inputs, support budget and what would falsify the hypothesis. Measure against paired controls with target labels reserved for scoring, avoiding per-case leakage, and task/episode-level uncertainty. Label any proxy or synthetic result explicitly. If no full benchmark, still execute the strongest controlled screen possible. Do not present it as native task success. {resources} No other agents or jobs on shared GPU allocations. Stop after a reproducible test answers this lane's question or a concrete blocker plus the strongest feasible substitute. Save code, concise plan/result, reproduction command, original revision and compact metrics under outputs/frozen-vla-factory-20261123-r1/{lane}/ only. Use an isolated branch off the pin, push your lane commit to its own devin/ branch, return SHA and branch so the parent integrates a single PR. No merge, separate PR, shared-ledger edit or unrelated artifacts. Report job/receipt, measurements, failure modes and a recommendation. If pushing fails, attach a git format-patch. Do not wait for sibling lanes."""
    try:
        report = await agent(prompt, phase="Experiments", label=lane, schema=SCHEMA, repos=[REPO])
    except WorkflowAgentError as exc:
        report = {"scope": lane, "limitations": str(exc), "jobs": "Agent failed; no verified job"}
    log(f"{lane}: {report}")
    return report

async def main():
    await register_workflow({"name": "Frozen VLA architecture factory", "description": "Independent adapter experiments with paired controls on recorded or live robot tasks", "soft_time_limit_minutes": 35, "phases": [{"title": "Experiments", "detail": "Run each architecture comparison and return its measured evidence", "labels": list(LANES)}]})
    log("Dispatching eleven architecture investigations and one native rollout worker.")
    results = await parallel([lambda lane=lane: run_lane(lane) for lane in LANES])
    log(f"Factory returned {len(results)} lane reports.")
    return results

asyncio.run(main())
