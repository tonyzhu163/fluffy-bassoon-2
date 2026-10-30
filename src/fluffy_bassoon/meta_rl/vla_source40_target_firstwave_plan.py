"""Predeclared fixed-weight first wave across all LIBERO90 and shifted cases.

This module plans metadata only. It imports no model/simulator packages,
reserves no budget, starts no allocation and reads no target trajectories.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


FORMAT = "source40_ad_target_firstwave_plan_v1"
CASE_FORMAT = "source40_ppo_target_case_v1"
PI05_RECEIPT = "ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170"
LIBERO_COMMIT = "87edbd1b823e464bc66557d213cc45e9f88aa543"
ZERO_METHODS = ("native", "ad_full", "ad_static", "dpt_full", "dpt_static")
ONE_METHODS = ("ad_full", "ad_episode_reset", "dpt_full", "dpt_episode_reset")
DEFAULT_OLD = Path("/mowenjie/fluffy-bassoon")
DEFAULT_ROOT = Path("/mowenjie/fluffy-bassoon-2")


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def _literal(node, names):
    """Read the small declarative constants without importing benchmark code."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(value, names) for value in node.elts]
    if isinstance(node, ast.Dict):
        return {_literal(key, names): _literal(value, names) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _literal(node.left, names), _literal(node.right, names)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict" and not node.args:
        if all(keyword.arg is not None for keyword in node.keywords):
            return {keyword.arg: _literal(keyword.value, names) for keyword in node.keywords}
    raise ValueError("Benchmark declaration is no longer a supported literal; review it explicitly")


def declared_constants(path, wanted):
    names = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in wanted:
                names[name] = _literal(node.value, names)
    if set(names) != set(wanted):
        raise ValueError("Pinned benchmark declarations are incomplete")
    return names


def _group(case_id, phase, method, budget, episode_actions):
    return dict(group_id=f"{case_id}/{phase}/{method}/k{budget}", case_id=case_id,
        phase=phase, method=method, support_budget=budget, episodes=budget + 1,
        updates=None if method == "native" else 500,
        source_condition="native" if method == "native" else "source40",
        maximum_actions=(budget + 1) * episode_actions,
        deployment_optimizer_updates=0)


def build_plan(*, worktree, source_commit, simulator_python="/mowenjie/envs/wam-libero-plus-py310/bin/python",
               project_root=DEFAULT_ROOT, old_root=DEFAULT_OLD, workers=6):
    worktree, project_root, old_root = Path(worktree).resolve(), Path(project_root), Path(old_root)
    if (len(source_commit) != 40 or any(char not in "0123456789abcdef" for char in source_commit)
            or workers != 6 or not Path(simulator_python).is_absolute()):
        raise ValueError("Require exact source SHA, six workers and an explicit simulator Python path")
    module_root = Path(__file__).resolve().parent
    plus_path, pro_path = module_root / "vla_plus_worker.py", module_root / "vla_libero_pro.py"
    plus = declared_constants(plus_path, ("PLUS_COMMIT", "BASE_NAME", "CASES"))
    pro = declared_constants(pro_path, ("PRO_COMMIT", "DATA_REVISION", "TASK_STEM", "CASES", "SETTLE_STEPS"))
    if set(plus["CASES"]) != {667, 290} or set(pro["CASES"]) != {"swap", "appearance"} or pro["SETTLE_STEPS"] != 5:
        raise ValueError("First wave requires the two predeclared Plus and Pro variants")
    libero = project_root / "deps/libero-dsrl-87edbd1-20261029-r1/repo"
    plus_root = old_root / "third_party/libero-plus-4976dc3-r3"
    pro_root = project_root / "deps/libero-pro-source-20261029-r1/repo"
    pro_data = project_root / "inputs/libero-pro-dev-c86fc3b8-20261029-r1"
    client = old_root / "third_party/openpi-215abfb/packages/openpi-client/src"
    deps = old_root / "envs/pi05-plus-client-deps"
    cases = []

    def add(benchmark, case_id, suite, task_id, state, seed, *, root, module, arguments,
            horizon, settle, expected, declared=None, phase_b=False, physical_case_kind):
        target = dict(benchmark=benchmark, case_id=case_id, suite=suite,
            task_id=task_id, initial_state=state, scene_seed=seed)
        groups = [_group(case_id, "A", method, 0, horizon + settle) for method in ZERO_METHODS]
        if phase_b:
            groups += [_group(case_id, "B", method, 1, horizon + settle) for method in ONE_METHODS]
        cases.append(dict(format=CASE_FORMAT, case=target,
            simulator=dict(python=str(simulator_python),
                pythonpath=":".join(map(str, (worktree / "src", root, client, deps))),
                module=module, arguments=arguments, horizon=horizon, settle_steps=settle,
                expected_metadata=dict(suite=suite, task_id=task_id, horizon=horizon,
                    settle_steps=settle, **expected)),
            declaration=declared, physical_case_kind=physical_case_kind, groups=groups,
            zero_shot_reuse={"ad_episode_reset": groups[1]["group_id"],
                "dpt_episode_reset": groups[3]["group_id"]},
            nominal_actions=sum(group["maximum_actions"] for group in groups),
            nominal_episodes=sum(group["episodes"] for group in groups),
            qualification_settling_allowance=0,
            reporting_scope=('Full LIBERO90 at declared H400' if benchmark == 'libero90' else
                'Official perturbation case under the declared280-step development cutoff; not a full benchmark score')))

    for task in range(90):
        add("libero90", f"libero90:{task:02d}/state-0/seed-123", "libero_90", task, 0, 123,
            root=libero, module="fluffy_bassoon.meta_rl.vla_libero",
            arguments=["--mode", "env-server", "--suite", "libero_90", "--task-id", str(task),
                "--horizon", "400", "--fresh-environment-on-reset"],
            horizon=400, settle=10, expected=dict(libero_commit=LIBERO_COMMIT, fresh_environment_on_reset=True),
            phase_b=task % 10 == 0, physical_case_kind="one nominal initialization for this LIBERO90 goal")
    for variant in (667, 290):
        for state in (0, 1):
            add("plus", f"plus:libero_object:{variant}/state-{state}/seed-123", "libero_object", variant, state, 123,
                root=plus_root, module="fluffy_bassoon.meta_rl.vla_plus_worker",
                arguments=["--source-commit", source_commit, "--libero-root", str(plus_root),
                    "--case-index", str(variant), "--horizon", "280", "--settle-steps", "10"],
                horizon=280, settle=10, expected=dict(benchmark_kind="plus", libero_commit=plus["PLUS_COMMIT"],
                    fresh_environment_on_reset=True), declared=plus["CASES"][variant], phase_b=True,
                physical_case_kind="official perturbation variant with distinct nominal initialization ID")
    for variant in ("swap", "appearance"):
        spec = pro["CASES"][variant]
        for seed in (123, 124):
            add("pro", f"{spec['case_id']}/seed-{seed}", spec["family"], 1, 0, seed,
                root=pro_root, module="fluffy_bassoon.meta_rl.vla_libero_pro",
                arguments=["--pro-root", str(pro_root), "--data-root", str(pro_data),
                    "--case", variant, "--horizon", "280"],
                horizon=280, settle=5, expected=dict(benchmark_kind="pro", pro_commit=pro["PRO_COMMIT"],
                    dataset_revision=pro["DATA_REVISION"], development_initial_state_indices=[0]),
                declared=spec, phase_b=True,
                physical_case_kind="seeded repeat of state0 in one of two Pro variants; not a new task or nominal state")
    groups = [dict(group, case_index=index) for index, case in enumerate(cases) for group in case["groups"]]
    worker_rows = []
    for index in range(workers):
        assigned = cases[index::workers]
        nominal = sum(case["nominal_actions"] for case in assigned)
        worker_rows.append(dict(worker=index, case_ids=[case["case"]["case_id"] for case in assigned],
            nominal_actions=nominal, action_cap=nominal + 1000, gpu_seconds=3600,
            qualification_settling_allowance=sum(case["qualification_settling_allowance"] for case in assigned)))
    contract = dict(source_commit=source_commit, worktree=str(worktree), source_only_pretraining=True,
        source_data_from_targets=False, frozen_motor_receipt_sha256=PI05_RECEIPT,
        policy_selection="Four fixed-update500 source40 AD/DPT full/static exports; actual matrix/assembly hashes bound by runner",
        deployment_optimizer_updates=0, phase_a=dict(support_budget=0, methods=list(ZERO_METHODS),
            task_selection="All LIBERO90 IDs0..89; both declared Plus variants×state0/1; both Pro variants×seed123/124 at state0"),
        phase_b=dict(support_budget=1, methods=list(ONE_METHODS),
            task_selection="LIBERO90 IDs0,10,...80 plus every shifted case; fixed before outcomes"),
        zero_shot_reuse="Episode-reset controller K0 is the same full-controller K0; reference it without another episode",
        ordinary_dsrl_and_ppo="Not evaluated by this first wave; required separately for the full method comparison",
        maximum_total_actions=250000, maximum_gpu_seconds=21600,
        qualification='Use the first planned reset, with zero additional physical resets/actions',
        benchmark_declarations=dict(plus_file_sha256=hashlib.sha256(plus_path.read_bytes()).hexdigest(),
            pro_file_sha256=hashlib.sha256(pro_path.read_bytes()).hexdigest()),
        restrictions=["No semantic filtering or outcome-dependent task selection", "No target data added to source pretraining",
            "Report full90 coverage separately from the eight shifted development cases",
            "Pro seeds are repeats of state0 across two variants, not four distinct tasks",
            "Source dataset exposure and prior target-development use must remain disclosed"])
    plan = dict(format=FORMAT, contract=contract, cases=cases, groups=groups, workers=worker_rows,
        counts=dict(libero90_cases=90, plus_cases=4, pro_seeded_cases=4, total_cases=len(cases),
            phase_a_groups=sum(group["phase"] == "A" for group in groups),
            phase_b_groups=sum(group["phase"] == "B" for group in groups),
            physical_episodes=sum(group["episodes"] for group in groups),
            nominal_actions=sum(group["maximum_actions"] for group in groups),
            worker_action_caps=sum(worker["action_cap"] for worker in worker_rows),
            worker_gpu_seconds=sum(worker["gpu_seconds"] for worker in worker_rows)))
    if (plan["counts"]["physical_episodes"] != 626 or plan["counts"]["nominal_actions"] != 243920
            or plan["counts"]["worker_action_caps"] != 249920
            or len({case["case"]["case_id"] for case in cases}) != 98):
        raise ValueError("First-wave episode/case/action accounting differs from its declared matrix")
    plan["firstwave_sha256"] = content_hash(plan)
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulator-python", default="/mowenjie/envs/wam-libero-plus-py310/bin/python")
    args = parser.parse_args(argv)
    plan = build_plan(worktree=args.worktree, source_commit=args.source_commit,
        simulator_python=args.simulator_python)
    with args.output.open("x") as stream:
        stream.write(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(firstwave_sha256=plan["firstwave_sha256"], counts=plan["counts"])))


if __name__ == "__main__":
    main()
