"""Native Goal 5 panel preparation and isolated public-observation session bridge."""

from contextlib import closing, contextmanager
import os
from pathlib import Path
import subprocess
import sys
import traceback
from uuid import uuid4

from large_panel_audit import ResetMismatch, audit_pair, capture
from large_panel_protocol import (
    ARMS, GEOMETRY_FIELDS, PHASES, audit_development, bind, directory, implementation,
    lock, manifest, mobile_joints, publish, read, report_for, require, selection, sha, training,
)

HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parent / "libero-paligemma-crossarm-20261120-r1"))


def released():
    import generated_layouts as g
    from libero_remote.reproducibility import canonical_fingerprint, numerical_state_fingerprint

    native = g.native_module()
    bank = native.benchmark.get_benchmark_dict()[g.SUITE]().get_task_init_states(g.TASK)
    hashes = g.released_row_fingerprints()
    rows = []
    with closing(g.make_sampler()) as env:
        native._reset_wrapper(env, 0)
        for i, (state, fingerprint) in enumerate(zip(bank, hashes)):
            require(numerical_state_fingerprint(state) == fingerprint, "Released bank changed")
            native.load_panda_row(env, state)
            positions = {k: env.sim.data.body_xpos[env.env.obj_body_id[k]][:2].tolist() for k in g.MOVABLES}
            rows.append(dict(released_id=i, initial_state_sha256=fingerprint,
                             movable_xy_hash=canonical_fingerprint(positions),
                             mobile_joint_xy_hash=mobile_joints(native.scene_state(env))))
    return rows


_authorized_target_rows = set()


def target_gate(root, phase):
    require(phase == "test", "Held-out body requires the identical sealed test layouts")
    seal = selection(root)
    audit_development(root, seal)
    panel = manifest(root, phase)
    require(panel["development_selection_sha256"] == sha(root / "accepted/selection.json"),
            "Test manifest does not bind the saved selection")
    stage = directory(root, phase, seal, seal["stride"])
    contract = read(stage / "contract.json")
    require(all(contract[k] == seal[k] for k in ("fit_id", "checkpoint_sha256", "step", "stride", "evaluator",
                                                  "architecture", "prompt")) and
            contract["phase"] == "test" and contract["panel_sha256"] == sha(root / phase / "manifest.json"),
            "Source confirmation differs from the accepted motor or test panel")
    report = report_for(panel, contract, stage)
    require(report["passed"] and read(stage / "report.json") == report,
            "Held-out reset requires a passing, complete two-arm source test")
    _authorized_target_rows.update((r["layout_id"], r["initial_state_sha256"], r["scene_hash"]) for r in panel["layouts"])
    return panel


def physical_reset(env, arm, camera, state, row):
    import generated_layouts as g
    native = g.native_module()
    target = env.native.arm if camera else env.arm
    if arm in ARMS:
        obs = g.reset_generated(env, arm, camera, state, row)
    else:
        # ArmEnv.reset performs the official donor transfer, bounded IK, ten settle
        # controls and scene validation. Its private singleton slot is restored even on errors.
        require(arm == "UR5e" and camera and target.arm == arm and
                (row["layout_id"], row["initial_state_sha256"], row["scene_hash"]) in _authorized_target_rows,
                "Held-out physical reset requires the audited source gate for this exact layout")
        original, target._states = target._states, (state,)
        try:
            obs = env.reset(0, row["layout_seed"])
        finally:
            target._states = original
        physics = env.reset_info["environment_reproducibility"]
        require(physics["init_state_id"] == 0 and physics["released_row_fingerprint"] == row["initial_state_sha256"],
                "Held-out reset used another donor row")
        physics["generated_row_fingerprint"] = physics.pop("released_row_fingerprint")
        physics.update(init_state_id=row["layout_id"], bank_row_index=0, source_kind=row["source_kind"],
                       native_sampler_seed=row["layout_seed"], layout_seed=row["layout_seed"])
        physics["settled_collision_check"] = g.collision_admission(target._env)
    info = env.reset_info
    p = info["environment_reproducibility"]
    require((info["model_identity"]["arm"], info["max_steps"], p["suite"], p["task_id"], p["seed"],
             p["init_state_id"], p["scene_fingerprint"], p["generated_row_fingerprint"]) ==
            (arm, 300, g.SUITE, g.TASK, row["layout_seed"], row["layout_id"], row["scene_hash"],
             row["initial_state_sha256"]) and p["perturbation_plan"]["case_id"] == "BASE" and
            p["perturbation_plan"]["seed"] == row["layout_seed"] and
            all(p[k] == row[k] for k in ("source_pin", "manifest_sha256", "bddl_sha256")),
            "Physical reset receipt differs from the fixed panel")
    poses = native.scene_body_poses(target._env)
    info["admission"] = capture(target._env, info, poses, obs["state"])
    if camera:
        renderer = info["renderer"]
        require(len(renderer) == 2 and "NVIDIA" in renderer[0].upper() and renderer[1],
                "Rendered reset requires a qualified NVIDIA renderer")
    if "panda_settled_poses" in row:
        errors = native.pose_errors(poses, row["panda_settled_poses"])
        require(errors["max_position_m"] <= native.POSITION_TOL_M and
                errors["max_rotation_rad"] <= native.ROTATION_TOL_RAD, "Post-settle scene differs from paired Panda")
        info["paired_scene_errors"] = errors
    if arm in row.get("preflight", {}):
        info["paired_reset_audit"] = audit_pair(row["preflight"][arm], info["admission"])
    return obs, info, poses


@contextmanager
def preparing(base, seed):
    try:
        yield
    except Exception as exc:
        bind(base / "errors" / f"{seed}-{uuid4().hex}.json",
             dict(layout_seed=seed, error=type(exc).__name__, reason=str(exc),
                  paired_reset=getattr(exc, "report", None)))
        raise


def prepare(args):
    import numpy as np
    import generated_layouts as g
    from expert import session
    root, phase = args.root, args.phase
    first, count, threshold = PHASES[phase]
    _, training_seeds, training_hashes = training(args.fit)
    seal_sha = None
    prior = []
    if phase == "test":
        accepted = selection(root)
        require(accepted["fit_id"] == sha(args.fit / "manifest.json") and accepted["evaluator"] == implementation(),
                "Sealed preparation requires the same accepted fit and evaluator")
        audit_development(root, accepted)
        seal_sha = sha(root / "accepted/selection.json")
        prior = manifest(root, "dev")["layouts"]
    base = root / phase
    with lock(base / ".prepare.lock"):
        versions = implementation()
        revision = os.environ.get("MOTOR_CODE_REVISION")
        bind(base / "prepare-contract.json", dict(phase=phase, revision=revision, implementation=versions,
             fitting_seeds=sorted(training_seeds), fitting_hashes=sorted(training_hashes),
             development_selection_sha256=seal_sha))
        if (base / "manifest.json").exists():
            # Publish the digest last. Recover a crash after atomic manifest publication.
            bind(base / "manifest.sha256.json", sha(base / "manifest.json"))
            existing = manifest(root, phase, fit=args.fit)
            require(existing["development_selection_sha256"] == seal_sha, "Prepared under another selection")
            return
        public = released()
        fields = GEOMETRY_FIELDS
        seen = {k: {r[k] for r in (*public, *prior) if k in r} | training_hashes for k in fields}
        layouts, rejected = [], []
        with closing(g.make_sampler()) as sampler:
            for seed in range(first, first + 10000):
                # Membership filtering lives here so every skip records the actual matched hash.
                # Other sampler, geometry, pairing or runtime exceptions abort this seed.
                with preparing(base, seed):
                    saved = base / "preflight" / f"{seed}.json"
                    if saved.exists():
                        row = read(saved)
                        require(row["layout_seed"] == seed and sha(base / row["state_file"]) == row["file_sha256"]
                                and all(row[k] not in seen[k] for k in fields), "Invalid resumed preflight")
                        for k in fields:
                            seen[k].add(row[k])
                        layouts.append(row)
                        if len(layouts) == count:
                            break
                        continue
                    state, origin = g.sample_layout(sampler, seed, frozenset())
                    row = dict(origin, split=phase, layout_id=origin["source_index"],
                               mobile_joint_xy_hash=mobile_joints(origin["sampled_scene"]),
                               state_file=f"rows/{seed}.npy", preflight={})
                    matched = {k: row[k] for k in fields if row[k] in seen[k]}
                    if matched:
                        rejection = dict(layout_seed=seed, matching_hashes=matched)
                        bind(base / "hash-rejects" / f"{seed}.json", rejection)
                        rejected.append(rejection)
                        continue
                    for arm in ARMS:
                        with session(arm, False) as env:
                            _, info, poses = physical_reset(env, arm, False, state, row)
                        row["preflight"][arm] = info["admission"]
                        if arm == "Panda":
                            row["panda_settled_poses"] = poses
                        else:
                            row["source_pair_errors"] = info["paired_scene_errors"]
                    if not layouts:
                        # Matched A/B in the prepare interpreter before any motor action.
                        # The original serialized state is identical on both reset paths.
                        row["camera_ab"] = {}
                        for arm in ARMS:
                            with session(arm, True) as env:
                                _, info, _ = physical_reset(env, arm, True, state, row)
                            row["camera_ab"][arm] = dict(rendered=info["admission"],
                                                        comparison=info["paired_reset_audit"])
                    path = base / row["state_file"]
                    with publish(path) as tmp, tmp.open("wb") as stream:
                        np.save(stream, state, allow_pickle=False)
                    row["file_sha256"] = sha(path)
                    bind(saved, row)
                    for k in fields:
                        seen[k].add(row[k])
                    layouts.append(row)
                    print(f"preflight {phase} {len(layouts)}/{count} seed={seed}", flush=True)
                    if len(layouts) == count:
                        break
        require(len(layouts) == count, "Reserved seed window exhausted")
        bind(base / "manifest.json", dict(phase=phase, size=count, required=threshold,
             seed_range=[first, first+9999], suite=g.SUITE, task_id=g.TASK, case=g.CASE,
             released_bank_sha256=g.BANK_SHA, released=public, layouts=layouts, rejected=rejected,
             preparation_fit_sha256=sha(args.fit / "manifest.json"), development_selection_sha256=seal_sha,
             fitting_seeds=sorted(training_seeds), fitting_hashes=sorted(training_hashes),
             revision=revision, implementation=versions))
        bind(base / "manifest.sha256.json", sha(base / "manifest.json"))
        manifest(root, phase)


def serve(fd, root, phase, arm):
    from multiprocessing.connection import Connection
    import numpy as np
    from expert import session
    from sim import _pack, _public
    with Connection(fd) as channel:
        try:
            require(arm in (*ARMS, "UR5e"), "Unknown panel arm")
            if arm == "UR5e":
                panel = target_gate(root, phase)
            else:
                if phase == "test":
                    selection(root)
                panel = manifest(root, phase)
            rows = {r["layout_id"]: r for r in panel["layouts"]}
            with session(arm, True) as env:
                while True:
                    op, value = channel.recv()
                    if op == "close":
                        break
                    if op == "reset":
                        row = rows[value]
                        file = root / phase / row["state_file"]
                        require(sha(file) == row["file_sha256"], "Panel array changed")
                        try:
                            obs, info, _ = physical_reset(env, arm, True, np.load(file, allow_pickle=False), row)
                        except ResetMismatch as exc:
                            bind(root / phase / "errors" / f"{arm}-{row['layout_seed']}-{uuid4().hex}.json",
                                 dict(arm=arm, layout_seed=row["layout_seed"], paired_reset=exc.report))
                            raise
                        result = _pack(obs), info
                    elif op == "advance":
                        require(1 <= len(value) <= 5, "Expected a bounded action prefix")
                        trace = []
                        for action in value:
                            # Preserve what ArmEnv actually issued and its native step counter.
                            packet = env.native.arm.step(action)
                            obs = _public(packet)
                            done = packet["terminated"] or packet["truncated"]
                            trace.append((packet["sensor_motion"].tolist(), packet["reward"], done,
                                          packet["success"], packet["issued_action"].tolist(),
                                          packet["worker_step_index"]))
                            if done:
                                break
                        result = _pack(obs), trace
                    else:
                        raise ValueError(f"Unknown generated-session operation {op}")
                    channel.send((True, result))
        except (EOFError, BrokenPipeError):
            pass
        except Exception:
            channel.send((False, traceback.format_exc()))


class Session:
    """Public reset(layout_id) and step(action); physical metadata stays in reset_info.

    The interpreter owns its arm, private donor state and native physics. The held-out
    arm only opens the sealed panel after auditing both complete source arms.
    """

    def __init__(self, root, phase, arm):
        from multiprocessing import Pipe
        require(phase in PHASES and arm in (*ARMS, "UR5e"), "Unsupported phase or arm")
        self.root = Path(root).resolve()
        self.reset_info = None
        self.pipe, peer = Pipe()
        self.closed = False
        try:
            runtime = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            self.worker = subprocess.Popen(
                [os.environ["CROSSARM_SIM_PYTHON"], "-Bu", str(Path(__file__).resolve()),
                 str(peer.fileno()), str(self.root), phase, arm], pass_fds=(peer.fileno(),), env=runtime)
        except BaseException:
            self.pipe.close()
            raise
        finally:
            peer.close()

    def _call(self, op, value):
        self.pipe.send((op, value))
        ok, response = self.pipe.recv()
        if not ok:
            raise RuntimeError(response)
        return response

    def reset(self, layout_id):
        from sim import _unpack
        self.reset_info = None
        packet, self.reset_info = self._call("reset", int(layout_id))
        return _unpack(packet)

    def advance_native(self, actions):
        from sim import _unpack
        import numpy as np
        packet, trace = self._call("advance", np.asarray(actions).tolist())
        return _unpack(packet), trace

    def advance(self, actions):
        obs, trace = self.advance_native(actions)
        return obs, [entry[:4] for entry in trace]

    def step(self, action):
        import numpy as np
        obs, ((motion, reward, done, success),) = self.advance([action])
        return obs, np.asarray(motion, dtype=np.float32), reward, done, success

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.pipe.send(("close", None))
        except (OSError, BrokenPipeError):
            pass
        finally:
            self.pipe.close()
        try:
            self.worker.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.worker.kill()
            self.worker.wait()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


if __name__ == "__main__":
    serve(int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4])
