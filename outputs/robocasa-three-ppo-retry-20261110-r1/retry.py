"""Native matched second attempts with the two frozen student adapters."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import time
from types import SimpleNamespace

ROOT = Path("/mowenjie/fluffy-bassoon-2")
PILOT = ROOT / "outputs/robocasa-three-ppo-dpt-lora-20261110-r1"
TASKS = ("OpenDrawer", "CloseToasterOvenDoor", "TurnOnSinkFaucet")
CONFIG_HASH = ("88c9e8f64c1b18ea1491548e56f6caf5d0172e253dc9611f8ac779ca6ac2f2f7",
               "a2e915b7160f3ae7007fbfb724c34d3ecd422abad6a11f330bc35f92eccdcc66",
               "b3066eed35e5d1a57a2e50c8591625f5a94b1fd70ba88aaa692a2b9c8c20bf91")
ARMS = ("frozen", "memoryless", "own", "donor")
BANK_SIZE = 8


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    staging = path.with_suffix(".partial")
    staging.write_text(json.dumps(obj, indent=2))
    staging.replace(path)


class ChunkRecorder:
    """Receive the native collector's payload before its optional disk compression."""
    def __init__(self, codec):
        self.codec = codec

    def start(self, arm):
        self.arm = arm
        self.tokens = []
        self.receipts = []

    def record(self, writer, path, payload):
        from prepare import support_token
        import numpy as np
        obs = payload["current_observation"]
        executed = np.asarray(payload["decoded_actions"])
        encoded = self.codec.encode(executed)
        reward, term, trunc = (payload[k] for k in ("rewards", "terminations", "truncations"))
        chunk = len(self.receipts) + 1
        assert tuple(reward.shape) == (8, 50) and not term.any()
        assert not trunc[:, :-1].any() and (trunc[:, -1] == (chunk == 12)).all()
        if self.arm == "support":
            self.tokens.append(support_token(obs, encoded, reward, term, trunc, chunk))
        self.receipts.append({"chunk":chunk, "actions_sha256":hashlib.sha256(executed.tobytes()).hexdigest(),
                              "reward":reward.sum(dim=1).tolist()})


def task_config(task):
    from omegaconf import OmegaConf
    if task == "TurnOnSinkFaucet":
        directory = ROOT / "outputs/robocasa-eight-hour-babysit-20261104-r1/source5" / task / "attempt-02"
    else:
        directory = ROOT / "outputs/robocasa-flow-sde-10m-per-task-20261103-r1" / task / "production-r1"
    path = directory / "resolved-config.json"
    assert sha(path) == CONFIG_HASH[TASKS.index(task)]
    cfg = OmegaConf.create(json.loads(path.read_text())["env"]["eval"])
    assert cfg.task_filter == [task] and cfg.ignore_terminations and not cfg.auto_reset and cfg.is_eval
    assert cfg.max_episode_steps == 600 and cfg.episode_horizon_source == "max_episode_steps"
    cfg.total_num_envs = BANK_SIZE
    cfg.video_cfg.save_video = False
    return cfg


def load_policy(code, checkpoint):
    import torch
    from fit import PairedStudents
    from hydra import compose, initialize_config_dir
    from rlinf.models.embodiment.openpi import get_model
    with initialize_config_dir(config_dir=str(code / "benchmarks/robocasa/configs"), version_base=None):
        cfg = compose(config_name="robocasa365_pi05_eval")
    native = get_model(cfg.rollout.model, torch_dtype=torch.bfloat16).cuda().eval().requires_grad_(False)
    assert native.num_steps == 10 and native.action_horizon == 50
    adapters = PairedStudents(native, len(checkpoint["history_mean"])).cuda()
    adapters.load_state_dict(checkpoint["students"], strict=True)
    adapters.eval().requires_grad_(False)
    return native, adapters


def run(args):
    import numpy as np
    import torch
    import torch.multiprocessing as mp
    from omegaconf import OmegaConf
    from rlinf.envs.sim.robocasa365.robocasa365_env import Robocasa365Env, map_released_control_scores
    from fluffy_bassoon.robocasa.flow_ppo_runtime import native_class
    from fluffy_bassoon.robocasa.residual import public_observation
    from fluffy_bassoon.rlinf.native_ppo import ActionJournal
    from fluffy_bassoon.meta_rl.flow_history import DirectActionCodec

    mp.set_start_method("spawn", force=True)
    torch.set_num_threads(2)
    torch.manual_seed(1001)
    idx = TASKS.index(args.task)
    env_cfg = task_config(args.task)
    source = PILOT / "fit-01/checkpoint.pt"
    assert sha(source) == "63cfcb66da28bb9d421b6c0363093b7e532f151d1a029439de858285a36cf397"
    ck = torch.load(source, weights_only=False, map_location="cpu")
    assert ck["step"] == 576 and ck["contract"]["data_sha256"] == sha(PILOT / "data-01/dataset.json")
    for name, step in (("memoryless", args.memoryless_step), ("dpt", args.dpt_step)):
        if step == 576:
            continue
        candidate = args.curves / "seed-991" / f"step-{step:04d}/checkpoint.pt"
        fitted = torch.load(candidate, weights_only=False, map_location="cpu")
        assert fitted["step"] == step and fitted["contract"]["seed"] == 991
        assert fitted["contract"]["data_sha256"] == ck["contract"]["data_sha256"]
        assert torch.equal(fitted["history_mean"], ck["history_mean"]) and torch.equal(fitted["history_scale"], ck["history_scale"])
        for key in ck["students"]:
            if key.startswith(f"adapters.{name}.") or (name == "dpt" and key.startswith(("encoder.", "project."))):
                ck["students"][key] = fitted["students"][key]

    original_banks = json.loads((PILOT / "data-01/dataset.json").read_text())["banks"]
    old_xml = {scene for b in original_banks for scene in b["scenes"]}
    old_seeds = {seed for b in original_banks for seed in b["seeds"]}
    stats = json.loads(Path(os.environ["ROBOCASA_PI05_NATIVE_NORM"]).read_text())
    recorder = ChunkRecorder(DirectActionCodec.from_stats(stats.get("norm_stats", stats)))
    owner = SimpleNamespace(_flow_contract={"task":args.task, "group_size":1, "chunk":50,
                                            "physics_tick_limit":(600+50)*128},
                            _rank=0, _flow_iteration=0, _flow_phase="support",
                            _submit_flow_io=recorder.record)
    owner.cfg = OmegaConf.create({"native_pilot":{"fresh_eval_processes":True, "output":str(args.out),
                                "deadline_unix":time.time()+86400}})
    owner._flow_journal = ActionJournal(args.out / "accounting", rank=0, world=1, attempt="retry",
                                       maximum_actions=120000)
    native, students = load_policy(args.code, ck)
    env = native_class(Robocasa365Env, owner)(env_cfg, num_envs=8, seed_offset=0, total_num_processes=1, worker_info=None)
    print("Simulator and checkpoint initialized", args.task, flush=True)
    try:
        for bank in range(args.first_bank, args.end_bank):
            folder = args.out / f"bank-{bank:02d}"
            folder.mkdir()
            owner.cfg.native_pilot.output = str(folder)
            first = 87110000 + 10000*idx + bank*BANK_SIZE
            seeds = list(range(first, first+8))
            assert old_seeds.isdisjoint(seeds)
            owner._flow_contract["eval_seed"] = first
            sequence = ARMS[bank%4:] + ARMS[:bank%4]
            history = None
            panels = {}
            donor_lane = (torch.arange(8) + 1 + bank%7) % 8
            arms = ["support", *(k for k in sequence if k in args.arms)]
            if args.reuse:
                original = args.reuse / f"{args.task}-{bank}" / f"bank-{bank:02d}"
                # Copy the original canonical snapshot, so the existing strict physical
                # comparison must pass before spending any query controls.
                shutil.copytree(original / "canonical-eval", folder / "canonical-eval")
                previous = json.loads((original / "support.json").read_text())
                assert [item["seed"] for item in previous["episodes"]] == seeds
                support = torch.load(original / "support.pt", weights_only=True)
                assert support.shape[:2] == (8,12) and torch.isfinite(support).all()
                history = ((support - ck["history_mean"]) / ck["history_scale"]).clamp(-12,12).cuda()
                panels["support"], scenes = previous, previous["scene_xml"]
                assert old_xml.isdisjoint(scenes) and len(set(scenes)) == 8
                old_xml.update(scenes)
                save(folder / "support.json", previous)
                save(folder / "support-source.json", dict(path=str(original), tensor_sha256=sha(original / "support.pt")))
                arms.remove("support")
            for arm in arms:
                owner._flow_phase = arm
                recorder.start(arm)
                students.select(None, None)
                obs, _ = env.reset()
                current = folder / "environment/rank-00" / env._flow_key
                initial = json.loads(Path(str(current)+"-initial.json").read_text())["report"]
                assert initial["matched"] and initial["physical_state_exact"] and initial["public_state_exact"]
                if arm == "support":
                    snapshot = torch.load(Path(str(current)+"-initial.pt"), weights_only=False, map_location="cpu")
                    scenes = [s["xml_sha256"] for s in snapshot["physical"]]
                    assert list(snapshot["case"]["seeds"]) == seeds and len(set(scenes)) == 8
                    assert old_xml.isdisjoint(scenes)
                    old_xml.update(scenes)
                seed = first + (191000000 if arm == "support" else 192000000)
                generator = torch.Generator(device="cuda").manual_seed(seed)
                noise_hash = hashlib.sha256()
                with torch.no_grad():
                    if arm == "memoryless":
                        students.select("memoryless", None)
                    if arm in ("own", "donor"):
                        students.select("dpt", history if arm == "own" else history[donor_lane])
                    for chunk in range(12):
                        noise = torch.randn((8,50,32), generator=generator, device="cuda")
                        noise_hash.update(noise.cpu().numpy().tobytes())
                        proposal, _ = native.predict_action_batch(public_observation(obs), mode="eval", noise=noise)
                        actions = map_released_control_scores(proposal.detach().float().cpu().numpy(), env_cfg.action_space)
                        result = env.chunk_step(actions)
                        obs = result[0][-1]
                        del result
                if arm == "support":
                    support = torch.stack(recorder.tokens, dim=1)
                    assert torch.isfinite(support).all() and support.shape[:2] == (8,12)
                    torch.save(support, folder / "support.pt")
                    history = ((support - ck["history_mean"]) / ck["history_scale"]).clamp(-12, 12).cuda()
                episodes = json.loads(Path(str(current)+"-episodes.json").read_text())["rows"]
                assert [row["seed"] for row in episodes] == seeds
                assert len(recorder.receipts) == 12
                rewards = np.array([r["reward"] for r in recorder.receipts]).sum(0)
                assert np.array_equal(rewards, [r["success_once"] for r in episodes])
                panel = dict(arm=arm, episodes=episodes, reset=initial, noise_sha256=noise_hash.hexdigest(),
                             chunks=recorder.receipts, scene_xml=scenes, donor_lanes=donor_lane.tolist())
                panels[arm] = panel
                save(folder / (arm+".json"), panel)
                print(args.task, bank, arm, "success", int(rewards.sum()), "/ 8", flush=True)
            assert len({panels[k]["noise_sha256"] for k in args.arms}) == 1
            assert panels["support"]["noise_sha256"] != panels[args.arms[0]]["noise_sha256"]
            save(folder/"complete.json", {"seeds":seeds,"arms":list(panels),"order":list(sequence)})
        save(args.out/"complete.json", {"task":args.task,"banks":list(range(args.first_bank,args.end_bank)),
                                       "charged_controls":owner._flow_journal.charged()})
    finally:
        if hasattr(env, "env"):
            env.env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("code", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--first-bank", type=int, default=0)
    parser.add_argument("--end-bank", type=int, default=4)
    parser.add_argument("--reuse", type=Path)
    parser.add_argument("--curves", type=Path)
    parser.add_argument("--memoryless-step", type=int, default=576)
    parser.add_argument("--dpt-step", type=int, default=576)
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS))
    run(parser.parse_args())
