"""Fit two independent student adapters with identical native flow-matching batches."""
import argparse
import copy
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from hydra import compose, initialize_config_dir
from rlinf.models.embodiment.openpi import get_model
from rlinf.models.embodiment.openpi.modules.model import preprocess_observation

from fluffy_bassoon.meta_rl.flow_history import DirectActionCodec, native_flow_targets
from fluffy_bassoon.robocasa.residual import combine_observations


class LowRank(nn.Module):
    def __init__(self, layer, rank=8):
        super().__init__()
        self.down = nn.Linear(layer.in_features, rank, bias=False)
        self.up = nn.Linear(rank, layer.out_features, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(x.float())).to(x.dtype)


class PairedStudents(nn.Module):
    def __init__(self, native, history_width):
        super().__init__()
        self.active = None
        self.context = None
        layers = {f"layer{i}_{name}": getattr(block.attn, name)[1]
                  for i, block in enumerate(native.llm.layers)
                  for name in ("q_proj", "v_proj") if i >= len(native.llm.layers) - 4}
        bank = nn.ModuleDict({name: LowRank(layer) for name, layer in layers.items()})
        self.adapters = nn.ModuleDict({"memoryless": bank, "dpt": copy.deepcopy(bank)})
        self.encoder = nn.GRU(history_width, 128, batch_first=True)
        self.project = nn.Linear(128, native.action_in_proj.out_features, bias=False)
        nn.init.zeros_(self.project.weight)
        for name, layer in layers.items():
            layer.register_forward_hook(self.lora_hook(name))
        native.action_in_proj.register_forward_hook(self.context_hook)

    def lora_hook(self, name):
        def add(module, inputs, output):
            if self.active is None:
                return output
            return output + self.adapters[self.active][name](inputs[0]).to(output.dtype)
        return add

    def context_hook(self, module, inputs, output):
        if self.active == "dpt":
            return output + self.context[:, None, :].to(output.dtype)
        return output

    def select(self, name, history):
        self.active = name
        self.context = None
        if name == "dpt":
            _, hidden = self.encoder(history)
            self.context = self.project(hidden[-1])


@lru_cache(maxsize=320)
def load(path):
    return torch.load(path, weights_only=False, map_location="cpu")


class Corpus:
    def __init__(self, folder, stats):
        self.folder, self.stats = folder, stats
        self.codec = DirectActionCodec.from_stats(stats)
        self.rows = {"train": {}, "validation": {}}
        self.history, self.donors = {}, {}
        self.manifest = (folder / "dataset.json").read_bytes()
        for bank in json.loads(self.manifest)["banks"]:
            task, rank = bank["task"], bank["rank"]
            support = load(folder / task / str(rank) / "support.pt")
            for lane, split in enumerate(bank["split"]):
                key = (task, rank, lane)
                self.history[key] = support[lane]
                self.rows[split].setdefault(task, []).extend((*key, chunk) for chunk in range(1, 13))
        for task_rows in self.rows["validation"].values():
            keys = sorted({row[:3] for row in task_rows})
            self.donors.update(dict(zip(keys, keys[1:] + keys[:1])))
        train_ids = {r[:3] for rows in self.rows["train"].values() for r in rows}
        x = torch.cat([self.history[k] for k in sorted(train_ids)])
        self.mean, self.scale = x.mean(0), x.std(0).clamp_min(.1)
        self.history = {k: ((v - self.mean) / self.scale).clamp(-12, 12) for k, v in self.history.items()}

    def batches(self, split, epoch):
        rng = random.Random(epoch + 789)
        all_rows = []
        for rows in self.rows[split].values():
            rows = rows.copy()
            rng.shuffle(rows)
            all_rows.append(rows)
        assert len({len(rows) for rows in all_rows}) == 1
        return [[row for rows in all_rows for row in rows[i:i + 2]]
                for i in range(0, len(all_rows[0]), 2)]

    def batch(self, rows):
        observations, targets, histories = [], [], []
        for task, rank, lane, chunk in rows:
            p = load(self.folder / task / str(rank) / f"query-{chunk:02d}.pt")
            observations.append({k: v[lane:lane+1] for k, v in p["observation"].items()})
            targets.append(p["target"][lane])
            histories.append(self.history[task, rank, lane])
        target = native_flow_targets(torch.stack(targets).numpy(), self.codec, self.stats["actions"])
        return combine_observations(observations), torch.from_numpy(target).cuda(), torch.stack(histories).cuda()


def paired_input(native, corpus, rows, seed):
    raw, target, history = corpus.batch(rows)
    with torch.no_grad():
        obs = preprocess_observation(native.env_obs_to_observation(raw), train=False)
        _, mask, cache = native.build_prefix_cache(obs)
        rng = torch.Generator(device="cuda").manual_seed(seed)
        noise = torch.randn(target.shape, generator=rng, device="cuda")
        t = torch.rand(len(rows), generator=rng, device="cuda").pow(1 / 1.5) * .999 + .001
        x = torch.lerp(target, noise, t[:, None, None])
    return (obs, x, t, cache, mask), (noise-target)[..., :12], history


def velocity(native, inputs):
    return native.velocity_from_suffix(native.run_suffix(*inputs))[..., :12]


def save_checkpoint(out, step, students, optimizers, corpus, contract):
    state = dict(step=step, students=students.state_dict(), optimizers={k:v.state_dict() for k,v in optimizers.items()},
                 sampler="epoch-seeded balanced permutations and absolute-step flow noise",
                 cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), contract=contract,
                 history_mean=corpus.mean, history_scale=corpus.scale)
    torch.save(state, out / "next.tmp")
    os.replace(out / "next.tmp", out / "checkpoint.pt")


@torch.no_grad()
def validate(native, students, corpus, out):
    scores = []
    for i, rows in enumerate(corpus.batches("validation", 0)):
        inputs, target, own = paired_input(native, corpus, rows, 910000 + i)
        donor = torch.stack([corpus.history[corpus.donors[r[:3]]] for r in rows]).cuda()
        losses = {}
        for label, arm, history in (("frozen", None, own), ("memoryless", "memoryless", own),
                                    ("own", "dpt", own), ("same_task_donor", "dpt", donor)):
            students.select(arm, history)
            losses[label] = (velocity(native, inputs) - target).square().mean((1, 2)).tolist()
        scores.extend(dict(task=r[0], rank=r[1], lane=r[2], chunk=r[3],
                           **{k:v[j] for k,v in losses.items()}) for j,r in enumerate(rows))
    (out / "held-out.json").write_text(json.dumps(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("code", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(991)
    assert torch.cuda.device_count() == 1
    stats = json.loads(Path(os.environ["ROBOCASA_PI05_NATIVE_NORM"]).read_text())
    corpus = Corpus(args.data, stats.get("norm_stats", stats))
    print("Loading frozen pi0.5", flush=True)
    with initialize_config_dir(config_dir=str(args.code / "benchmarks/robocasa/configs"), version_base=None):
        config = compose(config_name="robocasa365_pi05_eval")
    native = get_model(config.rollout.model, torch_dtype=torch.bfloat16).to("cuda").eval().requires_grad_(False)
    students = PairedStudents(native, len(corpus.mean)).cuda()
    trainables = {arm:list(bank.parameters()) for arm, bank in students.adapters.items()}
    trainables["dpt"] += list(students.encoder.parameters()) + list(students.project.parameters())
    optimizers = {k:torch.optim.AdamW(params, lr=3e-4, weight_decay=.01) for k,params in trainables.items()}
    contract = dict(data_sha256=hashlib.sha256(corpus.manifest).hexdigest(), code=os.environ["PILOT_COMMIT"],
                    batch=6, epochs=2, seed=991, rank=8, layers=list(students.adapters["memoryless"]),
                    trainable={k:sum(p.numel() for p in params) for k,params in trainables.items()})
    print(json.dumps(contract), flush=True)
    assert not any(p.requires_grad for p in native.parameters())
    start = 0
    if args.resume:
        restored = torch.load(args.resume, map_location="cpu", weights_only=False)
        assert restored["contract"] == contract
        students.load_state_dict(restored["students"])
        for k, opt in optimizers.items():
            opt.load_state_dict(restored["optimizers"][k])
        torch.set_rng_state(restored["cpu_rng"])
        torch.cuda.set_rng_state_all(restored["cuda_rng"])
        start = restored["step"]
    plan = corpus.batches("train", 0) + corpus.batches("train", 1)
    wall = time.monotonic()
    for step, rows in enumerate(plan, 1):
        if step <= start:
            continue
        inputs, target, history = paired_input(native, corpus, rows, 20000 + step)
        if step == 1:
            with torch.no_grad():
                students.select(None, history)
                reference = velocity(native, inputs)
                for arm in optimizers:
                    students.select(arm, history)
                    assert torch.equal(velocity(native, inputs), reference), arm
        report = {}
        for arm, opt in optimizers.items():
            students.select(arm, history)
            opt.zero_grad(set_to_none=True)
            loss = (velocity(native, inputs) - target).square().mean()
            loss.backward()
            norm = nn.utils.clip_grad_norm_(trainables[arm], 1., error_if_nonfinite=True)
            if step <= 2:
                assert norm > 0 and any(layer.up.weight.grad.abs().max() > 0 for layer in students.adapters[arm].values())
                if arm == "dpt":
                    assert students.project.weight.grad.abs().max() > 0
            opt.step()
            report[arm] = {"loss":loss.item(), "grad_norm":norm.item()}
        if step <= 2 or step % 8 == 0:
            print(json.dumps(dict(step=step, total=len(plan), elapsed_s=round(time.monotonic()-wall, 2),
                                  peak_gpu_gib=round(torch.cuda.max_memory_allocated()/2**30, 2), **report)), flush=True)
        if step in (1, 8) or step % 48 == 0 or step == len(plan):
            save_checkpoint(args.out, step, students, optimizers, corpus, contract)
    validate(native, students, corpus, args.out)
    print("Completed both fits and held-out comparison", flush=True)


if __name__ == "__main__":
    main()
