"""Join archived SFT support episodes to final PPO query actions on matched resets."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import torch

from fluffy_bassoon.meta_rl.flow_history import DirectActionCodec
from fluffy_bassoon.robocasa.residual import public_observation

ROOT = Path("/mowenjie/fluffy-bassoon-2")
OUTPUTS = ROOT / "outputs"
TASKS = {"OpenDrawer": 130, "CloseToasterOvenDoor": 130, "TurnOnSinkFaucet": 40}
NORMS = ROOT / "inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json"
VIEWS = ("main_images", "wrist_images", "extra_view_images")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def archive_entries(task):
    archive = Path("/mnt/bos/adgfyl/strong-plant-archives/robocasa-flow-sde-10m-20261103-r1/corpora") / task / "native-130-r1.tar"
    files = json.loads(Path(str(archive) + ".manifest.json").read_text())["files"]
    selected, offset = {}, 0
    for name, record in sorted(files.items()):
        member = tarfile.TarInfo(name)
        member.size = record["bytes"]
        member.mode = record["mode"]
        member.mtime = record["mtime_ns"] / 1e9
        offset += len(member.tobuf(tarfile.PAX_FORMAT))
        if "/environment/rank-" in name and any(s in name for s in ("/before-i000-", "/after-i130-")):
            selected[name] = (offset, record)
        offset += ((member.size + 511) // 512) * 512
    return archive, selected


def read_member(task, phase, rank, name, archive, entries):
    before = phase == "before"
    bank = f"environment/rank-{rank:02d}"
    if archive is None:
        attempt = "attempt-01" if before else "attempt-02"
        source = OUTPUTS / "robocasa-eight-hour-babysit-20261104-r1/source5" / task / attempt / bank / name
        raw = source.read_bytes()
        return raw, {"path": str(source), "sha256": digest(raw)}
    path = f"{task}/{'q1-r1' if before else 'production-r1'}/{bank}/{name}"
    offset, entry = entries[path]
    archive.seek(offset)
    raw = archive.read(entry["bytes"])
    if len(raw) != entry["bytes"] or digest(raw) != entry["sha256"]:
        raise ValueError(f"Archive member verification failed for {path}")
    return raw, {"path": f"{archive.name}/{path}", "sha256": entry["sha256"]}


def support_token(obs, actions, rewards, term, trunc, chunk):
    features = [torch.as_tensor(obs["states"]).float()]
    for key in VIEWS:
        img = obs[key].reshape(8, 256, 256, 3)
        pooled = img.reshape(8, 4, 64, 4, 64, 3).float().mean((2, 4))
        features.append(pooled.flatten(1) / 255)
    boundary = torch.tensor([chunk == 1, chunk == 12]).float().expand(8, -1)
    features.extend([torch.from_numpy(actions), rewards, term, trunc, boundary])
    return torch.cat(features, dim=1).float()


def prepare_bank(task, step, rank, folder, codec, archive, entries):
    paths, seeds, states, scenes = [], None, None, None
    for phase, version in (("before", 0), ("after", step)):
        prefix = f"{phase}-i{version:03d}-round00-"
        def read(name):
            raw, record = read_member(task, phase, rank, prefix + name, archive, entries)
            paths.append(record)
            return raw
        episodes = json.loads(read("episodes.json"))
        rows = episodes["rows"]
        current = [r["seed"] for r in rows]
        assert episodes["completed"] and len(set(current)) == len(rows) == 8
        assert all(r["native_counts"]["controls"] == 600 for r in rows)
        if phase == "before":
            seeds = current
        else:
            assert current == seeds
            snapshot = torch.load(io.BytesIO(read("initial.pt")), map_location="cpu", weights_only=False)
            assert list(snapshot["case"]["seeds"]) == seeds
            scenes = [item["xml_sha256"] for item in snapshot["physical"]]
            assert len(scenes) == 8 and all(len(s) == 64 for s in scenes)
            match = json.loads(read("initial.json"))["report"]
            assert all(match[k] for k in ("matched", "physical_state_exact", "public_state_exact"))
        tokens = []
        for chunk in range(1, 13):
            raw = read(f"chunk{chunk:02d}.pt.gz")
            p = torch.load(io.BytesIO(gzip.decompress(raw)), map_location="cpu", weights_only=False)
            assert list(p["seeds"]) == seeds and int(p["iteration"]) == version and p["phase"] == phase
            obs = public_observation(p["current_observation"])
            assert set(obs) == {*VIEWS, "states", "task_descriptions"}
            if chunk == 1:
                if phase == "before":
                    states = obs["states"].clone()
                else:
                    assert torch.equal(states, obs["states"])
            actions = codec.encode(p["decoded_actions"])
            rewards, term, trunc = [torch.as_tensor(p[k]) for k in ("rewards", "terminations", "truncations")]
            assert actions.shape == (8, 600) and rewards.shape == term.shape == trunc.shape == (8, 50)
            assert not (term | trunc)[:, :-1].any() and (chunk == 12 or not (term | trunc).any())
            if phase == "before":
                tokens.append(support_token(obs, actions, rewards, term, trunc, chunk))
            else:
                torch.save({"observation": obs, "target": torch.from_numpy(actions)},
                           folder / f"query-{chunk:02d}.pt")
        if phase == "before":
            torch.save(torch.stack(tokens, dim=1), folder / "support.pt")
    return {"task": task, "rank": rank, "seeds": seeds, "scenes": scenes, "sources": paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(3)
    stats = json.loads(NORMS.read_text())
    codec = DirectActionCodec.from_stats(stats.get("norm_stats", stats))
    banks = []
    for task, step in TASKS.items():
        archive, entries = (None, {}) if step == 40 else archive_entries(task)
        stream = archive.open("rb") if archive else None
        try:
            for rank in range(8):
                folder = out / task / str(rank)
                folder.mkdir(parents=True)
                bank = prepare_bank(task, step, rank, folder, codec, stream, entries)
                banks.append(bank)
                print(task, rank, "paired", len(bank["seeds"]), flush=True)
        finally:
            if stream:
                stream.close()
    # Keep every query and both original episodes of a reset in one partition.
    for task in TASKS:
        cases = [(b, lane, seed) for b in banks if b["task"] == task for lane, seed in enumerate(b["seeds"])]
        assert len({s for _, _, s in cases}) == len(cases) == 64
        held = {s for _, _, s in sorted(cases, key=lambda x: digest(f"{task}/{x[2]}/split1".encode()))[:16]}
        for b, lane, seed in cases:
            b.setdefault("split", []).append("validation" if seed in held else "train")
    by_split = {split: {scene for b in banks for scene, part in zip(b["scenes"], b["split"]) if part == split}
                for split in ("train", "validation")}
    assert by_split["train"].isdisjoint(by_split["validation"]), "Scene XML appears in both partitions"
    (out / "dataset.json").write_text(json.dumps({"tasks": TASKS, "banks": banks}, indent=2))
    print("Dataset complete", flush=True)


if __name__ == "__main__":
    main()
