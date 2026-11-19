"""Frozen SigLIP and matched RGB readouts on the archived paired bowl tapes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

START = perf_counter()
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path("/mowenjie/fluffy-bassoon-2")
WITNESS = ROOT / "outputs/libero-visual-witness-20261119-r1"
HUB = Path("/mowenjie/.cache/huggingface/hub")
ROBOT = Path("/mowenjie/fluffy-bassoon/inputs/pi05-libero-lerobot-v044-7152f13e-20261027-r1")
IDS = {
    "robot": ("lerobot/pi05_libero_finetuned_v044", "7152f13e34c6c115d808aea0e5d08aca74eac14b"),
    "pt": ("google/paligemma-3b-pt-224", "35e4f46485b4d07967e7e9935bc3786aad50687c"),
    "mix": ("google/paligemma-3b-mix-224", "d1d8734c9c3ad0ccfeea4afc270faa356c2ba515"),
}
SHARD = "model-00001-of-00003.safetensors"
STAGED_PT = ROOT / "inputs/libero-witness-paligemma-pt-20261119-r1"
CONTROLS = (0, 75, 150, 225, 300)
ARCH = dict(hidden_size=1152, intermediate_size=4304, num_hidden_layers=27,
            num_attention_heads=16, patch_size=14, image_size=224, num_channels=3,
            hidden_act="gelu_pytorch_tanh", layer_norm_eps=1e-6,
            attention_dropout=0.0, vision_use_head=False)


def snapshot(kind):
    repo, revision = IDS[kind]
    return HUB / ("models--" + repo.replace("/", "--")) / "snapshots" / revision


def checkpoint(args):
    if args.arm == "robot":
        return "robot", args.robot_weights
    if args.pt_weights is not None and not args.pt_weights.is_file():
        raise FileNotFoundError(args.pt_weights)
    pt_candidates = ([args.pt_weights] if args.pt_weights else
                     [STAGED_PT / "vision.safetensors", STAGED_PT / SHARD, snapshot("pt") / SHARD])
    for path in pt_candidates:
        if path.is_file():
            return "pt", path
    return "mix", args.mix_weights


def load_encoder(kind, path):
    import transformers
    from safetensors import safe_open
    from transformers import SiglipVisionConfig, SiglipVisionModel
    from transformers.modeling_utils import no_init_weights

    config = SiglipVisionConfig(**ARCH, attn_implementation="sdpa")
    with no_init_weights():
        model = SiglipVisionModel(config)
    prefix = ("model.paligemma_with_expert.paligemma.model.vision_tower."
              if kind == "robot" else "vision_tower.")
    state, sentinels, source_dtypes = {}, {}, set()
    with safe_open(str(path), framework="pt", device="cpu") as weights:
        for source_key in weights.keys():
            if not source_key.startswith(prefix):
                continue
            key = source_key[len(prefix):]
            value = weights.get_tensor(source_key)
            source_dtypes.add(str(value.dtype))
            if key in ("vision_model.embeddings.patch_embedding.weight",
                       "vision_model.encoder.layers.26.layer_norm2.weight"):
                sentinels[source_key] = hashlib.sha256(value.view(torch.uint8).numpy()).hexdigest()
            state[key] = value.to(torch.bfloat16)
    model.load_state_dict(state, strict=True, assign=True)
    model.requires_grad_(False).eval()
    repo, revision = IDS[kind]
    return model, dict(id=repo, revision=revision, file=str(path), reference=kind,
                       prefix=prefix, source_dtypes=sorted(source_dtypes),
                       source_tensor_sha256=sentinels, tensor_count=len(state),
                       parameter_count=sum(p.numel() for p in model.parameters()),
                       architecture=ARCH, transformers_version=transformers.__version__,
                       attention_implementation=config._attn_implementation)


@torch.inference_mode()
def features(frames, model, batch_size):
    if frames.shape != (13, 2, 96, 96, 3) or frames.dtype != np.uint8:
        raise ValueError(f"Unexpected RGB tape {frames.shape}, {frames.dtype}")
    selected = frames[np.array(CONTROLS) // 25].reshape(10, 96, 96, 3)
    pooled = []
    for batch in torch.from_numpy(selected).split(batch_size):
        x = F.interpolate(batch.permute(0, 3, 1, 2).float(), size=(224, 224),
                          mode="bilinear", align_corners=False, antialias=True)
        x = x.div_(127.5).sub_(1).to(torch.bfloat16)
        if model is None:
            grid = x
        else:
            patches = model(pixel_values=x).last_hidden_state
            if patches.shape != (len(batch), 256, 1152):
                raise ValueError(f"Unexpected final patch grid {patches.shape}")
            grid = patches.transpose(1, 2).reshape(len(batch), 1152, 16, 16)
        pooled.append(F.adaptive_avg_pool2d(grid.float(), (2, 2)))
    cells = torch.cat(pooled).reshape(5, 2, -1, 2, 2)
    delta = np.diff(cells.numpy().astype(np.float64), axis=0).ravel()
    if not np.isfinite(delta).all():
        raise ValueError("Nonfinite features")
    return delta


def fit_predict(vectors, labels):
    predictions, diagnostics = {}, {}
    for goal in (0, 1):
        keys = [(p, goal, v) for p in range(12, 16) for v in ("BASE", "D18")]
        x = np.stack([vectors[k] for k in keys])
        y = np.array([labels[k] for k in keys], dtype=bool)
        available = y.any() and (~y).any()
        if available:
            miss, hit = x[~y].mean(axis=0), x[y].mean(axis=0)
        diagnostics[goal] = dict(hits=int(y.sum()), misses=int((~y).sum()),
                                 mean_feature_variance=float(x.var(axis=0, ddof=0).mean()))
        for key, z in vectors.items():
            if key[1] == goal:
                score = None
                if available:
                    d0, d1 = np.linalg.norm(z - miss), np.linalg.norm(z - hit)
                    score = float(d0 / (d0 + d1)) if d0 + d1 else 0.5
                predictions[key] = score
    return predictions, diagnostics


def preference(a, b):
    if a is None or b is None or abs(a - b) <= 1e-12:
        return 0.5
    return float(a < b)


def count(values):
    return dict(numerator=sum(values), denominator=len(values))


def summarize(split, predictions):
    rows = []
    for tape in split["physical_tapes"]:
        key = tape["pose"], tape["goal"], tape["variant"]
        score = predictions[key]
        hit_credit = preference(0.5, score)
        rows.append(dict(pose=key[0], goal=key[1], variant=key[2], hit=tape["own_hit"],
                         score=score, correct=hit_credit if tape["own_hit"] else 1-hit_credit))
    panels = {}
    groups = {(g["A"], g["B"], g["query"]): g for g in split["groups"] if g["strict"]}
    for name, panel in split["panels"].items():
        own, swapped = [], []
        for a, b, q in panel["triples"]:
            group = groups.get((a, b, q))
            if group is None:
                continue
            for case in group["cases"]:
                av, bv = case["variants"]
                for left, right, out in ((av, bv, own), (bv, av, swapped)):
                    prob_a = preference(predictions[a, 0, left], predictions[b, 1, right])
                    out.append(prob_a if case["hidden"] == 0 else 1 - prob_a)
        if len(own) != panel["assignments"]:
            raise ValueError(f"Strict assignment count disagrees for {name}")
        panels[name] = dict(strict_groups=panel["strict_groups"], own=count(own),
                            swapped=count(swapped),
                            own_minus_swapped=count([x-y for x, y in zip(own, swapped)]))
    return dict(physical_completion=count([r["correct"] for r in rows]),
                physical_by_class={c: count([r["correct"] for r in rows if r["hit"] == label])
                                   for c, label in (("hit", True), ("miss", False))},
                tapes=rows, balanced_strict_corrections=panels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("robot", "vlm", "rgb"), required=True)
    parser.add_argument("--tapes", type=Path, default=WITNESS / "cpu02")
    parser.add_argument("--score-json", type=Path, default=WITNESS / "score01/score.json")
    parser.add_argument("--robot-weights", type=Path, default=ROBOT / "model.safetensors")
    parser.add_argument("--pt-weights", type=Path, help="Pinned PT vision shard or extracted tower")
    parser.add_argument("--mix-weights", type=Path, default=snapshot("mix") / SHARD)
    parser.add_argument("--batch-size", type=int, default=4, choices=range(1, 11))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    archived_bytes = args.score_json.read_bytes()
    splits = json.loads(archived_bytes)["splits"]
    model, info = None, None
    if args.arm != "rgb":
        model, info = load_encoder(*checkpoint(args))
    vectors, labels, raw_scores = {}, {}, {}
    for split in splits.values():
        for row in split["physical_tapes"]:
            p, g, v = key = row["pose"], row["goal"], row["variant"]
            with np.load(args.tapes / f"p{p}-g{g}-{v}.npz", allow_pickle=False) as tape:
                hits = tape["hits"]
                if (hits.shape != (300, 2) or not np.isin(hits, (0, 1)).all()
                        or hits.any(axis=0).tolist() != row["ever"]
                        or hits[-1].astype(bool).tolist() != row["final"]):
                    raise ValueError(f"Tape predicates disagree with archived score for {key}")
                vectors[key] = features(tape["frames"], model, args.batch_size)
            labels[key] = row["own_hit"]
            raw_scores[key] = row["hit_score"]["rgb"]
            if len(vectors) % 4 == 0:
                print(args.arm, len(vectors), "of 32 tapes", flush=True)
    if len(vectors) != 32:
        raise ValueError("Expected 32 distinct tapes")
    predictions, diagnostics = fit_predict(vectors, labels)
    result = dict(arm=args.arm, checkpoint=info, device="cpu", torch_version=str(torch.__version__),
                  threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                  batch_size=args.batch_size, input_dtype="bfloat16",
                  encoder_dtype="bfloat16" if model is not None else None,
                  pooling_dtype="float32", difference_centroid_distance_variance_dtype="float64",
                  variance="mean of per-coordinate population variances on fitting tapes per goal; no whitening",
                  matmul_precision=torch.get_float32_matmul_precision(), controls=CONTROLS,
                  cameras=2, input_hw=[96, 96], resize_hw=[224, 224], scale="[-1,1]",
                  interpolation="bilinear, antialias=True, align_corners=False",
                  grid_hw=[16, 16] if model is not None else [224, 224], pooled_hw=[2, 2],
                  channels=1152 if model is not None else 3, feature_dim=next(iter(vectors.values())).size,
                  fit_diagnostics=diagnostics, tapes_path=str(args.tapes),
                  strict_reference=str(args.score_json),
                  strict_reference_sha256=hashlib.sha256(archived_bytes).hexdigest(),
                  splits={name: summarize(s, predictions) for name, s in splits.items()},
                  main_raw_rgb=dict(controls=list(range(0, 301, 25)), pooled_hw=[8, 8],
                                    feature_dim=12*2*8*8*3, source="archived score.json",
                                    splits={name: summarize(s, raw_scores) for name, s in splits.items()}))
    result["elapsed_seconds"] = perf_counter() - START
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
