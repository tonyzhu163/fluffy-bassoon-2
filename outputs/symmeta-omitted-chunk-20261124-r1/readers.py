"""Matched task-disjoint refits and four views of each original source support."""
import argparse
from dataclasses import asdict
from pathlib import Path
import time

import cv2
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

import eval_wider_family as pr7
from eval_stage2a_live_ur5e import load_v2_checkpoint
from fluffy_bassoon.xemb.equiv_v2 import (
    ReaderV2Config, apply_reader_v2, init_reader_v2, predict_embodiment_v2, reader_loss_v2,
)
from source import CODE_OUTPUTS, IDS, ROOT, read_json, require, save, sha, tasks_by_role, write_json

TUNED = "perceiver_cont_warm"
GEOM = "equiv_v2_dis_dz"
SEEDS = (41, 42, 43)
GEOM_CFG = ReaderV2Config(model_type="factored_equiv_v2", visual_mode="dis_lumaflow", max_chunks=32,
                         use_dz_parallax=True)
TUNED_SPEC, = [s for s in pr7.TRAINED if s[0] == TUNED]
FEATURES = pr7.CHUNK_KEYS
TRAIN_FIELDS = (*FEATURES, *pr7.LABEL_KEYS)


def views(tape, omit):
    """Preserve absolute token positions. Remove every field of exactly one original chunk."""
    require(set(tape) == set(FEATURES), "Model inputs must contain only observed sensor/command arrays")
    if omit is None:
        return tape
    require(omit in range(3) and tape["valid"].shape[1] == 3, "Wrong support grid")
    keep = np.arange(3) != omit
    output = {}
    for k, a in tape.items():
        b = a.copy()
        b[:, omit] = 0
        require(np.array_equal(a[:, keep], b[:, keep]) and not b[:, omit].any(), k)
        output[k] = b
    return output


def decoded(params, cfg, observed):
    # Use the same unjitted forward and NumPy decode as predict_embodiment_v2.
    c, s, lg, grip = map(np.asarray, apply_reader_v2(params, cfg, observed))
    require(all(np.isfinite(x).all() for x in (c, s, lg, grip)), "Nonfinite output")
    yaw = np.degrees(np.arctan2(s, c)).astype(np.float32)
    gain = np.exp(np.clip(lg, np.log(.25), np.log(4.))).astype(np.float32)
    sign = np.where(grip >= 0, 1, -1).astype(np.int32)
    angular_valid = ((c != 0) | (s != 0)) & observed["valid"].any(axis=1)
    return yaw, gain, sign, angular_valid


def inspect_masks(params, cfg, sample):
    """Compare native full decoding, poisoned omitted slots and position-preserving token removal."""
    require(sample["valid"].all(), "mask check requires three real chunks")
    native = predict_embodiment_v2(params, cfg, sample)
    full = decoded(params, cfg, views(sample, None))
    require(all(np.array_equal(a,b) for a,b in zip(native,full[:3],strict=True)), "Published full decoder")
    report = []
    for omit in range(3):
        masked = views(sample, omit)
        y, g, sign, _ = decoded(params, cfg, masked)
        poison = {k:v.copy() for k,v in masked.items()}
        for k, v in poison.items():
            if k != "valid":
                v[:, omit] = 255 if v.dtype == np.uint8 else 123.75
        poisoned = predict_embodiment_v2(params, cfg, poison)
        require(all(np.array_equal(a,b) for a,b in zip((y,g,sign),poisoned,strict=True)), "Omitted input was visible")
        retained = [i for i in range(3) if i != omit]
        compact = {k:v[:,retained] for k,v in sample.items()}
        p = params
        if cfg.model_type == "perceiver_icba":
            # Removing a token shifts the lookup indices. Assign each kept token its original learned position.
            att = params["perceiver"]
            p = dict(params, perceiver=dict(att, pos=att["pos"][:,retained,:]))
        yc, gc, sc = predict_embodiment_v2(p, cfg, compact)
        err = np.abs((y.astype(float) - yc + 180) % 360 - 180)
        # Alternate sequence/batch lengths regroup float32 GEMMs. Compare normalized sine/cosine,
        # retain actual discrepancies for readback. Same-shape poison/native tests above are exact.
        cs = lambda angle: np.exp(1j * np.deg2rad(angle.astype(float)))
        np.testing.assert_allclose(cs(y), cs(yc), atol=2e-4, rtol=0)
        np.testing.assert_allclose(g, gc, rtol=2e-4, atol=2e-6)
        np.testing.assert_array_equal(sign, sc)
        report.append(dict(omit=omit, retained=retained, poison_exact=True,
                           max_compact_yaw_difference_deg=float(err.max()),
                           max_compact_gain_difference=float(np.max(np.abs(g-gc))),
                           different_from_full=int(np.sum(y != full[0]))))
    return dict(native_full_exact=True, n=len(sample["valid"]), folds=report)


def geom_fit(train, seed, steps=3500):
    """The stage1_offline_gate.train_and_audit_v2 fit loop with its target audit removed."""
    cfg = GEOM_CFG
    p = init_reader_v2(jax.random.key(seed), cfg)
    schedule = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=3500, alpha=.05)
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(schedule, weight_decay=1e-4))
    state = optimizer.init(p)

    @jax.jit
    def update(params, opt_state, batch):
        (loss, metrics), grad = jax.value_and_grad(lambda pp: reader_loss_v2(pp, cfg, batch), has_aux=True)(params)
        changes, opt_state = optimizer.update(grad, opt_state, params)
        return optax.apply_updates(params, changes), opt_state, loss, metrics

    rng = np.random.default_rng(seed)
    for step in range(1, steps+1):
        idx = rng.integers(len(train["valid"]), size=64)
        batch = {k:jnp.asarray(v[idx]) for k,v in train.items()}
        p, state, loss, _ = update(p, state, batch)
        if step % 500 == 0:
            print(f"{GEOM} seed {seed} step {step} loss {float(loss):.6f}", flush=True)
    return p, float(loss)


def verify_geom_update(train):
    """Compare the extracted loop with the original CPU fit, before the original dev audit begins."""
    from unittest.mock import patch
    import stage1_offline_gate as original
    class ReachedAudit(Exception):
        pass
    captured = []
    def capture(params, config, inputs):
        captured.append(params)
        raise ReachedAudit
    sample = {k:v[:8] for k,v in train.items()}
    schedule = optax.cosine_decay_schedule(1e-3, 3500, alpha=.05)
    with patch.object(original, "predict_embodiment_v2", capture):
        try:
            original.train_and_audit_v2("passive_attempt1", GEOM, GEOM_CFG, sample, sample, [],
                                        np.arange(8), seeds=(41,), steps=1, batch_size=64,
                                        ckpt_dir=None, nominal_gain=None, archive_sha="", lr_schedule=schedule)
        except ReachedAudit:
            pass
    p, loss = geom_fit(sample, 41, steps=1)
    require(len(captured)==1 and jax.tree.structure(p)==jax.tree.structure(captured[0]), "Original update not captured")
    for a, b in zip(jax.tree.leaves(p), jax.tree.leaves(captured[0]), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    return dict(original_one_update_exact=True, leaves=len(jax.tree.leaves(p)), batch=64, loss=loss)


def load(path, fields):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k] for k in fields}


def checked_file(data, seal, name):
    path = data / name
    require(sha(path) == seal["sha256"][name], f"Source input hash mismatch: {name}")
    return path


def checked_train(data, seal, model):
    path = checked_file(data, seal, f"train-{model}.npz")
    ids = load(path, ("task","case_id"))
    require(set(ids["task"]) == set(tasks_by_role()["readerfit"]), "Unexpected training tasks")
    require(len(set(ids["case_id"])) == len(ids["task"]) == (732 if model=="tuned" else 672), path)
    return load(path, TRAIN_FIELDS)


def historic(model, seed):
    lockdir = CODE_OUTPUTS / "symmeta-perceiver-retry-20261123-r1"
    tag = TUNED if model=="tuned" else GEOM
    name = f"{tag}-passive_attempt1-seed{seed}.msgpack"
    if model=="tuned":
        path = ROOT / lockdir.name / "train02pro/checkpoints" / name
        receipt_path = path.parents[1] / "training-sanity.json"
        require(sha(receipt_path)=="9b5232526159a9395ff65cbbee39572a6c62578d642abd008452f4c8dc5368ff", receipt_path)
        receipt = read_json(receipt_path)
        digest, = [r["checkpoint_sha256"] for r in receipt["seeds"] if r["seed"]==seed]
    else:
        path = ROOT / "xemb-icml-campaign-20261121-r1/gpu03v2/stage1/checkpoints" / name
        digest = read_json(lockdir / "export-inputs.json")["sha256"][path.relative_to(ROOT).as_posix()]
    require(sha(path)==digest, path)
    return load_v2_checkpoint(path)


def qualify(data, out, seal):
    train = checked_train(data, seal, "geom")
    report = {"geom_recipe":verify_geom_update(train), "models":{}}
    sample = {k:train[k][[0,1,200,600]] for k in FEATURES}
    # Only reader-fit cases, never a held-out prediction with overlapping historical checkpoints.
    for reader in ("tuned","geom"):
        p,c,meta = historic(reader,41)
        require(c == (TUNED_SPEC[2] if reader=="tuned" else GEOM_CFG) and meta["seed"]==41, reader)
        report["models"][reader] = inspect_masks(p,c,sample)
        block = {k:v[:32] for k,v in train.items() if k in FEATURES}
        decoded(p,c,block)
        t = time.monotonic()
        for omit in (None,0,1,2):
            decoded(p,c,views(block,omit))
        report["models"][reader]["seconds_32_traces_all_four_views"] = time.monotonic()-t
    write_json(out / "qualification.json", report)
    print(report, flush=True)


def refit(data, out, pin, seal):
    training = {r:checked_train(data,seal,r) for r in ("tuned","geom")}
    paths = []
    for model in training:
        fields = training[model]
        for seed in SEEDS:
            t = time.monotonic()
            if model=="tuned":
                tag, head, cfg, families, lr, warm, steps = TUNED_SPEC
                require((head,cfg.max_chunks,families,lr,warm,steps)==("factors",24,("family",),3e-4,200,2500),
                        "PR7 recipe changed")
                bsz = 32
                p, loss = pr7.train_model(fields,head,cfg,families,lr,warm,steps,seed,bsz)
            else:
                tag,cfg,steps,bsz = GEOM,GEOM_CFG,3500,64
                p, loss = geom_fit(fields,seed)
            require(np.isfinite(loss), f"Nonfinite loss: {tag} seed {seed}")
            path = out / f"{tag}-passive_attempt1-seed{seed}.msgpack"
            payload = dict(params=p, reader_v2_config=asdict(cfg), mode="passive_attempt1", tag=tag,
                           seed=seed, steps=steps, batch_size=bsz, loss=loss, revision=pin,
                           train_sha256=seal["sha256"][f"train-{model}.npz"], fit_tasks=tasks_by_role()["readerfit"])
            with path.open("xb") as f:
                f.write(serialization.to_bytes(payload))
            restored,c,meta = load_v2_checkpoint(path)
            require(c==cfg and meta["tag"]==tag and meta["seed"]==seed, path)
            # Compare checkpoint leaves exactly; qualification compares decoded factors independently.
            for a,b in zip(jax.tree.leaves(p),jax.tree.leaves(restored),strict=True):
                np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
            paths.append(dict(model=model,seed=seed,path=path.name,sha256=sha(path),loss=loss,steps=steps,
                              batch_size=bsz,seconds=time.monotonic()-t))
            print("Restored exact:", paths[-1], flush=True)
    write_json(out / "checkpoints.json", dict(revision=pin,data_seal=sha(data / "source-seal.json"),rows=paths))


def predict(data, out, ckpts, pin, seal):
    ck = read_json(ckpts / "checkpoints.json")
    require(ck["revision"]==pin and ck["data_seal"]==sha(data/"source-seal.json"), "Unmatched checkpoints")
    parts = []
    # Model code never opens the label files or sees task/arm/case IDs.
    for split in ("cal","eval"):
        path = checked_file(data,seal,f"reader-{split}.npz")
        with np.load(path,allow_pickle=False) as z:
            require(set(z.files)==set((*FEATURES,*IDS)), f"Unexpected reader input {path}")
        part = load(path, (*FEATURES,*IDS))
        require(set(part["task"])==set(tasks_by_role()[split]) and set(part["split"])=={split}, split)
        parts.append(part)
    panel = {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    n = len(panel["valid"])
    require(n==672 and panel["valid"].all() and len(set(panel["case_id"]))==n, "Incomplete predictions")
    full,loo,gains,signs,fv,lv = [],[],[],[],[],[]
    diagnostics = []
    for seed in SEEDS:
        model_preds = []
        for model in ("tuned","geom"):
            r, = [r for r in ck["rows"] if (r["model"],r["seed"])==(model,seed)]
            path = ckpts / r["path"]
            require(sha(path)==r["sha256"], path)
            p,c,meta = load_v2_checkpoint(path)
            require(meta["revision"]==pin and meta["seed"]==seed and
                    meta["train_sha256"]==seal["sha256"][f"train-{model}.npz"], path)
            diagnostics.append(dict(seed=seed,model=model,**inspect_masks(p,c,{k:panel[k][[0,223,224,671]] for k in FEATURES})))
            masks = []
            for omit in (None,0,1,2):
                blocks = []
                size = 1 if model == "geom" else 96
                for start in range(0,n,size):
                    b = {k:panel[k][start:start+size] for k in FEATURES}
                    blocks.append(decoded(p,c,views(b,omit)))
                masks.append(tuple(np.concatenate(a) for a in zip(*blocks,strict=True)))
            model_preds.append(masks)
        full.append(np.stack([m[0][0] for m in model_preds],-1))
        fv.append(np.stack([m[0][3] for m in model_preds],-1))
        loo.append(np.stack([np.stack([m[i][0] for i in (1,2,3)],1) for m in model_preds],-1))
        lv.append(np.stack([np.stack([m[i][3] for i in (1,2,3)],1) for m in model_preds],-1))
        gains.append(model_preds[0][0][1])
        signs.append(model_preds[0][0][2])
        print("Completed four views of", n, "traces on both readers, seed", seed, flush=True)
    arrays = {k:np.tile(panel[k],len(SEEDS)) for k in IDS}
    arrays.update(seed=np.repeat(SEEDS,n), cmd=np.tile(panel["issued_mean"][...,:2],(len(SEEDS),1,1)),
                  valid=np.tile(panel["valid"],(len(SEEDS),1)), tuned_gain=np.concatenate(gains),
                  tuned_sign=np.concatenate(signs), full_yaw=np.concatenate(full), loo_yaw=np.concatenate(loo),
                  full_valid=np.concatenate(fv), loo_valid=np.concatenate(lv))
    require(arrays["full_yaw"].shape==(2016,2) and arrays["loo_yaw"].shape==(2016,3,2), "Output schema")
    save(out,"readers.npz",arrays)
    write_json(out/"prediction-receipt.json",dict(revision=pin,checkpoints=sha(ckpts/"checkpoints.json"),
               data_seal=ck["data_seal"],reader_sha256=sha(out/"readers.npz"),masks=diagnostics,
               rows=len(arrays["seed"]),physical_traces=n,
               invalid_full=int((~arrays["full_valid"]).sum()),invalid_loo=int((~arrays["loo_valid"]).sum())))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("qualify","refit","predict"))
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("pin")
    ap.add_argument("--checkpoints",type=Path)
    a = ap.parse_args()
    cv2.setNumThreads(1)
    seal = read_json(a.data/"source-seal.json")
    require(seal["revision"]==a.pin and seal["roles"]==tasks_by_role(), "Source pin/split changed")
    if a.mode=="qualify":
        require(jax.default_backend()=="cpu", "Expected CPU diagnostic")
        qualify(a.data,a.out,seal)
    elif a.mode=="refit":
        probe = read_json(a.out/"gpu.json")
        require(jax.default_backend()=="gpu" and len(jax.devices())==1 and probe["status"]=="passed"
                and probe["exclusive_training_admission"] and len(probe["devices"])==1, "GPU admission")
        name = probe["devices"][0]["cuda_name"].upper()
        require(name in ("NVIDIA GEFORCE RTX 4090","NVIDIA RPBZZZ6") or "RTX PRO 6000" in name,name)
        refit(a.data,a.out,a.pin,seal)
    else:
        require(a.checkpoints is not None, "Missing refit checkpoint directory")
        predict(a.data,a.out,a.checkpoints,a.pin,seal)


if __name__ == "__main__":
    main()
