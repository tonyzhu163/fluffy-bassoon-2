"""Strictly load both immutable motors before spending simulator episodes."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--rlinf-root', type=Path, required=True)
    parser.add_argument('--rlinf-loader-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if (sys.platform != 'linux' or
            subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != args.source_commit or
            subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()):
        raise RuntimeError('Require allocated Linux and exact clean source')
    sys.path.insert(0, str(root/'src'))
    from fluffy_bassoon.rl import maniskill_pi05 as ms
    ms.execution_mode()
    loader = args.rlinf_root/'rlinf/models/embodiment/openpi/checkpoint.py'
    if ms.sha256_file(loader) != args.rlinf_loader_sha256:
        raise ValueError('Native loader bytes differ')
    sys.path.insert(0, str(args.rlinf_root))
    import torch
    from safetensors import safe_open
    from rlinf.models import get_model
    from fluffy_bassoon.rl.rlinf_checkpoint_compat import (
        LEGACY_PREFIX, LEGACY_VALUE_PREFIX, NATIVE_PREFIX, NATIVE_VALUE_PREFIX,
        map_legacy_rl_heads)
    base = Path('/mowenjie/fluffy-bassoon/inputs')
    rl = base/'rlinf-pi05-maniskill-rl-a0cbb898-20261028-r1/checkpoints/global_step_150/actor'
    sft = base/'rlinf-pi05-maniskill-sft-eb0e2c90-20261028-r1'
    cases = [('rl', rl, rl/'assets/global_step_150/meta/norm_stats.json',
              '0adb5b40f064760ff245f53ae0a4e078b86680b65f77d1f5991a173543edd5a1'),
             ('sft', sft, sft/'physical-intelligence/maniskill/norm_stats.json',
              '9877be1633a9bc89834d00ef890e1a4cd34a91c155d710b97c92fbab8a5caa89')]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for name, path, norm, expected in cases:
        if ms.sha256_file(path/'model.safetensors') != expected:
            raise ValueError('Immutable motor bytes differ')
        cfg = ms.model_config(path, norm)
        model = get_model(cfg).cuda().eval()
        model.requires_grad_(False)
        target = model.state_dict()
        with safe_open(path/'model.safetensors', framework='pt', device='cpu') as saved:
            extras = {key: saved.get_tensor(key) for key in saved.keys()
                      if key.startswith((LEGACY_PREFIX, LEGACY_VALUE_PREFIX,
                                         NATIVE_PREFIX, NATIVE_VALUE_PREFIX))}
        mapped = map_legacy_rl_heads(extras, target)
        hashes = {}
        for key, tensor in mapped.items():
            actual = target[key].detach().cpu()
            if actual.dtype != tensor.dtype or not torch.equal(actual, tensor):
                raise ValueError('Loaded auxiliary tensor differs from saved bytes: '+key)
            hashes[key] = hashlib.sha256(actual.contiguous().numpy().tobytes()).hexdigest()
        if len(mapped) != (16 if name == 'rl' else 0):
            raise ValueError('Unexpected saved auxiliary head count')
        if bool(cfg.add_value_head) != (name == 'rl') or hasattr(model, 'value_head') != (name == 'rl'):
            raise ValueError('Native critic construction differs from saved checkpoint')
        if hasattr(model, 'noise_head') != (name == 'rl'):
            raise ValueError('Native exploration construction differs from saved checkpoint')
        results.append(dict(name=name, model_sha256=expected,
            strict_load=True, auxiliary_tensors=len(mapped), auxiliary_sha256=hashes,
            add_value_head=bool(cfg.add_value_head), noise_method=cfg.openpi.noise_method,
            native_tensor_count=len(target), motor_fingerprint=ms.model_fingerprint(model)))
        del model, target, extras, mapped
        gc.collect()
        torch.cuda.empty_cache()
        (args.output_dir/'progress.json').write_text(json.dumps(dict(cases=results), indent=2)+'\n')
    result = dict(completed=True, source_commit=args.source_commit, job_id=ms.job_id(),
                  loader_sha256=args.rlinf_loader_sha256, native_episodes=0, cases=results)
    (args.output_dir/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
