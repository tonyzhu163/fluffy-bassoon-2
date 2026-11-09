"""Reconstruct a frozen Sidney actor in the original LeRobot inference layout."""
import argparse
import gc
import os
from pathlib import Path
import subprocess
import sys

import torch
from safetensors.torch import load_file, save_file

ROOT = Path('/mowenjie/fluffy-bassoon-2')
BOS = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')
RLINF = ROOT/'sources/robotwin-baseline-smoke-20261028-r1/RLinf'
BASE_HASH = 'c71114641cf0819900f8b1d7b0568158a12e631b2ca08b2b19bc2d5b761b69b3'
EXPORT_HASH = {'ppo40': '043fa7435b8face4998aef631d96caef55808c99a3ca6867a3625157c84e2838',
               'grpo40': 'ee68ab24cc0da7586374725ae2a0d8eade9e241c238cf76a4dfd7c521578cefe'}
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'robotwin-motus-release-baseline-20261109-r1'))
from evaluate import save, sha


def prepare(arm, original, dest, receipt):
    assert os.uname().nodename.startswith('job-') and torch.cuda.device_count() == 1
    assert subprocess.check_output(['git', '-C', RLINF, 'rev-parse', 'HEAD'], text=True).strip() == 'c605b9d55ed93a180119b099e6f836abff293b08'
    assert not subprocess.check_output(['git', '-C', RLINF, 'status', '--porcelain', '-uno'])
    sys.path.insert(0, str(RLINF))
    from rlinf.utils.ckpt_convertor.openpi.openpi_to_openpi_pytorch import new_to_old_state_dict
    from rlinf.utils.ckpt_convertor.openpi.openpi_pytorch_to_openpi import old_to_new_state_dict

    trained_path = BOS/'exports01'/arm.removesuffix('40')/'policy.pt'
    base_path = ROOT/'cache/robotwin-sidney-rlinf-20261105-r1/model.safetensors'
    assert sha(trained_path) == EXPORT_HASH[arm] and sha(base_path) == BASE_HASH
    trained = torch.load(trained_path, map_location='cpu', weights_only=True)
    base = load_file(str(base_path))
    extra = trained.keys() - base.keys()
    assert len(base) == 667 and base.keys() <= trained.keys()
    assert len(extra) == (8 if arm == 'ppo40' else 0) and all(k.startswith('value_head.') for k in extra)
    changes = {}
    for key, value in base.items():
        actual = trained[key]
        assert value.shape == actual.shape
        if not torch.equal(actual, value.to(actual.dtype)):
            changes[key] = actual
    assert changes and 'llm.embedder.embedding.weight' not in changes
    replacement = new_to_old_state_dict(changes)
    assert replacement
    mapped_back = old_to_new_state_dict(replacement)
    assert mapped_back.keys() == changes.keys()
    assert all(mapped_back[k].dtype == v.dtype and torch.equal(mapped_back[k], v)
               for k, v in changes.items())
    del mapped_back, base
    gc.collect()

    native = load_file(str(original/'model.safetensors'))
    assert len(native) == 813
    for key, value in replacement.items():
        key = 'model.'+key
        assert key in native and value.shape == native[key].shape
        native[key] = value.contiguous()
    dest.mkdir()
    for p in original.iterdir():
        if p.is_file() and p.name != 'model.safetensors':
            (dest/p.name).symlink_to(p)
    output = dest/'model.safetensors'
    save_file(native, str(output))
    del native, replacement
    gc.collect()

    # Read the saved native file and encode every tensor back to the export layout.
    # Sidney's input embedding and output head are separate in the original file.
    reopened = {k.removeprefix('model.'): v for k,v in load_file(str(output)).items()}
    converted = old_to_new_state_dict(reopened)
    converted['llm.embedder.embedding.weight'] = reopened['paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight']
    assert converted.keys() == trained.keys() - extra
    for k, v in converted.items():
        actual = trained[k]
        assert actual.shape == v.shape and torch.equal(v.to(actual.dtype), actual), k
    save(receipt, dict(arm=arm, adam_step=40, source_export_sha256=EXPORT_HASH[arm],
                       base_port_sha256=BASE_HASH, output_sha256=sha(output),
                       exact_roundtrip_policy_tensors=len(converted), original_layout_tensors=len(reopened),
                       changed_port_tensors=len(changes), omitted_critic_tensors=sorted(extra),
                       extra_optimizer_steps=0))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('arm', choices=EXPORT_HASH)
    for arg in ('original', 'dest', 'receipt'):
        p.add_argument(arg, type=Path)
    args = p.parse_args()
    prepare(args.arm, args.original, args.dest, args.receipt)
