"""Allocated-runtime conversion; use the released RLinf layout transforms."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024**2),b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    import torch
    from safetensors.torch import load_file,save_file
    from rlinf.utils.ckpt_convertor.openpi.openpi_pytorch_to_openpi import old_to_new_state_dict
    from fluffy_bassoon.rl.sidney import REVISION
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    assert os.uname().nodename.startswith('job-') and torch.cuda.is_available()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    original=load_file(str(args.source/'model.safetensors'))
    assert len(original)==813 and all(k.startswith('model.') for k in original)
    source={k.removeprefix('model.'):v for k,v in original.items()}
    converted=old_to_new_state_dict(source)
    embedding_key='paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight'
    actual_embedding=source[embedding_key]
    assert actual_embedding.shape==(257152,2048)
    assert not torch.equal(actual_embedding,source['paligemma_with_expert.paligemma.lm_head.weight'])
    converted['llm.embedder.embedding.weight']=actual_embedding
    stats=load_file(str(args.source/'policy_preprocessor_step_3_normalizer_processor.safetensors'))
    post=load_file(str(args.source/'policy_postprocessor_step_0_unnormalizer_processor.safetensors'))
    assert torch.equal(stats['action.mean'],post['action.mean'])
    assert torch.equal(stats['action.std'],post['action.std'])
    metadata=dict(revision=REVISION,normalization='MEAN_STD',action_convention='14 absolute ALOHA joints',
        state_mean=stats['observation.state.mean'].tolist(),state_std=stats['observation.state.std'].tolist(),
        action_mean=stats['action.mean'].tolist(),action_std=stats['action.std'].tolist(),
        source_checkpoint_sha256='9a5381c3260fc58fdb4b90d17b93f9090bcbfbcd8e0287f9bf2acea2477a2bc3',
        embedding_source=embedding_key,full_vision_precision='float32')
    (args.output/'sidney-processors.json').write_text(json.dumps(metadata,indent=2)+'\n')
    save_file({k:v.contiguous() for k,v in converted.items()},str(args.output/'model.safetensors'))
    receipt=dict(status='converted',revision=REVISION,source=str(args.source),output=str(args.output),
        converter='pinned RLinf old_to_new_state_dict plus actual input embedding override',
        source_tensors=len(source),output_tensors=len(converted),
        output_parameters=sum(v.numel() for v in converted.values()),
        model_sha256=sha(args.output/'model.safetensors'),
        processors_sha256=sha(args.output/'sidney-processors.json'))
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
