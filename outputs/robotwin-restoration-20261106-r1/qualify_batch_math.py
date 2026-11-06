"""Localize batch drift: native vs port, native repeat and preprocessing.

Frozen numerical diagnosis on actual saved source observations, not a success
evaluation or a change to the prior singleton-parity gate. No simulator acts.
"""
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

from qualify_motor import BASE, MOTOR, READOUT, save, sha


def main():
    import numpy as np
    import torch
    from rlinf.models import get_model
    from fluffy_bassoon.rl.robotwin_motor import FrozenRoboTwinMotor,case_noise,stack_public_observations
    from fluffy_bassoon.rl.robotwin_noise import model_config

    root,code = Path(sys.argv[1]),Path(os.environ['TASK_CODE'])
    assert sys.platform == 'linux' and os.uname().nodename.startswith('job-')
    assert root.is_dir() and not (root/'RESULT.json').exists() and torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    contract = json.loads((code/'outputs/robotwin-restoration-20261106-r1/BATCHMATH01-CONTRACT.json').read_text())
    previous = BASE/'outputs/robotwin-restoration-20261106-r1/motor03'
    assert json.loads((previous/'RESULT.json').read_text())['status'] == 'complete'
    assert json.loads((previous/'EXIT.json').read_text())['exit_code'] == 0
    manifest = json.loads((previous/'INPUTS.json').read_text())
    assert Path(manifest['root']) == READOUT
    norm = MOTOR/'assets/pi0.5_clean_randomize_joint_training/norm_stats.json'
    assert sha(MOTOR/'model.safetensors') == contract['model_sha256']
    assert sha(norm) == contract['norm_stats_sha256']
    report = dict(status='starting',source_commit=os.environ['TASK_SOURCE'],optimizer_updates=0,
        simulator_actions=0,model_sha256=contract['model_sha256'],norm_stats_sha256=contract['norm_stats_sha256'],
        input_manifest_sha256=sha(previous/'INPUTS.json'),batches=[],started_unix=time.time(),
        protocol='Same native H32/K5 ODE, weights, precision and explicit case noise as motor03; no tolerance relaxation')
    save(root/'RESULT.json',report)

    def snapshot(observation):
        result = {key:getattr(observation,key)[:1].detach().cpu().clone()
                  for key in ('state','tokenized_prompt','tokenized_prompt_mask')}
        for key,value in observation.images.items(): result['image/'+key] = value[:1].detach().cpu().clone()
        for key,value in observation.image_masks.items(): result['image_mask/'+key] = value[:1].detach().cpu().clone()
        return result

    try:
        model = get_model(model_config(os.environ['REPO_PATH'],MOTOR,norm,action_horizon=32,num_steps=5)).cuda()
        port = FrozenRoboTwinMotor(model)
        report['parameter_elements_by_dtype'] = dict(Counter({dtype:sum(p.numel() for p in model.parameters()
            if str(p.dtype)==dtype) for dtype in {str(p.dtype) for p in model.parameters()}}))
        report['torch_flags'] = dict(tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
            bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            flash_sdp=torch.backends.cuda.flash_sdp_enabled(),math_sdp=torch.backends.cuda.math_sdp_enabled(),
            memory_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled())
        rows,identities = [],manifest['paths'][:64]
        for relative in identities:
            path=READOUT/relative
            with np.load(path,allow_pickle=False) as data:
                images=data['images'].copy();state=data['state'].copy();prompt=str(data['instruction'].item())
            rows.append(dict(main_images=torch.from_numpy(images[:1]),wrist_images=torch.from_numpy(images[None,1:]),
                states=torch.from_numpy(state[None]).float(),task_descriptions=[prompt]))
        noise = case_noise([f'profile/{s}' for s in identities],seed=20261106,horizon=32,
                          action_dim=32,device=model.device)
        singleton = []
        for i,raw in enumerate(rows):
            assert time.time() < float(os.environ['TASK_DEADLINE'])-60
            with torch.no_grad(): native,_ = model.predict_action_batch(raw,mode='eval',noise=noise[i:i+1])
            singleton.append(native.detach().cpu())
        singleton = torch.cat(singleton)
        first_preprocessed,first_feature = None,None
        for batch in contract['batch_candidates']:
            assert time.time() < float(os.environ['TASK_DEADLINE'])-60
            raw = stack_public_observations(rows[:batch]);draw = noise[:batch]
            with torch.no_grad():
                native,_ = model.predict_action_batch(raw,mode='eval',noise=draw)
                repeat,_ = model.predict_action_batch(raw,mode='eval',noise=draw)
            features,token = port.prepare_batch(raw)
            prepared = snapshot(token.observation)
            actions = port.propose_batch(token,noise=draw)
            row = dict(batch=batch,native_vs_port_max_abs=float((native-actions).abs().max()),
                native_repeat_max_abs=float((native-repeat).abs().max()),
                native_vs_singleton_max_abs=float((native.cpu()-singleton[:batch]).abs().max()))
            if first_preprocessed is None:
                first_preprocessed,first_feature = prepared,features[:1].cpu().clone()
            row['first_row_preprocessing_exact'] = {key:torch.equal(value,first_preprocessed[key])
                                                    for key,value in prepared.items()}
            row['first_row_feature_max_abs_vs_singleton'] = float((features[:1].cpu()-first_feature).abs().max())
            row['native_adapter_parity'] = row['native_vs_port_max_abs'] <= 1e-5
            row['native_repeat_exact'] = row['native_repeat_max_abs'] == 0
            report['batches'].append(row)
            np.savez_compressed(root/f'batch-{batch:03d}.npz',native=native.cpu().numpy(),
                adapter=actions.cpu().numpy(),singleton=singleton[:batch].numpy(),features=features.cpu().numpy())
            save(root/'RESULT.json',report);print(json.dumps(row),flush=True)
            del token,native,repeat,features,actions
        port.check_frozen()
        report.update(status='complete',native_batch_adapter_parity=all(r['native_adapter_parity'] for r in report['batches']),
            native_batch_repeat_exact=all(r['native_repeat_exact'] for r in report['batches']),
            first_row_preprocessing_exact=all(all(r['first_row_preprocessing_exact'].values()) for r in report['batches']),
            frozen_parameter_and_buffer_versions_unchanged=True)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        report['ended_unix']=time.time();save(root/'RESULT.json',report)


if __name__ == '__main__': main()
