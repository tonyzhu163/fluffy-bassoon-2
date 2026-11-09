"""Direct reference versus actual RLinf RL models on frozen real SFT inputs."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SFT = HERE.parent/'robotwin-motus-release-baseline-20261109-r1'
sys.path.insert(0, str(SFT))
from evaluate import configuration, sha, save


def fixtures():
    cases = json.loads((SFT/'PHONE100-CASES.json').read_bytes())
    root = Path(os.environ['TASK_FIXTURES'])
    selected = []
    for domain in ('demo_clean', 'demo_randomized'):
        for row in cases['settings'][domain][:8]:
            path = root/row['initial_input_path']
            assert sha(path) == row['initial_input_sha256']
            with np.load(path, allow_pickle=False) as value:
                item = dict(state=value['state'].copy(), prompt=str(value['prompt']),
                    images={key:value[key].copy() for key in
                            ('cam_high','cam_left_wrist','cam_right_wrist')})
            assert item['prompt'] == row['instruction']
            selected.append((domain, row, item))
    assert len(selected) == 16
    return selected


def public(item):
    return dict(states=item['state'][None].copy(),
        main_images=item['images']['cam_high'].transpose(1,2,0)[None].copy(),
        wrist_images=np.stack([item['images'][key].transpose(1,2,0)
            for key in ('cam_left_wrist','cam_right_wrist')])[None],
        task_descriptions=[item['prompt']])


def compare(left, right, path=''):
    if isinstance(left, dict):
        assert set(left) == set(right), (path, set(left), set(right))
        return [row for key in left for row in compare(left[key],right[key],path+'/'+key)]
    if torch.is_tensor(right): right=right.detach().cpu().numpy()
    left, right = np.asarray(left), np.asarray(right)
    assert right.shape == (1,*left.shape), (path,left.shape,right.shape)
    right=right[0]
    assert left.dtype == right.dtype and np.array_equal(left,right), (path,left.dtype,right.dtype)
    return [dict(key=path,shape=list(left.shape),dtype=str(left.dtype),exact=True)]


def config(rlinf, model, critic):
    from fluffy_bassoon.rl.robotwin_noise import model_config
    cfg=model_config(rlinf,model,model/'assets/pi0.5_clean_randomize_joint_training/norm_stats.json',
        action_horizon=32,num_steps=10)
    cfg.openpi.task='rl';cfg.add_value_head=critic
    cfg.openpi.noise_method='flow_sde';cfg.openpi.noise_level=.3
    cfg.openpi.train_expert_only=True;cfg.openpi.value_after_vlm=critic
    cfg.openpi.detach_critic_input=True
    return cfg


def main(args):
    from openpi import transforms
    from rlinf.models.embodiment.openpi.transforms.pipeline import build_openpi_transforms
    from rlinf.models.embodiment.openpi.transforms.apply import apply_input_transform
    from rlinf.models.embodiment.openpi.transforms.env import repack_env_obs
    rlinf=Path(os.environ['REPO_PATH']);model=Path(os.environ['NATIVE_PPO_MODEL'])
    assert sys.platform=='linux'
    assert subprocess.check_output(['git','-C',str(rlinf),'rev-parse','HEAD'],text=True).strip() == 'c605b9d55ed93a180119b099e6f836abff293b08'
    assert not subprocess.check_output(['git','-C',str(rlinf),'status','--porcelain'],text=True).strip()
    baseline=json.loads((SFT/'PHONE100-READBACK.json').read_bytes())
    assert baseline['status']=='passed' and baseline['frozen'] and baseline['optimizer_updates']==0
    assert sha(model/'model.safetensors') == json.loads((SFT/'PHONE100-CASES.json').read_bytes())['model_sha256']
    data=configuration(model).data.create(configuration(model).assets_dirs,configuration(model).model)
    reference_input=transforms.compose([transforms.InjectDefaultPrompt(None),*data.data_transforms.inputs,
        transforms.Normalize(data.norm_stats,use_quantiles=data.use_quantile_norm),*data.model_transforms.inputs])
    native_inputs,_=build_openpi_transforms(str(model),'pi05_aloha_robotwin',
        data_kwargs=dict(norm_stats_path=str(model/'assets/pi0.5_clean_randomize_joint_training/norm_stats.json'),
                         adapt_to_pi=True,extra_delta_transform=True))
    native_input=transforms.compose(native_inputs)
    rows=fixtures()
    report=dict(status='running',source_commit=os.environ['TASK_SOURCE'],optimizer_updates=0,
        baseline_result_sha256=baseline['result_sha256'],model_sha256=sha(model/'model.safetensors'),
        rlinf_commit='c605b9d55ed93a180119b099e6f836abff293b08',
        rlinf_model_files={str(p.relative_to(rlinf)):sha(p) for p in sorted((rlinf/'rlinf/models/embodiment/openpi').rglob('*.py'))},
        mode='CPU preprocessing' if args.check_only else 'GPU model parity',
        selection='first8 initial inputs in each domain; independent of success',
        batch_size=1,action_horizon=32,denoising_steps=10,action_absolute_tolerance=.001,
        preprocessing=[],algorithms={},started_unix=time.time())
    for domain,row,item in rows:
        processed=reference_input(copy.deepcopy(item))
        repacked=repack_env_obs('pi05_aloha_robotwin',public(item),select_state=lambda x:x)
        actual=apply_input_transform(native_input,repacked,transpose=False)
        report['preprocessing'].append(dict(domain=domain,index=row['index'],seed=row['seed'],
            fields=compare(processed,actual)))
    report['preprocessing_exact']=True
    save(args.output/'RESULT.json',report)
    if args.check_only:
        assert not torch.cuda.is_initialized()
        report.update(status='passed',model_constructed=False,ended_unix=time.time())
        save(args.output/'RESULT.json',report)
        return
    assert os.uname().nodename.startswith('job-') and torch.cuda.device_count()==1
    torch.cuda.set_device(0)
    from openpi.policies.policy_config import create_trained_policy
    from rlinf.models import get_model
    policy=create_trained_policy(configuration(model),model,pytorch_device='cuda:0',sample_kwargs={'num_steps':10})
    original=getattr(policy._model.sample_actions,'_torchdynamo_orig_callable',None)
    if original is not None:policy._model.sample_actions=original;policy._sample_actions=original
    policy._model.eval().requires_grad_(False)
    reference_versions=[(p,p._version) for p in policy._model.parameters()]
    reference=[]
    for index,(_,_,item) in enumerate(rows):
        noise=np.random.default_rng(20261109+index).standard_normal((32,32)).astype(np.float32)
        with torch.inference_mode():value=policy.infer(copy.deepcopy(item),noise=noise)['actions']
        assert value.shape==(32,14) and np.isfinite(value).all()
        np.save(args.output/f'reference-{index:02d}.npy',value)
        reference.append((noise,value))
    for algorithm in ('grpo','ppo'):
        native=get_model(config(rlinf,model,algorithm=='ppo')).cuda().eval().requires_grad_(False)
        assert native.action_horizon==32 and native.num_steps==10
        assert native.rl_cfg.add_value_head==(algorithm=='ppo')
        versions=[(p,p._version) for p in native.parameters()]
        errors=[]
        for index,((domain,row,item),(noise,expected)) in enumerate(zip(rows,reference,strict=True)):
            with torch.inference_mode():value,_=native.predict_action_batch(public(item),mode='eval',
                noise=torch.from_numpy(noise[None]).cuda())
            value=value[0].cpu().numpy()
            assert value.shape==(32,14) and np.isfinite(value).all()
            np.save(args.output/f'{algorithm}-{index:02d}.npy',value)
            errors.append(dict(domain=domain,index=row['index'],seed=row['seed'],
                max_abs=float(np.max(np.abs(value-expected)))))
        frozen=all(p._version==version and p.grad is None for p,version in versions)
        report['algorithms'][algorithm]=dict(rows=errors,max_abs=max(e['max_abs'] for e in errors),
            frozen=frozen,passed=frozen and max(e['max_abs'] for e in errors)<=.001)
        save(args.output/'RESULT.json',report)
        del native;__import__('gc').collect();torch.cuda.empty_cache()
    report['reference_frozen']=all(p._version==version and p.grad is None for p,version in reference_versions)
    report.update(status='passed' if report['reference_frozen'] and all(x['passed'] for x in report['algorithms'].values()) else 'failed',
        model_constructed=True,ended_unix=time.time())
    save(args.output/'RESULT.json',report)
    assert report['status']=='passed','Direct reference / RLinf action parity failed; hold training'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    assert args.output.is_dir() and not (args.output/'RESULT.json').exists()
    main(args)
