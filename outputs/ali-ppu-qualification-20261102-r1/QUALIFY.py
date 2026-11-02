"""Small synthetic offline qualification; run only on the allocated Ali host."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-hostname', required=True)
    args = parser.parse_args()
    if platform.system() != 'Linux' or socket.gethostname() != args.expected_hostname:
        raise RuntimeError('Require the exact allocated Ali Linux instance')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(hostname=socket.gethostname(), started_unix=time.time(),
                  python=sys.version, synthetic_only=True, simulator_controls=0,
                  scientific_training=False, checks={})

    def save():
        report['updated_unix'] = time.time()
        target = args.output / 'RESULT.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(target)

    def check(name, fn):
        started = time.monotonic()
        try:
            result = fn()
            report['checks'][name] = dict(passed=True, seconds=time.monotonic()-started, result=result)
        except Exception:
            report['checks'][name] = dict(passed=False, seconds=time.monotonic()-started,
                                           error=traceback.format_exc())
        save()
        print(json.dumps({name: report['checks'][name]}), flush=True)

    import numpy as np
    import torch
    report.update(torch=torch.__version__, torch_file=torch.__file__, cuda_version=torch.version.cuda,
                  cuda_available=torch.cuda.is_available(), cpu_count=os.cpu_count())
    save()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('One working allocated accelerator required; CPU fallback does not qualify')
    device = torch.device('cuda:0')
    properties = torch.cuda.get_device_properties(device)
    report['accelerator'] = dict(name=properties.name, memory_bytes=properties.total_memory,
                                 count=torch.cuda.device_count())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = Path(__file__).resolve().parents[2]
    report['source_commit'] = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    save()

    def arithmetic(dtype):
        torch.manual_seed(412)
        value = torch.randn(32, 32).to(dtype)
        reference = value.float() @ value.float().T
        actual = value.to(device).requires_grad_()
        product = actual @ actual.T
        loss = product.float().square().mean()
        loss.backward()
        assert torch.isfinite(product).all() and torch.isfinite(actual.grad).all()
        torch.testing.assert_close(product.float().cpu(), reference, rtol=.02 if dtype==torch.bfloat16 else 2e-4,
                                   atol=.1 if dtype==torch.bfloat16 else 2e-4)
        return dict(dtype=str(dtype), max_absolute_error=float((product.float().cpu()-reference).abs().max()))

    check('fp32_forward_backward', lambda: arithmetic(torch.float32))
    check('bf16_forward_backward', lambda: arithmetic(torch.bfloat16))

    def history(backbone, objective):
        from fluffy_bassoon.meta_rl.vla_history import HistoryConfig, HistoryNoisePolicy, sample_batch, validate_trial
        from fluffy_bassoon.meta_rl.vla_history_train import supervised_update
        rng = np.random.default_rng(413)
        length = 12
        noise = rng.normal(size=(length, 6)).astype(np.float32) * .2
        dones = np.zeros(length, dtype=bool); dones[3::4] = True
        trial = dict(observations=rng.normal(size=(length, 2088)).astype(np.float32),
                     executed_noise=noise, target_noise=noise.copy() if objective=='ad' else noise+.1,
                     rewards=dones.astype(np.float32), dones=dones,
                     episode_starts=np.r_[True, dones[:-1]], teacher_steps=np.repeat(np.arange(3),4).astype(np.int64))
        validate_trial(trial, objective=objective)
        torch.manual_seed(414)
        cpu = HistoryNoisePolicy(HistoryConfig(2088,6,width=128,layers=2,heads=4,context_length=300,backbone=backbone))
        cpu.fit_observation_normalization([trial])
        gpu = deepcopy(cpu).to(device)
        batch = sample_batch([trial], cpu, np.random.default_rng(415), batch_size=2)
        cuda_batch = tuple(value.to(device) for value in batch)
        with torch.no_grad():
            cpu_loss = cpu.sequence_nll(*batch)
            gpu_loss = gpu.sequence_nll(*cuda_batch)
        torch.testing.assert_close(gpu_loss.cpu(), cpu_loss, rtol=2e-4, atol=2e-4)
        optimizer = torch.optim.AdamW(gpu.parameters(),lr=3e-4,weight_decay=0.)
        updates = [supervised_update(gpu,optimizer,cuda_batch,mode='full') for _ in range(3)]
        boundary = args.output / f'{backbone}-{objective}-boundary.pt'
        torch.save(dict(model=gpu.state_dict(),optimizer=optimizer.state_dict(),
                        cuda_rng=torch.cuda.get_rng_state_all(),config=asdict(gpu.config)),boundary)
        original = supervised_update(gpu,optimizer,cuda_batch,mode='full')
        saved = torch.load(boundary,map_location=device,weights_only=False)
        restored = HistoryNoisePolicy(HistoryConfig(**saved['config'])).to(device)
        restored.load_state_dict(saved['model'])
        restored_optimizer = torch.optim.AdamW(restored.parameters(),lr=3e-4,weight_decay=0.)
        restored_optimizer.load_state_dict(saved['optimizer'])
        torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
        continued = supervised_update(restored,restored_optimizer,cuda_batch,mode='full')
        error = max(float((a-b).abs().max()) for a,b in zip(gpu.parameters(),restored.parameters(),strict=True))
        for a,b in zip(gpu.parameters(),restored.parameters(),strict=True):
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        return dict(backbone=backbone,objective=objective,cpu_initial_nll=float(cpu_loss),
                    accelerator_initial_nll=float(gpu_loss),updates=updates,next_update=original,
                    restored_update=continued,checkpoint_next_parameter_max_error=error)

    for backbone in ('decoder','gtrxl'):
        for objective in ('ad','dpt'):
            check(f'{backbone}_{objective}_train_and_resume',lambda b=backbone,o=objective:history(b,o))

    def residual():
        from fluffy_bassoon.meta_rl.residual_ppo import ResidualPPOConfig, ResidualActorCritic, update
        torch.manual_seed(416)
        policy = ResidualActorCritic(ResidualPPOConfig(2088,6,epochs=2,minibatch_size=16,target_kl=.02)).to(device)
        features = torch.randn(24,2088,device=device)
        controls = torch.randn(24,6,device=device)*.2
        with torch.no_grad():
            log_probs,_,values = policy.evaluate(features,controls)
        rollout = dict(features=features[:,None],controls=controls[:,None],log_probs=log_probs[:,None],
                       values=values[:,None],rewards=torch.linspace(0,1,24,device=device)[:,None],
                       ends=torch.ones(24,1,dtype=torch.bool,device=device))
        optimizer = torch.optim.Adam(policy.parameters(),lr=3e-4)
        generator = torch.Generator(device=device).manual_seed(417)
        before = {key:value.clone() for key,value in policy.state_dict().items()}
        metrics = update(policy,optimizer,rollout,generator=generator)
        assert any(not torch.equal(before[key],value) for key,value in policy.state_dict().items())
        assert all(torch.isfinite(value).all() for value in policy.parameters())
        return dict(config=asdict(policy.config),metrics=metrics)

    check('residual_ppo_update',residual)
    torch.cuda.synchronize()
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
    report['completed'] = True
    report['all_passed'] = all(value['passed'] for value in report['checks'].values())
    save()
    raise SystemExit(0 if report['all_passed'] else 1)


if __name__ == '__main__':
    main()
