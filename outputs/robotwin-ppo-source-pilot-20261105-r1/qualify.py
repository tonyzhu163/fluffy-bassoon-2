"""Allocated one-GPU native Motus H50 episode before source PPO."""
import argparse
import json
import os
from pathlib import Path
import time


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    import numpy as np
    from fluffy_bassoon.rl.robotwin_noise import RobotwinNoiseEnv

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rlinf = Path(os.environ['REPO_PATH'])
    model = Path('/mowenjie/fluffy-bassoon-2/cache/robotwin-ppo-pilot-20261105-r1/pi05-motus-effc6e1a-r1')
    norm = model / 'assets/pi0.5_clean_randomize_joint_training/norm_stats.json'
    task = 'place_phone_stand'
    seeds = json.loads((rlinf / 'rlinf/envs/sim/robotwin/seeds/eval_seeds.json').read_text())
    seed = int(seeds[task]['success_seeds'][0])
    started = time.time()
    report = dict(status='started', task=task, requested_seed=seed, horizon=400,
                  action_chunk=50, flow_steps=5, optimizer_updates=0, chunks=[],
                  source_commit=os.environ['TASK_SOURCE'], started_unix=started)
    save(args.output / 'RESULT.json', report)
    env = None
    try:
        env = RobotwinNoiseEnv(model_path=model, norm_stats=norm,
            robotwin_root=os.environ['ROBOTWIN_PATH'], rlinf_root=rlinf,
            episode_limit=400, action_horizon=50, num_steps=5,
            task_config_name='demo_clean', output_dir=args.output / 'native')
        report['contract'] = env.contract
        report['frozen_before'] = env.frozen_fingerprint()
        env.reset(task, seed)
        report['reset'] = env.last_reset_info
        rng = np.random.default_rng(1234)
        report['native_parity'] = env.check_noise_parity(rng.standard_normal(env.noise_shape).astype(np.float32))
        while env.active:
            index = len(report['chunks'])
            raw = env.backend._raw_obs
            images = np.stack([raw['main_images'][0].cpu().numpy(),
                               raw['wrist_images'][0, 0].cpu().numpy(),
                               raw['wrist_images'][0, 1].cpu().numpy()])
            assert all(float(view.std()) > 1 for view in images)
            noise = rng.standard_normal(env.noise_shape).astype(np.float32)
            _, reward, terminated, truncated, info = env.step(noise)
            np.savez_compressed(args.output / f'chunk-{index:02d}.npz', images=images, noise=noise)
            row = dict(chunk=index, reward=reward, terminated=terminated, truncated=truncated,
                       primitive_steps=info['primitive_steps'], success=bool(info['success']),
                       actions_sha256=info['actions_sha256'])
            report['chunks'].append(row)
            save(args.output / 'RESULT.json', report)
            print(json.dumps(row), flush=True)
        report['frozen_after'] = env.frozen_fingerprint()
        assert report['frozen_before'] == report['frozen_after']
        report.update(status='complete', physical_controls=env.elapsed_steps,
                      success=any(x['success'] for x in report['chunks']),
                      frozen_unchanged=True)
    except BaseException as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if env is not None:
            env.close()
        report.update(ended_unix=time.time(), wall_seconds=time.time()-started)
        save(args.output / 'RESULT.json', report)


if __name__ == '__main__':
    main()
