"""Use the released batch-one WebSocket server with the original JAX policy."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--port', type=int, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import jax
    import numpy as np
    from openpi.training import config
    from openpi.policies import policy_config
    from openpi.shared import normalize
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from reference_eval import with_checkpoint_stats

    devices = jax.devices()
    assert len(devices) == 1 and devices[0].platform == 'gpu'
    stats = normalize.load(args.checkpoint / 'assets')
    cfg = with_checkpoint_stats(config.get_config('pi05_pretrain_human300'), stats)
    policy = policy_config.create_trained_policy(cfg, args.checkpoint,
        sample_kwargs={'num_steps': 10}, norm_stats=stats)
    calls = 0

    def save_state(*_):
        value = dict(calls=calls, rng_key=np.asarray(jax.random.key_data(policy._rng)).tolist(),
                     unix=time.time(), original_checkpoint=str(args.checkpoint),
                     normalization_sha256=hashlib.sha256((args.checkpoint/'assets/norm_stats.json').read_bytes()).hexdigest(),
                     training_updates=0)
        (args.output/'SAMPLER-STATE.json').write_text(json.dumps(value, indent=2)+'\n')

    class CountedPolicy:
        def infer(self, obs):
            nonlocal calls
            result = policy.infer(obs)
            calls += 1
            return result

    def stop(signum, frame):
        save_state()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGUSR1, save_state)
    (args.output/'READY.json').write_text(json.dumps(dict(devices=[str(d) for d in devices],
        batch_size=1, port=args.port, training_updates=0, unix=time.time()))+'\n')
    try:
        WebsocketPolicyServer(CountedPolicy(), host='127.0.0.1', port=args.port).serve_forever()
    finally:
        save_state()


if __name__ == '__main__':
    main()
