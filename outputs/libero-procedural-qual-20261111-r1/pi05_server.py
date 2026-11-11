"""The release's JAX policy and websocket server, restricted to loopback."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8027)
    args = parser.parse_args()
    import jax
    import numpy as np
    from openpi.policies import policy_config
    from openpi.training import config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    devices = jax.devices()
    assert len(devices) == 1 and devices[0].platform == 'gpu', devices
    # A synchronized accelerator computation checks this process, in addition
    # to the required job-level Torch probe in the simulator environment.
    import jax.numpy as jnp
    got = np.asarray((jnp.arange(16).reshape(4, 4) @ jnp.eye(4)).block_until_ready())
    np.testing.assert_array_equal(got, np.arange(16).reshape(4, 4))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'jax-device.json').write_text(json.dumps({
        'jax_version': jax.__version__, 'devices': [str(d) for d in devices],
        'device_kind': devices[0].device_kind, 'synchronized_matmul': True,
    }, indent=2) + '\n')
    policy = policy_config.create_trained_policy(
        config.get_config('pi05_libero_gen_spatial_combination_lora'), args.checkpoint)
    print('PI05_POLICY_LOADED', flush=True)
    WebsocketPolicyServer(policy, host='127.0.0.1', port=args.port,
                          metadata={'config': 'pi05_libero_gen_spatial_combination_lora'}).serve_forever()


if __name__ == '__main__':
    main()
