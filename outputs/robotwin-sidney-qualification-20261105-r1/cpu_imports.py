"""Allocated CPU import and saved-processor checks; never construct a model."""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-visible-gpus', type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='started', model_constructed=False, simulator_episodes=0,
                  optimizer_updates=0, model_forward_calls=0,
                  started_unix=time.time())
    try:
        import inspect
        import numpy as np
        import torch
        import transformers
        from safetensors.torch import load_file
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.factory import make_pre_post_processors

        assert torch.cuda.device_count() == args.expected_visible_gpus
        report['visible_gpu_count'] = args.expected_visible_gpus
        source = Path('/mowenjie/fluffy-bassoon-2/sources/lerobot-sidney-92f96f33-20261105-r1')
        assert Path(inspect.getfile(PI05Policy)).is_relative_to(source)
        model = Path('/mowenjie/fluffy-bassoon-2/cache/robotwin-sidney-qualification-20261105-r1')
        tokenizer = Path('/mowenjie/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c')
        cfg = PI05Config.from_pretrained(str(model))
        assert cfg.chunk_size == cfg.n_action_steps == 50
        assert cfg.num_inference_steps == 10 and not cfg.use_relative_actions
        cfg.device = 'cpu'
        pre, post = make_pre_post_processors(cfg, pretrained_path=str(model),
            preprocessor_overrides={'device_processor': {'device': 'cpu'},
                                   'tokenizer_processor': {'tokenizer_name': str(tokenizer)}})
        stats = load_file(str(model / 'policy_preprocessor_step_3_normalizer_processor.safetensors'))
        state = torch.arange(14, dtype=torch.float32)/10
        batch = {'observation.state': state, 'task': 'place the phone on the stand'}
        for key in ('cam_high', 'cam_left_wrist', 'cam_right_wrist'):
            batch[f'observation.images.{key}'] = torch.zeros(3, 480, 640)
        normalized = pre(batch)
        expected = (state-stats['observation.state.mean'])/(stats['observation.state.std']+1e-8)
        torch.testing.assert_close(normalized['observation.state'].reshape(14), expected, atol=1e-5, rtol=1e-5)
        action = torch.linspace(-1, 1, 50*14).reshape(1, 50, 14)
        output = post(action)
        expected_action = action*(stats['action.std']+1e-8)+stats['action.mean']
        torch.testing.assert_close(output, expected_action, atol=1e-6, rtol=1e-6)
        report.update(status='passed', torch=torch.__version__, numpy=np.__version__,
                      transformers=transformers.__version__, policy_source=inspect.getfile(PI05Policy),
                      state_normalization_max_abs=float((normalized['observation.state'].reshape(14)-expected).abs().max()),
                      action_unnormalization_max_abs=float((output-expected_action).abs().max()),
                      token_shape=list(normalized['observation.language.tokens'].shape))
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(ended_unix=time.time())
        (args.output/'RESULT.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
