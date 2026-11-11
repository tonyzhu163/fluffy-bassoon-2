"""One prompt condition on matched custom layouts, with frozen BPP weights.

Derived from the qualified custom-layout motor. Use separate processes for
each asset root because LIBERO caches its dataset configuration at import.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time

CUSTOM = Path(__file__).resolve().parent.parent / 'libero-gen-custom-layouts-20261111-r1'
sys.path.insert(0, str(CUSTOM))

REFERENCE_PROMPTS = [
    '31f72d8029d396693f588231e6325e437f9632808cce7efe6492080b4e4d729e',
    '43196559472b14d5b4f2da9414ba4ccc22802b30aa19ca1648ac57a94eda55fe',
]


TASKS = [
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_2",
]
SPLIT = "libero_spatial_selected_combinations_inverse_view"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 2**20):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--admission-sha256")
    parser.add_argument("--indices", help="Fixed comma-separated layout indices")
    parser.add_argument('--condition', choices=['original', 'repaired'], required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import numpy as np
    import yaml
    import torch
    import dill
    import hydra
    from omegaconf import OmegaConf
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    random.seed(7301)
    np.random.seed(7301)
    torch.manual_seed(7301)
    base = args.source / "behavior_prompting/train_network"
    init_dir = args.output / "paired_init" / SPLIT
    init_dir.mkdir(parents=True)
    from geometry import sha
    assert all(value is not None for value in [args.geometry, args.admission, args.admission_sha256, args.indices])
    assert digest(args.admission) == args.admission_sha256
    admission = json.loads(args.admission.read_text())
    assert digest(args.geometry / "result.json") == admission["geometry_result_sha256"]
    manifest = json.loads((args.geometry / "result.json").read_text())
    indices = [int(x) for x in args.indices.split(",")]
    assert indices == [0, 1], 'This repair compares the demo scene and its frozen nearby scene'
    state_bank = []
    for i in indices:
        state = np.load(args.geometry / f"layout-{i:02d}/state.npy", allow_pickle=False)
        assert sha(state) == manifest["layouts"][i]["state_sha256"]
        state_bank.append(state)
    for task in TASKS:
        torch.save(np.stack(state_bank), init_dir / (task + ".pruned_init"))
    config_dir = args.output / "libero_config"
    config_dir.mkdir()
    config = {"benchmark_root": str(args.source / "deps/LIBERO/libero/libero"),
              "assets": str(args.source / "deps/LIBERO/libero/libero/assets"),
              "bddl_files": str(base / "env/libero/bddl_files"),
              "init_states": str(args.output / "paired_init"),
              "datasets": str(args.assets / "dataset/demonstration_data")}
    (config_dir / "config.yaml").write_text(yaml.safe_dump(config))
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    print("MOTOR_STAGE", "importing_bpp_runner", flush=True)
    from behavior_prompting.train_network.env_runner.libero_image_runner import LiberoImageRunner
    from behavior_prompting.train_network.utils.libero_util import discover_and_register_benchmarks
    discover_and_register_benchmarks()
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    checkpoint = args.assets / "model/liberogen_spatial_combination_behavior_prompting.ckpt"
    assert digest(checkpoint) == "74e0f841f3b86f1383adda23dab59fddee8806a4bcb5685ed7f826afb320c097"
    print("MOTOR_STAGE", "reading_released_checkpoint", flush=True)
    payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    (args.output / "checkpoint-config.yaml").write_text(OmegaConf.to_yaml(cfg))
    # Two inactive language options were removed from the released runner API.
    # Admit only their exact disabled values; reject every other unknown field.
    runner_config = OmegaConf.to_container(cfg.task.env_runner, resolve=True)
    runner_config.pop("_target_")
    assert cfg.task.shape_meta.obs.task_language.ignore_by_policy is True
    removed_runner_fields = {}
    for field, disabled in [("language_embedding", None),
                            ("use_finetuned_language_encoder", False)]:
        if field in runner_config:
            assert runner_config[field] is disabled
            removed_runner_fields[field] = runner_config.pop(field)
    runner_config["shape_meta"] = cfg.task.shape_meta
    runner_signature = inspect.signature(LiberoImageRunner)
    runner_signature.bind(dataset_path="checked-per-task", output_dir="checked-per-task",
                          **runner_config)
    (args.output / "runner-interface-receipt.json").write_text(json.dumps({
        "removed_inactive_legacy_fields": removed_runner_fields,
        "task_language_ignored_by_policy": True,
        "remaining_checkpoint_kwargs_accepted": True,
        "signature": str(runner_signature)}, indent=2) + "\n")
    # Preserve the checkpoint encoder's pretrained/frozen/group-norm branches.
    # Suppress only timm's external weight download: the released checkpoint
    # strictly replaces every parameter and buffer immediately after construction.
    import timm
    from unittest.mock import patch
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    saved_state = payload["state_dicts"]["model"]
    original_tensor_count = sum(isinstance(v, torch.Tensor) for v in saved_state.values())
    tokenizer_cfg = model_cfg.obs_encoder.obs_encoder
    assert tokenizer_cfg._target_ == "behavior_prompting.train_network.model.prompt.prompt_obs_encoder.PairPromptTransformerTokenizer"
    compatibility_overrides = []
    # Official libero_policy_dunetp.yaml explicitly disables this for the paper
    # models. Older released checkpoint configs omit the field, while the newer
    # Python constructor defaults to True and would add an untrained parameter.
    legacy_key = "obs_encoder.obs_encoder.pool_modality_pos_embed"
    if "use_pool_modality_pos_embed" not in tokenizer_cfg:
        assert legacy_key not in saved_state
        tokenizer_cfg.use_pool_modality_pos_embed = False
        compatibility_overrides.append({
            "path": "model.obs_encoder.obs_encoder.use_pool_modality_pos_embed",
            "checkpoint_value": "absent", "effective_value": False,
            "evidence": "BPP 58125e6 config/libero_policy_dunetp.yaml:12; paper models trained with false"})
    else:
        assert tokenizer_cfg.use_pool_modality_pos_embed is False and legacy_key not in saved_state
    print("MOTOR_COMPATIBILITY", json.dumps(compatibility_overrides), flush=True)
    original_factory = timm.create_model
    factory_calls = []
    def create_without_download(*factory_args, **factory_kwargs):
        factory_calls.append({"model_name": factory_kwargs.get("model_name", factory_args[0] if factory_args else None),
                              "requested_pretrained": factory_kwargs.get("pretrained", False),
                              "effective_pretrained": False})
        factory_kwargs["pretrained"] = False
        return original_factory(*factory_args, **factory_kwargs)
    print("MOTOR_STAGE", "constructing_checkpoint_model", flush=True)
    with patch.object(timm, "create_model", create_without_download):
        policy = hydra.utils.instantiate(model_cfg)
    assert timm.create_model is original_factory
    assert factory_calls, "Expected the checkpoint visual backbone to use timm"
    policy.load_state_dict(saved_state, strict=True)
    loaded_state = policy.state_dict()
    verified_tensors = 0
    for name, value in saved_state.items():
        if isinstance(value, torch.Tensor):
            loaded = loaded_state[name]
            assert loaded.dtype == value.dtype and loaded.shape == value.shape
            assert torch.equal(loaded, value), f"Released tensor changed: {name}"
            verified_tensors += 1
    assert verified_tensors > 0
    (args.output / "construction-receipt.json").write_text(json.dumps({
        "factory_calls": factory_calls, "factory_restored": True,
        "strict_checkpoint_load": True, "exact_saved_tensors_verified": verified_tensors,
        "original_checkpoint_tensor_count": original_tensor_count,
        "checkpoint_model_config": "preserved except documented legacy-default completion",
        "compatibility_overrides": compatibility_overrides}, indent=2) + "\n")
    del payload, saved_state, loaded_state
    policy.requires_grad_(False).eval().cuda()
    print("MOTOR_STAGE", "strict_loaded_frozen_cuda", verified_tensors, flush=True)
    versions = {name: p._version for name, p in policy.named_parameters()}
    (args.output / "instantiated-model-config.yaml").write_text(OmegaConf.to_yaml(model_cfg))

    from custom_env import CustomControlEnv, PUBLIC_HORIZON
    import behavior_prompting.train_network.env_runner.libero_image_runner as runner_module

    class TraceProxy:
        def __init__(self, inner):
            self.inner = inner
            self.records = []
            self.reset_hashes = []
        def reset(self, *a, **kw):
            obs = self.inner.reset(*a, **kw)
            self.reset_hashes.append({k: sha(v) for k, v in obs.items()})
            return obs
        def __getattr__(self, name):
            return getattr(self.inner, name)
        def call(self, name, *a, **kw):
            value = self.inner.call(name, *a, **kw)
            if name == 'get_attr' and a == ('reward',):
                records = self.inner.call('finish_record')
                assert len(records) == len(value) == 1
                assert float(np.max(value[0])) == records[0]['selected_skill_success']
                self.records.extend(records)
            return value

    def prompt_hash(value):
        h = hashlib.sha256()
        def visit(x):
            if isinstance(x, dict):
                for k in sorted(x):
                    h.update(str(k).encode()); visit(x[k])
            elif isinstance(x, (list, tuple)):
                for v in x: visit(v)
            elif isinstance(x, torch.Tensor):
                h.update(str((x.dtype, tuple(x.shape))).encode())
                h.update(x.detach().cpu().contiguous().numpy().tobytes())
            else:
                h.update(repr(x).encode())
        visit(value)
        return h.hexdigest()

    results = []
    original_prompt = policy.prompt
    original_reset = policy.reset
    for candidate, task in enumerate(TASKS):
        folder = args.output / f'candidate-{candidate}'
        folder.mkdir()
        prompt_digests = []
        episode_seeds = []
        def seeded_reset(*a, **kw):
            # The runner resets once per layout and once at cleanup. Seed each
            # actual episode independently so earlier stopping cannot change
            # the later episode's diffusion noise in the other condition.
            if len(episode_seeds) < len(indices):
                seed = 91001 + 10000 * candidate + indices[len(episode_seeds)]
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                episode_seeds.append(seed)
            return original_reset(*a, **kw)
        policy.reset = seeded_reset
        def traced_prompt(value):
            current = prompt_hash(value)
            if args.condition == 'original':
                assert current == REFERENCE_PROMPTS[candidate], 'Copied original demo changed the qualified prompt tensors'
            if prompt_digests:
                assert current == prompt_digests[0], 'Candidate demonstration changed across layouts'
            prompt_digests.append(current)
            return original_prompt(value)
        policy.prompt = traced_prompt

        def create_custom_env(bddl_file_name, camera_heights, camera_widths, control_delta,
                              enable_render=True, render_size=None, render_gpu_device_id=-1):
            assert camera_heights == camera_widths == 128
            assert control_delta is True
            assert render_size is None, 'Geometry admission used native128px observations'
            return CustomControlEnv(
                manifest=manifest, geometry_root=args.geometry, output_root=args.output,
                candidate=candidate, bddl_file_name=bddl_file_name,
                camera_heights=camera_heights, camera_widths=camera_widths,
                has_offscreen_renderer=enable_render, has_renderer=False,
                use_camera_obs=enable_render, control_delta=control_delta,
                render_size=render_size, render_gpu_device_id=render_gpu_device_id)

        runner_kwargs = dict(runner_config)
        runner_kwargs.update(dataset_path=str(args.assets / 'dataset/demonstration_data' / SPLIT / (task + '_demo.hdf5')),
                             output_dir=str(folder), cache_dir=str(args.output / 'prompt-cache'),
                             n_envs=1, n_train=0, n_train_vis=0, n_test=len(indices),
                             n_test_vis=len(indices), is_seen=False, eval_seen=True, eval_unseen=False,
                             vis_prompt=False, vis_attention_map=False, vis_actions_rollout=False,
                             vis_proprio_rollout=False, vis_trajectory_video=False, render_gpu_device_id=0)
        assert runner_kwargs['max_steps'] == 550 and runner_kwargs['exec_action_horizon'] == 12
        runner_signature.bind(**runner_kwargs)
        print('CUSTOM_STAGE', 'constructing_runner', candidate, indices, flush=True)
        with patch.object(runner_module, 'create_env', create_custom_env):
            runner = LiberoImageRunner(**runner_kwargs)
            assert runner.max_steps == PUBLIC_HORIZON
            proxy = TraceProxy(runner.env)
            runner.env = proxy
            started = time.monotonic()
            try:
                with torch.inference_mode():
                    scores = runner.run(policy, enable_expensive_vis=False)
                assert len(proxy.records) == len(indices) == len(prompt_digests)
                assert len(episode_seeds) == len(indices)
                result = {'candidate': candidate, 'indices': indices,
                          'condition': args.condition, 'policy_seeds': episode_seeds,
                          'prompt_hdf5_sha256': digest(Path(runner_kwargs['dataset_path'])),
                          'seconds': time.monotonic() - started,
                          'fixed_prompt_sha256': prompt_digests[0],
                          'prompt_calls': len(prompt_digests),
                          'scores': {k: float(v) for k, v in scores.items() if isinstance(v, (int, float, np.number))},
                          'records': proxy.records, 'outer_reset_hashes': proxy.reset_hashes}
                (folder / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
                results.append(result)
            finally:
                runner.close()
                policy.reset = original_reset
    policy.prompt = original_prompt
    assert versions == {name: p._version for name, p in policy.named_parameters()}, 'Motor parameters changed'
    for i, index in enumerate(indices):
        a, b = [result['records'][i] for result in results]
        assert a['index'] == b['index'] == index
        assert a['initial_state_sha256'] == b['initial_state_sha256']
        assert a['initial_observation_hashes'] == b['initial_observation_hashes']
    final = {'complete': True, 'paired_initial_observations': True, 'indices': indices,
             'admission_sha256': args.admission_sha256, 'public_horizon': PUBLIC_HORIZON,
             'checkpoint_sha256': digest(checkpoint), 'gpu': torch.cuda.get_device_name(),
             'reader_trained': False, 'results': results,
             'mode': 'prompt_repair', 'condition': args.condition,
             'new_prompt_training_split_filter': False,
             'matched_policy_seed_rule': '91001 + 10000*candidate + layout_index'}
    (args.output / 'result.json').write_text(json.dumps(final, indent=2) + '\n')
    (args.output / 'complete').write_text('complete\n')


if __name__ == '__main__':
    main()

