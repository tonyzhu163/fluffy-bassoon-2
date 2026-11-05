"""Sidney LeRobot pi0.5 I/O for the pinned RLinf native learner.

The checkpoint uses absolute ALOHA joints, mean/std and Hugging Face Gemma
tokenization. Keep these at the model boundary; the RLinf PPO learner and
Flow-SDE sampler are unchanged. Heavy imports stay inside the allocated runtime.
"""
from pathlib import Path
import json

CONFIG_NAME = 'pi05_aloha_robotwin_sidney'
REVISION = 'e49e2ab6c11f07511573b67261bd129e88d0a416'
TOKENIZER = '/mowenjie/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c'


def _vision_mlp(self, x):
    import torch.nn.functional as F
    return self.fc2(self.dropout(F.gelu(self.fc1(x), approximate='tanh')))


def _gemma_mlp(self, x):
    import torch
    import torch.nn.functional as F
    gate = torch.matmul(x, self.w_gating[0].to(x.dtype))
    value = torch.matmul(x, self.w_gating[1].to(x.dtype))
    return torch.matmul(F.gelu(gate, approximate='tanh')*value, self.w_linear.to(x.dtype))


def _install_native_activations(model):
    from types import MethodType
    # LeRobot's SigLIP and both Gemma experts use gelu_pytorch_tanh.
    # The pinned RLinf implementation uses exact GELU. Keep Sidney's saved
    # architecture without changing the sampler, learner or other checkpoints.
    for layer in model.img.encoder.layers:
        layer.mlp.forward = MethodType(_vision_mlp, layer.mlp)
    for layer in model.llm.layers:
        for mlp in layer.mlps:
            mlp.forward = MethodType(_gemma_mlp, mlp)


def deterministic_reference():
    import os
    import torch
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before importing Torch')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


class SidneyInputs:
    def __init__(self, metadata):
        import numpy as np
        import torch
        from transformers import AutoTokenizer
        self.mean = np.asarray(metadata['state_mean'], dtype=np.float32)
        self.std = np.asarray(metadata['state_std'], dtype=np.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
        self.tokenizer.padding_side = 'right'
        self.resize_device=torch.device('cuda',torch.cuda.current_device())

    def __call__(self, sample):
        import numpy as np
        import torch
        import torch.nn.functional as F
        state = np.asarray(sample['observation/state'], dtype=np.float32)
        if state.shape != (14,):
            raise ValueError(f'Sidney requires14 raw joints, got {state.shape}')
        state = (state-self.mean)/(self.std+1e-8)
        bins = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1])-1
        task = sample['prompt'].strip().replace('_', ' ').replace('\n', ' ')
        prompt = f"Task: {task}, State: {' '.join(map(str,bins))};\nAction: "
        encoded = self.tokenizer(prompt, padding='max_length', max_length=200,
                                 truncation=True, return_tensors='np')
        high = sample['observation/image']
        wrists = sample['observation/wrist_image']
        cameras = (high, wrists[0], wrists[1])
        keys = ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb')
        images = {}
        for key, camera in zip(keys, cameras, strict=True):
            camera = np.asarray(camera)
            if camera.dtype != np.uint8 or camera.ndim != 3 or camera.shape[-1] != 3:
                raise ValueError('Sidney requires uint8 HWC RGB before preprocessing')
            # Match native LeRobot: float[0,1], bilinear resize, zero padding,
            # then map to[-1,1]. Do not round back to uint8 after resize.
            image = (torch.from_numpy(camera.copy()).permute(2,0,1)[None].float()/255).to(self.resize_device)
            height, width = camera.shape[:2]
            ratio = max(width/224, height/224)
            resized_h, resized_w = int(height/ratio), int(width/ratio)
            image = F.interpolate(image, size=(resized_h,resized_w), mode='bilinear', align_corners=False)
            image = image.clamp(0.,1.)
            top = (224-resized_h)//2
            left = (224-resized_w)//2
            image = F.pad(image,(left,224-resized_w-left,top,224-resized_h-top),value=0.)
            images[key] = (image[0].permute(1,2,0)*2-1).cpu().numpy()
        return dict(image=images,image_mask={key:np.asarray(True) for key in keys},
                    state=state,tokenized_prompt=encoded['input_ids'][0],
                    tokenized_prompt_mask=encoded['attention_mask'][0].astype(bool))


class SidneyOutputs:
    def __init__(self, metadata):
        import numpy as np
        self.mean = np.asarray(metadata['action_mean'],dtype=np.float32)
        self.std = np.asarray(metadata['action_std'],dtype=np.float32)

    def __call__(self, sample):
        return dict(sample, actions=sample['actions'][...,:14]*self.std+self.mean)


def install_rlinf_transforms():
    """Install this named checkpoint adapter in the current worker only."""
    import rlinf.models.embodiment.openpi as factory
    original = factory._install_transforms
    if getattr(original, '_fluffy_sidney', False):
        return

    def install(model, cfg, config_name):
        if config_name != CONFIG_NAME:
            return original(model,cfg,config_name)
        metadata = json.loads((Path(cfg.model_path)/'sidney-processors.json').read_text())
        if metadata['revision'] != REVISION or metadata['normalization'] != 'MEAN_STD':
            raise ValueError('Unexpected Sidney processor identity')
        if cfg.openpi_data.adapt_to_pi or cfg.openpi_data.extra_delta_transform:
            raise ValueError('Sidney uses raw absolute ALOHA joints')
        model.setup_transforms([SidneyInputs(metadata)],[SidneyOutputs(metadata)])
        _install_native_activations(model)
        model._sidney_full_vision_fp32 = True
        # Pi0.__init__ enables TF32 globally; restore the qualified reference
        # setting after construction and before any inference.
        deterministic_reference()

    install._fluffy_sidney = True
    factory._install_transforms = install
    original_dtypes = factory._apply_openpi_param_dtypes

    def dtypes(model, target_dtype):
        if not getattr(model,'_sidney_full_vision_fp32',False):
            return original_dtypes(model,target_dtype)
        import torch
        if target_dtype not in (None,torch.bfloat16,torch.float32):
            raise ValueError('Sidney supports native mixed BF16 or an FP32 diagnostic')
        # Choose final dtypes directly. Casting all parameters to BF16 and then
        # restoring FP32 would irreversibly round the checkpoint's FP32 tower
        # and norm tensors. Native LeRobot chooses dtypes BEFORE weight load.
        norm_names=('pre_attention_norms','pre_ffw_norms','final_norms')
        for name,param in model.named_parameters():
            backbone=name.startswith('llm.') and not any(s in name for s in norm_names)
            dtype=torch.bfloat16 if backbone and target_dtype!=torch.float32 else torch.float32
            param.data=param.data.to(dtype=dtype)
        model.img.dtype_mm=torch.float32
        model.llm.embed_dtype=torch.float32 if target_dtype==torch.float32 else torch.bfloat16

    factory._apply_openpi_param_dtypes = dtypes


def model_config(rlinf_root, model_path, *, flow_steps=10, task='eval'):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(Path(rlinf_root)/'examples/embodiment/config/model/pi0_5.yaml')
    return OmegaConf.merge(cfg,dict(model_path=str(model_path),precision=None,pi05=True,
        num_action_chunks=50,action_dim=14,num_steps=flow_steps,add_value_head=(task=='rl'),
        openpi_data=dict(adapt_to_pi=False,extra_delta_transform=False),
        openpi=dict(task=task,config_name=CONFIG_NAME,num_images_in_input=3,
                    action_horizon=50,action_chunk=50,action_env_dim=14,
                    model_action_dim=32,num_steps=flow_steps,torch_compile=False,
                    max_token_len=200,discrete_state_input=True)))
