# Sidney to RLinf: static compatibility audit

This is code and checkpoint-metadata evidence, not numerical qualification or
a trained result. The native GPU job remains unchanged at main `34a3a429`.

## Weight layout and input embeddings

The Sidney safetensors header contains 813 tensors prefixed with `model.`.
After removing that wrapper prefix, all 812 Motus tensor names are present
with the same shapes. Sidney additionally stores
`paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight`.
Same shapes do not establish the same values or inference behavior.

The input embedding and `paligemma.lm_head.weight` each contain
257152 x2048 BF16 elements, but their serialized byte hashes differ. See
`EMBEDDING-IDENTITY.json`. The hash check read checkpoint byte ranges without
constructing a model or performing tensor computation. The native LeRobot
policy uses the input embedding to encode language.

The pinned RLinf
[`old_to_new_state_dict`](https://github.com/RLinf/RLinf/blob/c605b9d55ed93a180119b099e6f836abff293b08/rlinf/utils/ckpt_convertor/openpi/openpi_pytorch_to_openpi.py)
assumes tied embeddings and fills `llm.embedder.embedding.weight` from the
language output head. That assumption is unsuitable for this checkpoint.
An explicit conversion must remove the `model.` prefix, apply the released
QKV/MLP layout transforms, and preserve the actual input embedding. No copied
or converted weight file has been produced yet.

The native worker uses strict `safetensors.torch.load_model`, avoiding the
LeRobot convenience loader's caught exceptions and weight-remapping fallback.
If the chosen native runtime ties these distinct checkpoint tensors, the
strict check must reject that incompatible representation rather than silently
choose one. Actual strict loading still awaits the allocated job.

## Input and output transforms

The pinned RLinf
[`LeRobotAlohaDataConfig`](https://github.com/RLinf/RLinf/blob/c605b9d55ed93a180119b099e6f836abff293b08/rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_dataconfig.py)
defaults to `adapt_to_pi=True` and `extra_delta_transform=True`. Both must be
disabled for Sidney's raw ALOHA absolute-joint convention. The inherited pi0.5
data config also selects quantile normalization; Sidney requires the saved
mean/std arrays. Do not substitute a different normalization file or derive
statistics from held-out episodes.

The transform builder reads `data_config.use_quantile_norm` after calling
the data factory's `create`. A process-local Sidney data-factory registration
can return a dataclass copy with that value set to False, preserving the
external source checkout and existing Motus recipe. A plain YAML boolean is
insufficient unless the actual factory supports it. Verify the instantiated
transform chain and its numerical outputs.

The first conversion qualification must compare the native and RLinf paths
on the same saved RGB/state/instruction arrays and explicit `[1,50,32]`
Gaussian noise: camera mapping and pixels after resize, normalized state,
token IDs/masks, flow predictions and 14-D decoded absolute actions. Include
multiple source observations and report errors; successful key loading alone
does not establish parity. Initially keep H50/K10. The VLRLA K5 sampler is a
separate declared migration after native parity, before collecting histories.

## CPU check attempt

`cpu_imports.py` and its wrapper were committed at `3a059722` for a separate
allocated CPU import/processor check. A zero-GPU copy-form submission was not
observed as a created task, and no `cpu01` directory appeared. The unaccepted
draft was canceled. No CPU test result exists. This does not qualify the
native runtime or justify modifying its queued environment.
