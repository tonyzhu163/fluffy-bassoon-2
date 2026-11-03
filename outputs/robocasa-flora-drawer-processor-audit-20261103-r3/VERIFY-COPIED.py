"""Verify downloaded bytes and recorded CPU results; no model code executes."""
import hashlib
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parent
archive = Path('/Users/mowenjie/Downloads/drawer-processor-audit-r3.zip')
source = 'd565865a0c15db29010f7dd5d84373a1686bdb6f'
sha = lambda b: hashlib.sha256(b).hexdigest()
read = lambda p: json.loads(p.read_text())
with zipfile.ZipFile(archive) as z:
    manifest = json.loads(z.read('MANIFEST.json'))
    assert manifest['source_commit'] == source and len(manifest['files']) == 6
    for name,digest in manifest['files'].items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        data = z.read(name)
        assert sha(data) == digest,name
        path = root/name
        if path.exists():assert path.read_bytes() == data,name
        else:path.write_bytes(data)
    (root/'MANIFEST.json').write_bytes(z.read('MANIFEST.json'))
receipt,processor = read(root/'RECEIPT.json'),read(root/'PROCESSOR.json')
assert receipt['completed'] and receipt['source_commit'] == source
assert receipt['case_count'] == 4 and receipt['reconstructed_frames'] == 256
assert receipt['raw_rgb_matches_previous_scoring_and_visual_audit']
assert receipt['all_float32_roundtrips_byte_exact']
assert not receipt['torch_cuda_initialized'] and receipt['cuda_visible_devices'] == ''
assert receipt['model_weight_loads'] == receipt['model_inference_calls'] == receipt['simulator_calls'] == 0
assert receipt['policy_controls'] == receipt['motor_calls'] == receipt['optimizer_steps'] == 0
assert processor['source_survey'] == read(root/'SOURCE-SURVEY.json')
assert processor['normalization']['matches_saved_model_config']
assert processor['normalization']['mean'] == processor['normalization']['std'] == [0.5,0.5,0.5]
assert processor['request_kwargs'] == dict(do_sample_frames=False,size=dict(shortest_edge=4096,longest_edge=29360128))
prior = root.parent/'robocasa-flora-drawer-target-hints-20261103-r1'
references = read(prior/'FRAME-REFERENCE.json')
prompts = {p['id']:p for file in (prior/'scoring').glob('*.prompts.json') for p in read(file)}
expected = {('000112',350),('000256',750),('000376',750),('000128',450)}
assert {(c['case'],c['prefix_controls']) for c in receipt['cases']} == expected
for case in receipt['cases']:
    uid = case['id']
    assert case['raw_shape'] == [64,320,960,3]
    assert case['video_grid_thw'] == [32,20,60] and case['feature_shape'] == [38400,1536]
    assert case['feature_dtype'] == 'torch.float32' and case['merged_vision_tokens'] == 9600
    assert case['full_and_direct_video_processor_equal'] and case['actual_previous_rendered_prompt_matches']
    assert case['rendered_prompt_sha256'] == prompts[uid+'/native_boundary']['rendered_sha256']
    assert case['indices'] == references[uid]['indices'] == case['metadata']['frames_indices']
    assert case['metadata']['fps'] == 20. and case['metadata']['do_sample_frames'] is False
    assert case['raw_rgb_sha256'] == case['roundtrip_rgb_sha256'] == references[uid]['rgb_sha256']
    assert len(case['raw_rgb_sha256']) == 64
    assert case['raw_byte_differences'] == 0
    assert case['bf16_reconstruction_differing_channels'] == case['bf16_reconstruction_max_byte_error'] == 0
    assert sha((root/case['image_file']).read_bytes()) == case['image_sha256']
report = dict(completed=True,source_commit=source,verified_artifacts=len(manifest['files']),
    archive_bytes=archive.stat().st_size,archive_sha256=sha(archive.read_bytes()),
    cases=4,reconstructed_frames=256,source_frame_hashes_match=True,all_recorded_float32_rgb_bytes_match=True,
    all_recorded_bfloat16_reconstruction_rgb_bytes_match=True,actual_previous_rendered_prompts_match=4,
    normalization_matches_saved_model_config=True,cpu_only=True,model_inference_calls=0,
    check='Artifact bytes and recorded CPU numerical results only; previous GPU tensors were not saved')
(root/'COPIED-HASHES.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
print('VERSIONS',processor['versions'])
print('NORMALIZATION',processor['normalization'])
print('CPU_SECONDS',receipt['seconds'])
for case in receipt['cases']:
    print(case['case'],case['prefix_controls'],case['float_roundtrip_max_rgb_error'],case['bf16_reconstruction_max_rgb_float_error'])
