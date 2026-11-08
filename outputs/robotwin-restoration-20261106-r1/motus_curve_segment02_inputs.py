"""Qualification of the actual C6 boundary before the second learning segment."""
import json
from pathlib import Path

from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha
from fluffy_bassoon.meta_rl.tensor_archive import read_tensor
from qualify_motus_curve_readout_cpu import actual_inputs


def verify_inputs(plan):
    assert (plan['start_iteration'],plan['stop_iteration']) == (6,10)
    assert plan['algorithm'] == 'grpo'
    checkpoint = plan['resume_checkpoint']
    for value in (checkpoint['manifest_sha256'],plan['independent_prefix_readback_sha256']):
        assert isinstance(value,str) and len(value)==64, 'Actual C6 manifest and independent readback hashes are required'
        int(value,16)
    first_ordinals = plan['resume_dev_sampling_first_ordinals']
    assert isinstance(first_ordinals,list) and len(first_ordinals)==8
    assert all(type(n) is int and n>=0 for n in first_ordinals)
    assert Path(checkpoint['path']).name == 'global_step_6'
    # Reuse the full cold C0/C6, owned-Adam, all24 sidecar and unchanged native
    # numerical-configuration checks used for the fixed C6 readout input.
    view = dict(algorithm='grpo',bank='dev',selected_training_iteration=6,
        extended_source_curve_readout=True,output_identity=plan['output_identity'],
        original_config=plan['original_config'],source_SFT_reference_state=plan['source_SFT_reference_state'],
        checkpoints={'0':plan['source_SFT_checkpoint'],'6':checkpoint},
        independent_training_readback_result=plan['independent_prefix_readback_result'],
        independent_training_readback_sha256=plan['independent_prefix_readback_sha256'])
    checked = actual_inputs(view)
    prior = json.loads(Path(plan['independent_prefix_readback_result']).read_bytes())
    assert prior['checked_json_hashes'][plan['original_config']] == sha(plan['original_config'])
    hashes = {}
    for rank,first in enumerate(first_ordinals):
        root = Path(plan['resume_dev_sampling_root'])/f'rank-{rank:02d}'
        expected = [root/f'{first+offset:05d}.pt.gz' for offset in range(6)]
        assert sorted(root.glob('*.pt.gz'))[-6:] == expected
        for ordinal,path in enumerate(expected,first):
            saved = read_tensor(path)
            assert saved['mode']=='eval' and saved['producing_policy_iteration']==6
            assert saved['rank']==rank and saved['prediction_ordinal']==ordinal
            assert not saved['terminal_value_only']
            assert saved['forward_sampling_fields']['action'].shape == (2,448)
            assert saved['rng_before']['cuda'] and saved['rng_after']['cuda']
            hashes[str(path)] = sha(path)
    return dict(actual_checkpoint6_inputs_verified=True,
        resume_checkpoint_sha256=checkpoint['manifest_sha256'],
        original_training_config_sha256=sha(plan['original_config']),
        all48_actual_checkpoint6_DEV_reference_files=hashes,
        full_C0_C6_cold_input_qualification=checked,
        resumed_Adam_counter=90,final_Adam_counter=150,
        actual_GPU_model_resume_and_fresh_collections_still_required=True)
