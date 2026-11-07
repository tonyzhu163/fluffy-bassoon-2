"""Linux numerical qualification of native ODE observation and readout schema."""
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace, MethodType
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, write
from fluffy_bassoon.meta_rl.tensor_archive import read_tensor
from motus_source_readout import configuration, observe_native_ODE, selected_seeds
from recovery_signature import digest

ROOT = Path(__file__).resolve().parent
BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and not torch.cuda.is_initialized()
    import rlinf.models.embodiment.openpi.pi0 as pinned
    fake = SimpleNamespace(action_horizon=32,action_dim=32,
        build_prefix_cache=lambda observation:(None,None,None),
        run_suffix=lambda observation,x,t,cache,mask:x*.125,
        velocity_from_suffix=lambda suffix:suffix)
    observation = SimpleNamespace(state=torch.zeros(2,14))
    with patch.object(pinned.model,'preprocess_observation',lambda value,train=False:value):
        torch.manual_seed(31123)
        reference = pinned.Pi0.sample_actions(fake,observation,num_steps=5)
        reference_rng = torch.get_rng_state()
        torch.manual_seed(31123)
        result,noise = observe_native_ODE(MethodType(pinned.Pi0.sample_actions,fake),32,
            observation,num_steps=5)
        assert torch.equal(reference,result) and torch.equal(reference_rng,torch.get_rng_state())
    cases = json.loads((ROOT/'MOTUS-SOURCE-READOUT-CASES.json').read_bytes())
    assert len(set(cases['dev']+cases['confirmation'])) == 128
    for name in ('dev','confirmation'):
        rows = [seed for panel in range(4) for rank in range(8)
            for seed in selected_seeds(cases[name],panel,rank)]
        assert rows == cases[name]
    import rlinf.config as native
    class Placement:
        def get_world_size(self,role):return 8
    rows = []
    for method in ('ppo','grpo'):
        state = BASE/f'motusbehavior{method}01/state'
        plan = dict(algorithm=method,original_config=str(state/'resolved-native-config.json'),
            source_state=str(state),output_identity=f'motus-source-{method}-dev01',bank='dev')
        cfg = configuration(plan,Path('/tmp/readout-schema')/method)
        with patch.object(native,'Cluster',lambda *a,**k:None),patch.object(native,'HybridComponentPlacement',lambda *a,**k:Placement()):
            native.validate_cfg(cfg)
        assert cfg.env.eval.total_num_envs == 16 and cfg.env.eval.rollout_epoch == 1
        assert cfg.native_pilot.maximum_actions == 32768
        assert cfg.native_pilot.readout_bank == 'dev'
        assert configuration(dict(plan,bank='confirmation'),Path('/tmp/readout-confirm-schema')/method).native_pilot.readout_bank == 'confirmation'
        bank = Path(cfg.env.eval.seeds_path)
        assert sha(bank) == cases['full_source_sha256']
        assert json.loads(bank.read_bytes())['place_phone_stand']['success_seeds'][:128] == cases['dev']+cases['confirmation']
        train = json.loads(Path(cfg.env.train.seeds_path).read_bytes())['place_phone_stand']['success_seeds']
        assert not set(train) & set(cases['dev']+cases['confirmation'])
        rows.append(dict(algorithm=method,world_size=8,native_batch=2,episodes_per_checkpoint=64))
    source_hashes = {}
    for rank in range(8):
        for prediction in range(6):
            records = []
            for method in ('ppo','grpo'):
                path = BASE/f'motusbehavior{method}01/state/sampling/rank-{rank:02d}/{prediction:05d}.pt.gz'
                saved = read_tensor(path)
                assert saved['mode']=='eval' and saved['producing_policy_iteration']==0
                source_hashes[f'{method}-rank-{rank:02d}-prediction-{prediction}'] = sha(path)
                records.append(saved)
            left,right = records
            for field in ('public_observation_sha256','actions_sha256'):
                assert left[field] == right[field], (rank,prediction,field)
            assert digest(left['rng_before']) == digest(right['rng_before'])
            assert digest(left['rng_after']) == digest(right['rng_after'])
            for field in ('action','model_action'):
                assert torch.equal(left['forward_sampling_fields'][field],right['forward_sampling_fields'][field])
    subprocess.run(['bash','-n',str(ROOT/'motus_source_readout.sh')],check=True)
    output = BASE/'cpumotussourcereadout04';output.mkdir()
    write(output/'RESULT.json',dict(status='passed_readout_code_and_native_ODE_observation_reference',
        source_commit=os.environ['TASK_SOURCE'],gpu_seconds=0,models_or_simulators_constructed=False,
        actual_pinned_ODE_loop_used=True,deterministic_suffix_stub=True,
        ODE_output_and_final_RNG_exact=True,recorded_noise_shape=list(noise.shape),
        full_VLA_GPU_and_source_SFT_action_calibration_required_in_actual_readout=True,
        checkpoint2_input_admission_still_required=True,configurations=rows,
        source_SFT_baseline_all48_predictions_match=True,source_sampling_file_hashes=source_hashes,
        entry_sha256=sha(ROOT/'motus_source_readout.py'),launcher_sha256=sha(ROOT/'motus_source_readout.sh')))


if __name__=='__main__':main()
