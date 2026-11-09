"""Inference exports must not weaken full numerical training recovery."""
from copy import deepcopy

from omegaconf import OmegaConf
import pytest
import torch
from torch.distributed import checkpoint as dcp

from fluffy_bassoon.robocasa.flow_ppo import checkpoint_weight_export


def test_dcp_without_extra_weights_reproduces_next_sample_and_adam_step(tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(137)
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9)

        def update(network, optim, schedule):
            batch, target = torch.randn(7, 4), torch.randn(7, 2)
            loss = (network(batch)-target).square().mean()
            optim.zero_grad(); loss.backward(); optim.step(); schedule.step()
            return batch, target, loss.detach()

        update(model, optimizer, scheduler)
        config = OmegaConf.create({})
        saved = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), rng=torch.get_rng_state())
        with checkpoint_weight_export(config, step=1, final_step=3, export_steps=[3]):
            dcp.save(saved, checkpoint_id=tmp_path/'dcp')
            if config.get('save_full_model_weights', True):
                torch.save(model.state_dict(), tmp_path/'full_weights.pt')
        assert not (tmp_path/'full_weights.pt').exists()
        assert 'save_full_model_weights' not in config
        expected = update(model, optimizer, scheduler)
        expected_model, expected_optimizer = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())

        restored = torch.nn.Linear(4, 2)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.003)
        restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1, gamma=.9)
        update(restored, restored_optimizer, restored_scheduler)  # initialize Adam storage for DCP
        recovered = dict(model=restored.state_dict(), optimizer=restored_optimizer.state_dict(),
                         scheduler=restored_scheduler.state_dict(), rng=torch.get_rng_state())
        dcp.load(recovered, checkpoint_id=tmp_path/'dcp')
        restored.load_state_dict(recovered['model'])
        restored_optimizer.load_state_dict(recovered['optimizer'])
        restored_scheduler.load_state_dict(recovered['scheduler'])
        torch.set_rng_state(recovered['rng'])
        actual = update(restored, restored_optimizer, restored_scheduler)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for key, value in restored.state_dict().items():
            torch.testing.assert_close(value, expected_model[key], rtol=0, atol=0)
        for key, state in restored_optimizer.state_dict()['state'].items():
            for name, value in state.items():
                torch.testing.assert_close(value, expected_optimizer['state'][key][name], rtol=0, atol=0)
        assert restored_scheduler.state_dict() == scheduler.state_dict()

        with checkpoint_weight_export(config, step=3, final_step=3, export_steps=[3]):
            assert config['save_full_model_weights']
            torch.save(restored.state_dict(), tmp_path/'final_weights.pt')
        exported = torch.load(tmp_path/'final_weights.pt', weights_only=True)
        for key, value in exported.items():
            torch.testing.assert_close(value, expected_model[key], rtol=0, atol=0)


def test_failed_write_restores_setting_and_missing_final_is_rejected():
    config = OmegaConf.create({'save_full_model_weights': True})
    with pytest.raises(OSError, match='write failed'):
        with checkpoint_weight_export(config, step=1, final_step=3, export_steps=[3]):
            assert not config['save_full_model_weights']
            raise OSError('write failed')
    assert config['save_full_model_weights']
    with pytest.raises(ValueError, match='final teacher'):
        with checkpoint_weight_export(config, step=1, final_step=3, export_steps=[1]):
            pytest.fail('An incomplete export schedule must fail before saving')
    assert config['save_full_model_weights']
