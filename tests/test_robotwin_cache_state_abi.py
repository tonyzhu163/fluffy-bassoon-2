"""Allocated CPU only: actual full-bank sealer, official prepared-state ABI."""
import json
import time

import pytest
import torch

from test_robotwin_teacher_bank import complete_cache_fixture
from fluffy_bassoon.meta_rl import robotwin_teacher_bank as bank
from fluffy_bassoon.meta_rl.robotwin_demo_capacity import file_sha


def model_state_fixture(tmp_path,monkeypatch,width):
    value,path,receipt=complete_cache_fixture(tmp_path,monkeypatch)
    # Prepared model state is separate from physical14 commands/public fields.
    # These synthetic tensors are not transformed/replaced paid runtime data.
    for row in receipt['records']:
        saved=torch.load(row['path'],map_location='cpu',weights_only=False)
        saved['data']['state']=torch.zeros(1,width)
        saved['data']['state'][0,:min(14,width)]=torch.arange(min(14,width))/10
        torch.save(saved,row['path']);row['sha256']=file_sha(row['path'])
    path.write_text(json.dumps(receipt))
    return value,path,receipt


def test_official_padded_state_abi_seals_without_rewriting_any_cached_payload(tmp_path,monkeypatch):
    value,path,receipt=model_state_fixture(tmp_path,monkeypatch,32)
    before={row['path']:file_sha(row['path']) for row in receipt['records']}
    result=bank.seal_full_cache(value,[dict(path=str(path),sha256=file_sha(path))],deadline=time.time()+60)
    assert result['complete'] is True and len(result['records'])==24
    assert result['resource_fit_admitted'] is False
    assert before=={row['path']:file_sha(row['path']) for row in receipt['records']}
    first=torch.load(result['records'][0]['path'],map_location='cpu',weights_only=False)
    assert first['data']['state'].shape==(1,32) and first['data']['features'].shape==(1,2062)
    assert first['data']['base14'].shape==(1,32,14) and first['data']['teacher14'].shape==(1,32,14)


@pytest.mark.parametrize('width',[14,31,33])
def test_model_state_abi_refuses_physical14_or_other_wrong_width(tmp_path,monkeypatch,width):
    value,path,_=model_state_fixture(tmp_path,monkeypatch,width)
    with pytest.raises(ValueError,match='Saved cache masks/labels/context'):
        bank.seal_full_cache(value,[dict(path=str(path),sha256=file_sha(path))],deadline=time.time()+60)
