"""Resource-only regression; execute on an allocated Linux CPU worker."""
import json

import pytest

from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin, finish


def test_legacy_minimum_is_preserved(tmp_path):
    with pytest.raises(ValueError):
        begin(tmp_path, {}, slice_seconds=437, worker_seconds=600)
    assert not (tmp_path / 'worker-time.json').exists()


def test_opt_in_short_slice_preserves_prior_charge_and_fixed_ceiling(tmp_path):
    contract = {'phase': 'resource-only'}
    old = begin(tmp_path, contract, slice_seconds=60, worker_seconds=600,
                now=0, allow_short_slice=True)
    assert finish(tmp_path, old, 1, now=43)['charged_seconds'] == 103
    child = begin(tmp_path, contract, slice_seconds=437, worker_seconds=600,
                  resume=True, now=50, allow_short_slice=True)
    ledger = json.loads((tmp_path / 'worker-time.json').read_text())
    assert ledger['attempts'][0]['charged_seconds'] == 103
    assert ledger['attempts'][1]['reserved_seconds'] == 497
    assert ledger['worker_seconds'] == 600
    with pytest.raises(RuntimeError):
        begin(tmp_path, contract, slice_seconds=60, worker_seconds=600,
              resume=True, now=51, allow_short_slice=True)
    finish(tmp_path, child, 0, now=400)


def test_short_slice_rejects_ambiguous_or_nonpositive_admission(tmp_path):
    with pytest.raises(ValueError):
        begin(tmp_path, {}, slice_seconds=437, worker_seconds=600,
              allow_short_slice=1)
    with pytest.raises(ValueError):
        begin(tmp_path, {}, slice_seconds=59, worker_seconds=600,
              allow_short_slice=True)
