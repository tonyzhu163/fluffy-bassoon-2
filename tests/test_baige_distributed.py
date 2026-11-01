"""Scheduler node identity remains bound when using fragmented GPU nodes."""
import pytest

from fluffy_bassoon.baige_gpu import validate_container


def env():
    return dict(KUBERNETES_SERVICE_HOST='10.0.0.1', RANK='2', WORLD_SIZE='4',
                MASTER_ADDR='job-a1b2-master-0',
                NVIDIA_VISIBLE_DEVICES='GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee')


def test_distributed_worker_matches_scheduler_rank_and_master():
    assert validate_container(env(), 'job-a1b2-worker-1') == ['GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee']


def test_worker_cannot_claim_another_rank_or_master():
    for change in (dict(RANK='0'), dict(WORLD_SIZE='2'), dict(MASTER_ADDR='job-other-master-0')):
        with pytest.raises(ValueError, match='scheduler node rank'):
            validate_container(dict(env(), **change), 'job-a1b2-worker-1')


def test_unallocated_or_noncanonical_master_is_still_rejected():
    for hostname in ('job-a1b2-master-1', 'development-host', 'job-a1b2-worker-x'):
        with pytest.raises(ValueError):
            validate_container(env(), hostname)
