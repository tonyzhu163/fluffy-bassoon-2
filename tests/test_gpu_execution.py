import pytest
from fluffy_bassoon.gpu_execution import validate_execution


def test_direct_is_explicit_and_cannot_bypass_scheduler():
    for env in ({}, {'CUDA_VISIBLE_DEVICES':'0,0'}, {'CUDA_VISIBLE_DEVICES':'all'},
                {'CUDA_VISIBLE_DEVICES':'0','SLURM_JOB_ID':'123'}):
        with pytest.raises(ValueError): validate_execution('direct', env)
    validate_execution('direct', {'CUDA_VISIBLE_DEVICES':'6,7'})
    validate_execution('direct', {'CUDA_VISIBLE_DEVICES':'GPU-ec4f2acc-ac94-6ac1-a464-0db0ebecaf24'})


def test_scheduler_default_still_requires_job_step():
    for env in ({}, {'SLURM_JOB_ID':'123','SLURM_STEP_ID':'batch'}):
        with pytest.raises(ValueError): validate_execution('slurm', env)
    validate_execution('slurm', {'SLURM_JOB_ID':'123','SLURM_STEP_ID':'0'})

