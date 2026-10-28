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


@pytest.mark.parametrize('mode', ['direct', 'slurm'])
def test_direct_egl_filters_by_visible_cuda_ordinal_without_creating_context(mode):
    import ctypes
    import sys
    from types import SimpleNamespace
    from unittest.mock import patch
    from fluffy_bassoon.renderer_binding import configure_direct_egl
    signature=ctypes.CFUNCTYPE(ctypes.c_uint,ctypes.c_void_p,ctypes.c_int,ctypes.POINTER(ctypes.c_ssize_t))
    @signature
    def query(device,attribute,result):
        assert attribute==0x323A
        if device==2:result[0]=0;return 1
        return 0
    egl=SimpleNamespace(eglGetProcAddress=lambda _:ctypes.cast(query,ctypes.c_void_p).value,
                        eglQueryDevicesEXT=lambda:[1,2,3])
    env={'FLUFFY_GPU_EXECUTION':mode,'CUDA_VISIBLE_DEVICES':'7','MUJOCO_EGL_DEVICE_ID':'0'}
    if mode == 'slurm':
        env.update(SLURM_JOB_ID='123', SLURM_STEP_ID='0')
    def import_binding(name):
        import os
        assert name=='robosuite.utils.binding_utils'
        assert 'MUJOCO_EGL_DEVICE_ID' not in os.environ
        assert os.environ['CUDA_VISIBLE_DEVICES']=='7'
    with patch.dict('os.environ',env,clear=True), patch.dict(sys.modules,{'mujoco.egl':SimpleNamespace(egl_ext=egl)}), patch('importlib.import_module',side_effect=import_binding):
        record=configure_direct_egl()
        import os
        assert os.environ['MUJOCO_EGL_DEVICE_ID']=='0'
    assert record['selected_original_index']==1
    assert record['local_egl_index']==0
    assert egl.eglQueryDevicesEXT()==[2]
