"""Prepared CPU-only adapter tests; allocation owner executes them on Linux.

The injected backend and native oracle import no JAX/OpenPI or model weights.
These tests cannot establish allocated native-vs-cached numerical parity.
"""
from copy import deepcopy
import builtins
import importlib
import pickle
from types import SimpleNamespace

import numpy as np
import pytest


def observation():
    return dict(image=np.arange(18,dtype=np.uint8).reshape(2,3,3),
                wrist=np.arange(18,36,dtype=np.uint8).reshape(2,3,3),
                state=np.arange(8,dtype=np.float32)/10)


def features(obs,instruction):
    visual=float(np.asarray(obs["image"],np.float64).sum())/1000
    language=float(np.asarray(obs["wrist"],np.float64).sum()+sum(str(instruction).encode()))/1000
    return np.concatenate((np.full(2048,visual,np.float32),
                           np.full(2048,language,np.float32),np.asarray(obs["state"],np.float32)[:8]))


def split(key):
    value=np.asarray(key,np.uint32)
    return value+np.array([1,3],np.uint32),value+np.array([101,103],np.uint32)


def actions(obs,instruction,noise):
    offset=np.float32(features(obs,instruction)[:4096].sum()/4096)
    return (np.asarray(noise,np.float32)[:,:7]+offset+np.asarray(obs["state"],np.float32)[:7]).astype(np.float32)


class CanonicalFeatures:
    def __init__(self):
        self.calls=0; self.cache_hits=0; self.elapsed_seconds=0.; self.qualify_calls=0
        self._last_obs=self._last_instruction=self._last_features=None
        self.qualify_error=None
        self.contract=dict(kind="openpi_final_prefix_visual_mean_language_mean_public_state_v2",
            observation_dim=4104,visual_dim=2048,instruction_dim=2048,public_state_dim=8,
            trainable_feature_parameters=0,kv_cache_retained=False,
            pooling="separate NumPy float32 mask-weighted means after native hidden materialization",
            execution="standalone native prefix hidden materialized on host before pooling; native prefix recomputed for motor; both costs recorded")

    def __call__(self,obs,instruction):
        if self._last_obs is obs and self._last_instruction==instruction:
            self.cache_hits+=1
            return self._last_features.copy()
        self.calls+=1; self.elapsed_seconds+=.01
        self._last_obs,self._last_instruction=obs,instruction
        self._last_features=features(obs,instruction)
        return self._last_features.copy()

    def qualify(self,obs,instruction):
        self.qualify_calls+=1
        if self.qualify_error is not None: raise self.qualify_error
        value=self(obs,instruction)
        return dict(passed=True,maximum_delta=0.,public_feature_shape=list(value.shape),
                    feature_contract=deepcopy(self.contract))


class NativeMotor:
    def __init__(self,horizon=10):
        self.horizon=horizon; self.calls=0; self.native_infers=0; self.native_attestations=0
        self.native_splits=0; self.feature_encoder=CanonicalFeatures()
        self.metadata=dict(openpi_commit="pinned",receipt_sha256="a"*64,flow_steps=10,
                           predicted_horizon=horizon,action_dim=32,weights_frozen=True)
        self.policy=SimpleNamespace(_model=SimpleNamespace(weights={
            "prefix":np.arange(12,dtype=np.float32).reshape(3,4),
            "suffix":np.arange(28,dtype=np.float32).reshape(4,7)}),
            _rng=np.array([7,11],np.uint32),_sample_kwargs=dict(num_steps=10),
            _input_transform=lambda value:value,_output_transform=lambda value:value,
            _sample_actions=lambda *args,**kwargs:None)

    def infer(self,obs,instruction,noise):
        self.calls+=1; self.native_infers+=1; self.native_splits+=1
        self.policy._rng,_=split(self.policy._rng)
        result=actions(obs,instruction,noise)
        if self.calls==1: self.native_attestations+=1
        return result


class Backend:
    def __init__(self,native):
        self.native=native; self.prepares=0; self.decodes=0; self.splits=0; self.attestations=0
        self.prepare_error=self.decode_error=self.attest_error=None
        self.bad_features=self.bad_actions=None
        self.decoded_noises=[]

    def prepare(self,obs,instruction):
        self.prepares+=1
        if self.prepare_error is not None: raise self.prepare_error
        value=features(obs,instruction)
        if self.bad_features=="shape": value=value[:-1]
        if self.bad_features=="nan": value[0]=np.nan
        return SimpleNamespace(features=value,obs=deepcopy(obs),instruction=str(instruction))

    def split_rng(self,key):
        self.splits+=1
        return split(key)

    def decode(self,prepared,noise):
        self.decodes+=1
        self.decoded_noises.append(np.asarray(noise).copy())
        if self.decode_error is not None: raise self.decode_error
        value=actions(prepared.obs,prepared.instruction,noise)
        if self.bad_actions=="shape": value=value[:-1]
        if self.bad_actions=="nan": value[0,0]=np.nan
        return value

    def attest_first_call(self):
        self.attestations+=1
        if self.attest_error is not None: raise self.attest_error


@pytest.fixture
def make(monkeypatch):
    original=builtins.__import__
    def no_models(name,*args,**kwargs):
        if name.split(".")[0] in {"jax","jaxlib","openpi","openpi_client"}:
            raise AssertionError("CPU fake path imported model runtime: "+name)
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",no_models)
    module=importlib.import_module("fluffy_bassoon.meta_rl.vla_cached_scalar_motor")
    def create(horizon=10):
        native=NativeMotor(horizon); backend=Backend(native)
        motor=module.CachedScalarMotor(native,backend=backend)
        return SimpleNamespace(motor=motor,native=native,backend=backend,obs=observation(),
            instruction="put the cup on the plate",noise=np.full((horizon,32),.125,np.float32))
    return create


def test_prepare_reuses_equal_full_content_without_rng_decode_or_mutable_feature_alias(make):
    case=make(); key=case.native.policy._rng
    first=case.motor.feature_encoder(case.obs,case.instruction)
    expected=first.copy(); first[:]=-999
    again=case.motor.feature_encoder(deepcopy(case.obs),case.instruction)
    np.testing.assert_array_equal(again,expected)
    assert case.backend.prepares==1 and case.backend.decodes==case.backend.splits==0
    assert case.native.policy._rng is key and case.motor.calls==0
    assert case.motor.feature_encoder.calls==case.native.feature_encoder.calls==1
    assert case.motor.feature_encoder.cache_hits>=1


def test_cached_decode_matches_native_oracle_and_consumes_one_rng_and_attestation(make):
    horizon=10; case=make(horizon); oracle=NativeMotor(horizon)
    expected=oracle.infer(case.obs,case.instruction,case.noise)
    case.motor.feature_encoder(case.obs,case.instruction)
    result=case.motor.infer(deepcopy(case.obs),case.instruction,case.noise)
    np.testing.assert_array_equal(result,expected)
    np.testing.assert_array_equal(case.native.policy._rng,oracle.policy._rng)
    assert result.shape==(horizon,7)
    assert case.backend.prepares==case.backend.decodes==case.backend.splits==1
    assert case.backend.attestations==1 and case.native.native_infers==0
    assert case.motor.calls==case.native.calls==1
    # Consumed KV is never reused; normal native compatibility is still allowed.
    case.motor.infer(case.obs,case.instruction,case.noise)
    assert case.backend.decodes==case.backend.splits==1 and case.native.native_infers==1
    assert case.motor.calls==2 and case.backend.attestations==1


def test_cached_adapter_rejects_unsupported_h50_without_model_or_rng_work(make):
    with pytest.raises(ValueError,match="H10"):
        make(50)


def test_no_pending_prefix_keeps_initial_and_repeated_native_infer_compatibility(make):
    case=make()
    left=case.motor.infer(case.obs,case.instruction,case.noise)
    right=case.motor.infer(case.obs,case.instruction,case.noise)
    np.testing.assert_array_equal(left,right)
    assert case.native.native_infers==case.native.native_splits==case.motor.calls==2
    assert case.native.native_attestations==1
    assert case.backend.prepares==case.backend.decodes==case.backend.splits==case.backend.attestations==0


def test_source_sac_unbounded_gaussian_warmup_noise_is_not_clipped_or_rescaled(make):
    case=make()
    coordinate=np.linspace(-4,4,32,dtype=np.float32)
    noise=np.repeat(coordinate[None],10,axis=0)
    assert np.abs(noise).max()>1
    expected=NativeMotor().infer(case.obs,case.instruction,noise)
    case.motor.feature_encoder(case.obs,case.instruction)
    actual=case.motor.infer(case.obs,case.instruction,noise)
    np.testing.assert_array_equal(case.backend.decoded_noises[0],noise)
    np.testing.assert_array_equal(actual,expected)
    assert case.backend.splits==case.backend.decodes==1


def test_source_sac_post_infer_feature_reads_reuse_host_features_without_extra_prefix(make):
    case=make()
    initial=case.motor.feature_encoder(case.obs,case.instruction)
    case.motor.infer(case.obs,case.instruction,case.noise)
    assert not case.motor.cache_pending
    # OnlineSourceSimulator.step requests pending-observation features after
    # motor inference, before encoding the next public observation.
    retained=case.motor.feature_encoder(deepcopy(case.obs),case.instruction)
    np.testing.assert_array_equal(retained,initial)
    assert case.backend.prepares==1 and not case.motor.cache_pending
    following=deepcopy(case.obs); following["state"][0]+=1
    case.motor.feature_encoder(following,case.instruction)
    assert case.backend.prepares==2 and case.motor.cache_pending
    case.motor.infer(following,case.instruction,case.noise)
    assert case.backend.prepares==case.backend.decodes==case.backend.splits==2
    assert case.native.native_infers==0 and case.motor.calls==2
    expected,_=split(split(np.array([7,11],np.uint32))[0])
    np.testing.assert_array_equal(case.motor.policy._rng,expected)


def test_consumed_kv_is_not_revived_by_repeated_host_feature_reads(make):
    case=make(); expected=case.motor.feature_encoder(case.obs,case.instruction)
    case.motor.infer(case.obs,case.instruction,case.noise)
    for _ in range(2):
        retained=case.motor.feature_encoder(deepcopy(case.obs),case.instruction)
        np.testing.assert_array_equal(retained,expected)
        assert not case.motor.cache_pending
        case.motor.infer(case.obs,case.instruction,case.noise)
    assert case.backend.prepares==case.backend.decodes==case.backend.splits==1
    assert case.native.native_infers==case.native.native_splits==2
    assert case.motor.calls==3


@pytest.mark.parametrize("operation",["clear","restore","qualify"])
def test_explicit_boundaries_invalidate_retained_host_features_as_well_as_kv(make,operation):
    case=make(); case.motor.feature_encoder(case.obs,case.instruction)
    case.motor.infer(case.obs,case.instruction,case.noise)
    before=case.motor.policy._rng.copy()
    if operation=="clear": case.motor.clear_cache()
    elif operation=="restore": case.motor.restore_motor_rng(before.copy())
    else: case.motor.feature_encoder.qualify(case.obs,case.instruction)
    np.testing.assert_array_equal(case.motor.policy._rng,before)
    assert case.backend.prepares==case.backend.decodes==case.backend.splits==1
    case.motor.feature_encoder(case.obs,case.instruction)
    assert case.backend.prepares==2 and case.motor.cache_pending


@pytest.mark.parametrize("operation",["features","infer"])
def test_model_identity_error_invalidates_retained_host_features_before_recovery(make,operation):
    case=make(); case.motor.feature_encoder(case.obs,case.instruction)
    case.motor.infer(case.obs,case.instruction,case.noise)
    original=case.native.policy._model; before=case.native.policy._rng.copy()
    case.native.policy._model=deepcopy(original)
    with pytest.raises((ValueError,RuntimeError)):
        if operation=="features": case.motor.feature_encoder(case.obs,case.instruction)
        else: case.motor.infer(case.obs,case.instruction,case.noise)
    np.testing.assert_array_equal(case.native.policy._rng,before)
    assert case.backend.prepares==case.backend.decodes==case.backend.splits==1
    case.native.policy._model=original
    case.motor.feature_encoder(case.obs,case.instruction)
    assert case.backend.prepares==2 and case.motor.cache_pending


@pytest.mark.parametrize("changed",["image","wrist","state","instruction","dtype","shape","scalar_shape"])
def test_present_prefix_rejects_changed_current_input_before_rng_or_decode(make,changed):
    case=make()
    if changed=="scalar_shape": case.obs["extra_scalar"]=np.array(3,np.int32)
    case.motor.feature_encoder(case.obs,case.instruction)
    before=case.native.policy._rng.copy(); instruction=case.instruction
    if changed in ("image","wrist"): case.obs[changed][0,0,0]^=1
    elif changed=="state": case.obs["state"][-1]+=1
    elif changed=="instruction": instruction+=" now"
    elif changed=="dtype": case.obs["state"]=case.obs["state"].astype(np.float64)
    elif changed=="shape": case.obs["image"]=case.obs["image"].reshape(3,2,3)  # Same bytes, different shape.
    else: case.obs["extra_scalar"]=np.array([3],np.int32)
    with pytest.raises((ValueError,RuntimeError)):
        case.motor.infer(case.obs,instruction,case.noise)
    np.testing.assert_array_equal(case.native.policy._rng,before)
    assert case.backend.decodes==case.backend.splits==case.native.native_infers==case.motor.calls==0
    # A stale failure clears the cache, with no hidden retry in that failed call.
    case.motor.infer(case.obs,instruction,case.noise)
    assert case.native.native_infers==1 and case.backend.decodes==0


@pytest.mark.parametrize("invalid",["width","horizon","nan","inf"])
def test_invalid_cached_noise_never_consumes_rng_or_attempts_suffix(make,invalid):
    case=make(); case.motor.feature_encoder(case.obs,case.instruction)
    before=case.native.policy._rng.copy(); value=case.noise.copy()
    if invalid=="width": value=value[:,:31]
    elif invalid=="horizon": value=value[:-1]
    else: value[0,0]=np.nan if invalid=="nan" else np.inf
    with pytest.raises((ValueError,RuntimeError)):
        case.motor.infer(case.obs,case.instruction,value)
    np.testing.assert_array_equal(case.native.policy._rng,before)
    assert case.backend.splits==case.backend.decodes==case.native.native_infers==0
    case.motor.feature_encoder(case.obs,case.instruction)
    assert case.backend.prepares==2  # Validation error must discard host features too.


@pytest.mark.parametrize("failure",["suffix","shape","nan","attestation"])
def test_attempted_suffix_failure_is_charged_once_and_never_retried(make,failure):
    case=make(); case.motor.feature_encoder(case.obs,case.instruction)
    expected,_=split(case.native.policy._rng)
    if failure=="suffix": case.backend.decode_error=RuntimeError("native suffix failed")
    elif failure=="attestation": case.backend.attest_error=RuntimeError("binding failed")
    else: case.backend.bad_actions=failure
    with pytest.raises((ValueError,RuntimeError)):
        case.motor.infer(case.obs,case.instruction,case.noise)
    np.testing.assert_array_equal(case.native.policy._rng,expected)
    assert case.backend.splits==case.backend.decodes==case.motor.calls==1
    assert case.native.native_infers==0
    case.backend.decode_error=case.backend.attest_error=case.backend.bad_actions=None
    case.motor.infer(case.obs,case.instruction,case.noise)
    assert case.native.native_infers==1 and case.backend.decodes==case.backend.splits==1
    case.motor.feature_encoder(case.obs,case.instruction)
    assert case.backend.prepares==2  # Failed attempt cannot leave reusable host state.


@pytest.mark.parametrize("failure",["raise","shape","nan"])
def test_failed_prefix_prepare_clears_previous_ephemeral_cache_without_rng(make,failure):
    case=make(); case.motor.feature_encoder(case.obs,case.instruction)
    changed=deepcopy(case.obs); changed["state"][0]+=1
    if failure=="raise": case.backend.prepare_error=RuntimeError("prefix failed")
    else: case.backend.bad_features=failure
    before=case.native.policy._rng.copy()
    with pytest.raises((ValueError,RuntimeError)):
        case.motor.feature_encoder(changed,case.instruction)
    np.testing.assert_array_equal(case.native.policy._rng,before)
    assert case.backend.splits==case.backend.decodes==0
    case.backend.prepare_error=case.backend.bad_features=None
    case.motor.infer(case.obs,case.instruction,case.noise)
    assert case.native.native_infers==1 and case.backend.decodes==0


@pytest.mark.parametrize("fails",[False,True])
def test_feature_qualification_delegates_original_gate_and_leaves_no_pending_cache(make,fails):
    case=make(); case.motor.feature_encoder(case.obs,case.instruction)
    case.native.feature_encoder(case.obs,case.instruction)
    before=case.native.policy._rng.copy()
    if fails:
        case.native.feature_encoder.qualify_error=RuntimeError("original 2e-4 feature gate failed")
        with pytest.raises(RuntimeError,match="original 2e-4"):
            case.motor.feature_encoder.qualify(case.obs,case.instruction)
    else:
        result=case.motor.feature_encoder.qualify(case.obs,case.instruction)
        assert result["passed"] and result["feature_contract"]==case.native.feature_encoder.contract
    assert case.native.feature_encoder.qualify_calls==1
    np.testing.assert_array_equal(case.native.policy._rng,before)
    assert case.backend.splits==case.backend.decodes==0
    assert case.native.feature_encoder._last_obs is None
    case.motor.infer(case.obs,case.instruction,case.noise)
    assert case.native.native_infers==1 and case.backend.decodes==0


def test_explicit_clear_and_equal_value_rng_restore_invalidate_ephemeral_prefix(make):
    case=make(); case.motor.feature_encoder(case.obs,case.instruction)
    restored=case.native.policy._rng.copy()
    case.motor.restore_motor_rng(restored)
    assert case.native.policy._rng is restored
    assert case.backend.splits==case.backend.decodes==0
    case.motor.infer(case.obs,case.instruction,case.noise)
    assert case.native.native_infers==1 and case.backend.decodes==0
    case.motor.feature_encoder(case.obs,case.instruction)
    case.motor.clear_cache(); case.motor.clear_cache()
    case.motor.infer(case.obs,case.instruction,case.noise)
    assert case.native.native_infers==2 and case.backend.decodes==0


def test_ephemeral_kv_cannot_be_silently_serialized_into_a_recovery(make):
    case=make(); case.motor.feature_encoder(case.obs,case.instruction)
    with pytest.raises(TypeError,match="ephemeral"):
        pickle.dumps(case.motor)
    assert case.backend.splits==case.backend.decodes==0


def test_direct_native_rng_assignment_rejects_pre_restore_prefix_even_for_equal_values(make):
    case=make(); case.motor.feature_encoder(case.obs,case.instruction)
    restored=case.native.policy._rng.copy(); case.native.policy._rng=restored
    with pytest.raises((ValueError,RuntimeError)):
        case.motor.infer(case.obs,case.instruction,case.noise)
    assert case.native.policy._rng is restored
    assert case.backend.decodes==case.backend.splits==case.native.native_infers==0


@pytest.mark.parametrize("changed",["policy","model","input_transform","output_transform","steps","metadata","contract"])
def test_frozen_runtime_identity_change_is_rejected_before_cached_decode(make,changed):
    case=make(); case.motor.feature_encoder(case.obs,case.instruction)
    if changed=="policy": case.native.policy=deepcopy(case.native.policy)
    elif changed=="model": case.native.policy._model=deepcopy(case.native.policy._model)
    elif changed=="input_transform": case.native.policy._input_transform=lambda x:x
    elif changed=="output_transform": case.native.policy._output_transform=lambda x:x
    elif changed=="steps": case.native.policy._sample_kwargs["num_steps"]=5
    elif changed=="metadata": case.native.metadata["flow_steps"]=5
    else: case.native.feature_encoder.contract["pooling"]="changed"
    with pytest.raises((ValueError,RuntimeError)):
        case.motor.infer(case.obs,case.instruction,case.noise)
    assert case.backend.decodes==case.backend.splits==case.native.native_infers==0


def test_wrapper_preserves_native_policy_model_metadata_feature_contract_and_parameter_shapes(make):
    case=make(); original_policy=case.native.policy; original_model=original_policy._model
    weights=deepcopy(original_model.weights); metadata=deepcopy(case.native.metadata)
    contract=deepcopy(case.native.feature_encoder.contract)
    case.motor.feature_encoder(case.obs,case.instruction)
    case.motor.infer(case.obs,case.instruction,case.noise)
    assert case.motor.policy is case.native.policy is original_policy
    assert case.motor.policy._model is original_model
    assert case.motor.horizon==case.native.horizon==10
    assert case.motor.metadata==case.native.metadata==metadata
    assert case.motor.feature_encoder.contract==case.native.feature_encoder.contract==contract
    assert contract["trainable_feature_parameters"]==0 and contract["kv_cache_retained"] is False
    assert isinstance(case.motor.execution_metadata,dict)
    assert case.motor.feature_encoder.elapsed_seconds>=0
    for name,before in weights.items():
        assert original_model.weights[name].shape==before.shape
        np.testing.assert_array_equal(original_model.weights[name],before)
