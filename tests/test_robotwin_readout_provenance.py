"""Only the diagnosed contextmanager provenance path; allocated Linux CPU."""
from contextlib import contextmanager,nullcontext
import inspect
from pathlib import Path
import sys

import pytest

from fluffy_bassoon.meta_rl import robotwin_command_diagnostic_readout as readout

pytestmark=pytest.mark.skipif(sys.platform!='linux',reason='Actual allocated Linux CPU only')
ROOT=Path(readout.__file__).resolve().parents[3]
REL='src/fluffy_bassoon/meta_rl/robotwin_prelift_probe.py'


def expected():return {REL:readout.file_sha(ROOT/REL)}


def test_exact_approved_contextmanager_uses_original_repository_source_and_hash():
    # This confirms the exact failed object without entering its generator or physics.
    wrapped=Path(inspect.getfile(readout.observe_native_plans)).resolve()
    assert wrapped.name=='contextlib.py' and not wrapped.is_relative_to(ROOT)
    assert readout.checked_runtime_source(readout.observe_native_plans,ROOT,expected())==ROOT/REL


def test_unapproved_wrapper_cannot_bypass_containment_via_approved_wrapped_target():
    @contextmanager
    def foreign():yield None
    foreign.__wrapped__=inspect.unwrap(readout.observe_native_plans)
    foreign.__name__=readout.observe_native_plans.__name__
    foreign.__module__=readout.observe_native_plans.__module__
    # A generic unwrap would accept the approved target. This wrapper is not approved.
    assert foreign is not readout.observe_native_plans
    with pytest.raises(ValueError):readout.checked_runtime_source(foreign,ROOT,expected())


def test_approved_unwrap_cannot_escape_repository(monkeypatch):
    monkeypatch.setattr(readout.observe_native_plans,'__wrapped__',nullcontext)
    with pytest.raises(ValueError):readout.checked_runtime_source(readout.observe_native_plans,ROOT,expected())


def test_approved_unwrap_still_requires_the_same_exact_source_hash():
    with pytest.raises(ValueError,match='Loaded readout/runtime blob differs'):
        readout.checked_runtime_source(readout.observe_native_plans,ROOT,{REL:'0'*64})
