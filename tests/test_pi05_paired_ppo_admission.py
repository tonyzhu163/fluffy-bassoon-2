"""Paired PPO admission requires explicitly pinned reviewed live gate bytes."""

import json

import pytest

from fluffy_bassoon.evaluation_stages import sha256
from fluffy_bassoon.meta_rl.pi05_bridge_train import load_paired_gate_receipt
from fluffy_bassoon.meta_rl.pi05_paired_ppo_plan import GATE_PROTOCOL


def reviewed_receipt():
    # Synthetic exploratory n=2 evidence. Admission checks provenance and the
    # explicit reviewed decision, without inventing an automatic numeric rule.
    return dict(protocol=GATE_PROTOCOL, status="passed", complete=True,
        evidence_kind="live_same_interface", sampler="StochasticPi05BridgePolicy.sample_chunk",
        deterministic=True, denoise_steps=4, action_chunk=5, active_dims=7, residual_scale=1.,
        core="persistent_ctm", width=64, ticks=4, source_commit="a" * 40,
        evidence_sha256="b" * 64, motor_sha256="c" * 64,
        norm_stats_sha256="d" * 64, rlinf_sources={"native": "e" * 64},
        evaluated_theta_degrees=[-45., 0., 45.],
        decision="Synthetic fixture: reviewed exploratory competence, no automatic threshold",
        comparisons={str(theta): dict(bridge=[[1, 0]], zero_bridge=[[0, 0]],
                                     external_oracle=[[1, 0]]) for theta in (-45., 0., 45.)})


def write_receipt(tmp_path, receipt=None):
    path = tmp_path / "reviewed-gate.json"
    path.write_text(json.dumps(reviewed_receipt() if receipt is None else receipt, allow_nan=False) + "\n")
    return path, sha256(path)


@pytest.mark.parametrize("missing", ["path", "hash", "both"])
def test_paired_gate_loader_requires_both_external_path_and_hash(tmp_path, missing):
    path, digest = write_receipt(tmp_path)
    with pytest.raises(ValueError):
        load_paired_gate_receipt(None if missing in ("path", "both") else path,
                                 None if missing in ("hash", "both") else digest)


@pytest.mark.parametrize("digest", ["", "bad", "A" * 64, "0" * 64])
def test_paired_gate_loader_rejects_malformed_or_wrong_declared_hash(tmp_path, digest):
    path, _ = write_receipt(tmp_path)
    with pytest.raises(ValueError):
        load_paired_gate_receipt(path, digest)


def test_paired_gate_loader_rejects_modified_receipt_bytes(tmp_path):
    path, digest = write_receipt(tmp_path)
    receipt = reviewed_receipt()
    receipt["decision"] += "; edited after hashing"
    path.write_text(json.dumps(receipt) + "\n")
    with pytest.raises(ValueError):
        load_paired_gate_receipt(path, digest)


@pytest.mark.parametrize("changes", [dict(status="unresolved"), dict(complete=False),
                                     dict(evidence_kind="offline_flow_loss")])
def test_checksum_does_not_promote_incomplete_or_loss_only_evidence(tmp_path, changes):
    path, digest = write_receipt(tmp_path, dict(reviewed_receipt(), **changes))
    with pytest.raises(ValueError):
        load_paired_gate_receipt(path, digest)


def test_paired_gate_loader_accepts_reviewed_small_n_without_numeric_promotion_rule(tmp_path):
    path, digest = write_receipt(tmp_path)
    assert load_paired_gate_receipt(path, digest) == reviewed_receipt()
