"""Seal the parent's approved prelaunch panel from real offline score receipts, without submitting a job."""
import argparse
from pathlib import Path
from protocol import atomic, digest, encode, require


def asset(path):
    path = path.resolve(strict=True)
    return dict(path=str(path),sha256=digest(path.read_bytes()))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--training-seal", type=Path, required=True)
    p.add_argument("--physical-receipt", type=Path, action="append", required=True)
    p.add_argument("--geometry-evidence", type=Path, required=True)
    p.add_argument("--offline-physics-evidence", type=Path, required=True)
    p.add_argument("--digital-receipt", type=Path)
    p.add_argument("--controls", nargs="*", choices=("nominal","identity","true_map"), default=[])
    a = p.parse_args()
    require(len(set(a.controls)) == len(a.controls), "duplicate control")
    result = dict(seed=41, decision="release_native", training_seal=asset(a.training_seal),
                  physical_receipts=list(map(asset,a.physical_receipt)), geometry_evidence=asset(a.geometry_evidence),
                  offline_physics_evidence=asset(a.offline_physics_evidence),
                  digital_receipt=asset(a.digital_receipt) if a.digital_receipt else None, optional_controls=a.controls)
    atomic(a.out, encode(result))
    print(digest(a.out.read_bytes()))
