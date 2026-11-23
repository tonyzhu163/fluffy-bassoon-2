"""Independently verify the joined panel export and pack the unrounded values for the harness."""
from pathlib import Path
import gzip
import hashlib
import json
import math
from statistics import mean

HERE = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_set(rows, key):
    return digest("".join(r[key] + "\n" for r in rows).encode())


def main():
    config = json.loads((HERE / "QUERY_CONFIG.json").read_text())
    output = HERE / "evidence"
    output.mkdir(exist_ok=True)
    summary = {}
    for panel, contract in config["panels"].items():
        attempt = "cpu02" if panel == "gpu08pi0fix" else "cpu03"
        directory = HERE / attempt / panel
        raw = (directory / "cases.json").read_bytes()
        data = json.loads(raw)
        rows = data["rows"]
        resets = data["resets"]
        assert data["pin"] == contract["source_pin"] and data["native_reader_qualified"]
        assert len(rows) == 96 and len(resets) == 8
        assert all(len({r[k] for r in rows}) == 96 for k in ("index", "active_sha256", "support_sha256"))
        available = [r for r in rows if r["fk_available"]]
        assert data["qualified"] == (len(available) == 96)
        reader = json.loads((HERE / "cpu02" / panel / "reader.json").read_text())["rows"]
        for i, row in enumerate(rows):
            reset = resets[i // 12]
            assert all(row[k] == v for k, v in reader[i].items())
            assert row["index"] == i and row["shard"] == reset["shard"] == i // 12
            assert all(reset[k] == row[k] for k in ("suite", "task_id", "support_seed", "support_layout"))
            assert reset["archived_physics"] == row["physics"]
            assert row["native_cpu_identity"] and row["reader_available"]
            assert row["support_layout"] == 11 and row["query_layout"] == 13 and row["query_seed"] + 817117 == row["noise_seed"]
            assert math.isfinite(row["yaw_hat"]) and row["scale_hat"] > 0 and row["grip_hat"] in (-1, 1)
            if not row["fk_available"]:
                assert row["fk_yaw_hat"] is None and (not reset["active_replay"]["matched"] or row["frame0_max_abs"] > 2e-6)
                continue
            pairs = list(zip(row["issued_xy"], row["world_delta_xy_m"]))
            assert len(pairs) == 3
            dot = sum(ax*bx + ay*by for (ax, ay), (bx, by) in pairs)
            cross = sum(ax*by - ay*bx for (ax, ay), (bx, by) in pairs)
            assert math.isclose(math.degrees(math.atan2(cross,dot)), row["fk_yaw_hat"], rel_tol=0, abs_tol=1e-10)
        name = panel + "-cases.json.gz"
        compressed = gzip.compress(raw, mtime=0)
        (output / name).write_bytes(compressed)
        summary[panel] = dict(artifact=name, sha256=digest(compressed), json_sha256=digest(raw),
            run_attempt=attempt, run_pin=(directory / "launch.txt").read_text().split()[0],
            support_set_sha256=file_set(rows,"support_sha256"), active_set_sha256=file_set(rows,"active_sha256"),
            cases=len(rows), fk_available=sum(r["fk_available"] for r in rows),
            native_reader_identity=sum(r["native_cpu_identity"] for r in rows),
            historical_success=sum(r["historical"]["success"] for r in rows),
            historical_rounding_matches=sum(r["historical_rounding_match"] for r in rows),
            max_abs_yaw_residual_deg=max(abs(r["yaw_residual_deg"]) for r in rows),
            max_abs_scale_residual=max(abs(r["scale_residual"]) for r in rows),
            bitwise_resets=sum(r["bitwise_reset"] for r in resets),
            pi0_quaternion_flips=sum(r["quaternion_flipped"] for r in resets),
            replay_joint_max_rad=max(r["active_replay"]["joint_max_abs_rad"] for r in resets),
            replay_tcp_max_m=max(r["active_replay"]["tcp_max_abs_m"] for r in resets),
            fk_mae_deg=mean(r["fk_yaw_error_deg"] for r in available),
            reader_paired_mae_deg=mean(r["reader_yaw_error_deg"] for r in available),
            reader_mae_deg=mean(r["reader_yaw_error_deg"] for r in rows),
            max_xy_length_error_mm=max(r["xy_length_max_error_mm"] for r in available),
            max_z_error_mm=max(r["z_max_error_mm"] for r in available))
    (output / "summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
