"""Produce a per-key TSV on stdout and its summary on stderr from two frozen audit files."""
from collections import Counter, defaultdict
import csv
import json
import sys

receipts, survivors = json.load(sys.stdin.buffer)
groups = receipts["cross_host_keys"]
files = survivors["keys"]
assert set(groups) == set(files)
summary = {}
writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
writer.writerow(["key","publication_pair_at_cutoff","start1_utc_epoch","owner1_attempt/lane",
                 "owner1_host","publish1_utc_epoch","owner1_transitions","owner1_episodes",
                 "start2_utc_epoch","owner2_attempt/lane","owner2_host","publish2_utc_epoch",
                 "owner2_transitions","owner2_episodes","current_transitions","current_episodes",
                 "current_metadata_owner","npz_sha256","json_sha256","all_final_checks",
                 "before_after_stat_stable"])
for phase in ("collect", "babble"):
    c = Counter()
    stream_totals = defaultdict(Counter)
    for name, stream in receipts["streams"].items():
        for r in stream["rows"]:
            if r["unit"].split("/")[3] == phase:
                stream_totals[name.split("/")[2]][r["event"]] += 1
    for key in sorted(groups):
        if key.split("/")[3] != phase:
            continue
        group, item = groups[key], files[key]
        owners = defaultdict(dict)
        for r in group["rows"]:
            owner = (r["attempt"],r["lane"],r["host"])
            assert r["event"] not in owners[owner]
            owners[owner][r["event"]] = r
        assert len(owners)==2 and all("start" in rs for rs in owners.values())
        cells = [key,int(group["comparable"])]
        pair = sorted(owners.items(),key=lambda kv:kv[1]["start"]["utc"])
        for (attempt,lane,host), events in pair:
            p = events.get("published",{})
            cells += [events["start"]["utc"],attempt+"/"+lane,host,p.get("utc"),p.get("transitions"),p.get("episodes")]
        ok = bool(item.get("checks")) and all(item["checks"].values())
        explicit = item.get("sidecar_owner")
        cells += [item.get("transitions"),item.get("episodes"),"/".join(explicit) if explicit else "",
                  item.get("npz",{}).get("sha256"),item.get("sidecar",{}).get("sha256"),
                  ok if item["final_exists"] else "",item["stat_stable"]]
        writer.writerow(cells)
        c["cross_host_started"] += 1
        if item["final_exists"] and item["sidecar_exists"]:
            c["cross_host_final_pairs_read"] += 1
        if not group["comparable"]:
            incomplete = sum("published" in ev for owner,ev in pair)
            c[f"only_{incomplete}_publication_receipts"] += 1
            continue
        c["comparable_keys"] += 1
        c["stable_valid_surviving_pairs"] += int(ok and item["stat_stable"])
        c["report_pairs_equal_and_match_current"] += int(item["all_reported_counts_match"] and len(group["counts"])==1)
        c["transitions_counting_each_key_once"] += item["transitions"]
        c["episodes_counting_each_key_once"] += item["episodes"]
        c["overlapping_reported_intervals"] += int(pair[1][1]["start"]["utc"] < pair[0][1]["published"]["utc"])
        c["retained_attempt_specific_files"] += sum(map(len,item["partial_files"].values()))
        if phase=="collect":
            b = item["batch_evidence"]
            c["byte_equal_npz_reconstructed_from_surviving_checkpoints"] += int(b["rebuilt_npz_sha256"]==item["npz"]["sha256"])
            c["byte_equal_all_four_arrays"] += int(all(b["rebuilt_equals_final"].values()))
            logs = b["lane_log_batches"]
            cumulative = {}
            for name, rows in logs.items():
                cumulative[name] = {r["batch"]:r["cumulative"] for r in rows}
                assert len(cumulative[name])==len(rows)
            running = 0
            for cp in b["checkpoints"]:
                idx = cp["batch"]
                c["current_batch_archives"] += 1
                c["valid_batch_archives"] += int(all(cp[k] for k in ("zip_crc_ok","order_ok","identities_ok","trace_contract_ok")))
                running += sum(cp["lengths"])
                c["cumulative_steps_match_both_owner_logs"] += int(all(x.get(idx)==running for x in cumulative.values()))
                wrote = {owner for owner, rows in logs.items() for r in rows if r["batch"]==idx and r["event"]=="batch-checkpoint"}
                replay = {owner for owner, rows in logs.items() for r in rows if r["batch"]==idx and r["event"]=="replay"}
                c["both_lanes_logged_write"] += int(len(wrote)==2)
                c["one_logged_write_other_replay"] += int(len(wrote)==1 and len(replay)==1)
                c["archive_pid_unique_matches_logged_writer"] += int(len(cp["pid_lane_candidates"])==1 and
                                                                     cp["pid_lane_candidates"][0] in wrote)
    summary[phase] = dict(totals=c, per_attempt=stream_totals)
print(json.dumps(summary,indent=2,sort_keys=True),file=sys.stderr)
