"""Summarize the finished, portable CPU evidence using only the Python standard library."""
import csv
import hashlib
import itertools
import json
from pathlib import Path
import struct
import tarfile

HERE = Path(__file__).resolve().parent
EV = HERE / "evidence"
ROWS = (34, 35, 36, 37)
ARMS = ("IIWA", "Kinova3")


def digest(commands):
    prefix = repr(((len(commands), 7), "<f4")).encode()
    return hashlib.sha256(prefix + b"".join(struct.pack("<7f", *a) for a in commands)).hexdigest()


def contrast(rows, baseline):
    pairs = [(r["layout"], int(r["correct"])-int(r[baseline])) for r in rows]
    effects = [sum(d for i,d in pairs if i == row)/sum(i == row for i,_ in pairs) for row in ROWS]
    draws = sorted(sum(effects[i] for i in draw)/4 for draw in itertools.product(range(4), repeat=4))
    return dict(n=len(rows), rescues=sum(d==1 for _,d in pairs), losses=sum(d==-1 for _,d in pairs),
                ties=sum(d==0 for _,d in pairs), difference=sum(effects)/4, by_layout=effects,
                ordered_4_layout_resample_95pct=[draws[6],draws[249]])


def main():
    rc, start, end = map(int, (EV/"exit.txt").read_text().split())
    assert rc == 0
    manifest = json.loads((HERE/"inputs/manifest.json").read_bytes())
    with tarfile.open(HERE/"inputs/raw-tapes.tar.gz") as archive:
        tapes = {}
        for arm,row in itertools.product(ARMS, ROWS):
            name = f"gpu01/episodes/{arm}-{row}-raw.json"
            blob = archive.extractfile(name).read()
            assert hashlib.sha256(blob).hexdigest() == manifest["members"][name]
            tapes[arm,row] = json.loads(blob)
    cases = {}
    for path in sorted((EV/"cases").glob("*-from-*.json")):
        r = json.loads(path.read_bytes())
        key = (r["target"],r["layout"],r["donor"],r["donor_layout"])
        assert key not in cases
        donor = tapes[key[2:]]
        commands = [a for call in donor["calls"] for a in call["sent"]]
        n = r["controls"]
        assert r["donor_controls"] == len(commands) == donor["native"]["controls"]
        assert r["donor_action_sha256"] == digest(commands) and r["submitted_action_sha256"] == digest(commands[:n])
        assert n == len(r["native_rewards"]) == len(r["native_success"]) == len(r["native_truncated"]) == len(r["native_terminated"])
        terminal = [i for i,(v,w) in enumerate(zip(r["native_terminated"], r["native_truncated"]),1) if v or w]
        success = [i for i,v in enumerate(r["native_success"],1) if v]
        assert terminal == ([n] if r["terminal_control"] is not None else []) and r["terminal_control"] == next(iter(terminal), None)
        assert success == ([n] if r["success"] else []) and r["first_success_control"] == next(iter(success), None)
        assert r["native_rewards"] == [float(s) for s in r["native_success"]]
        assert r["reward_sum"] == sum(r["native_rewards"]) == len(success)
        assert n <= len(commands) and (n == len(commands) or terminal)
        assert r["stop"] == ("success" if success else "horizon" if terminal else "tape_end")
        assert bool(r["native_truncated"][-1]) == (n == 300 and not success)
        assert r["donor_gpu_success"] == donor["native"]["success"]
        assert r["own_gpu_success"] == tapes[key[:2]]["native"]["success"]
        cases[key] = r
    assert len(cases) == 32
    primary, oracle = [], []
    for a,b in (ARMS, ARMS[::-1]):
        for row in ROWS:
            own = cases[a,row,a,row]
            correct = cases[a,row,b,row]
            sham = cases[a,row,b,ROWS[(ROWS.index(row)+1)%4]]
            local = [r for key,r in cases.items() if key[:2] == (a,row)]
            assert len(local) == (5 if a == "Kinova3" else 3)
            assert all(r["reset"] == own["reset"] and r["body"] == own["body"] for r in local)
            assert correct["noise_sha256"] == correct["donor_noise_sha256"] == own["noise_sha256"]
            assert sham["noise_sha256"] != sham["donor_noise_sha256"]
            assert [r["group"] for r in (own,correct,sham)] == ["own","correct","sham"]
            original = tapes[a,row]["native"]
            assert own["own_gpu_controls"] == original["controls"]
            matches = own["own_proprio_at_original_queries"]
            assert matches and [q["t"] for q in matches] == list(range(0, own["controls"], 5))
            primary.append(dict(target=a, layout=row, recorded=int(original["success"]), recorded_controls=original["controls"],
                own=int(own["success"]), correct=int(correct["success"]), sham=int(sham["success"]),
                own_stop=own["stop"], own_controls=own["controls"], own_rewards_equal=own["native_rewards"]==original["native_rewards"],
                own_end_flags_equal=(own["native_terminated"][-1],own["native_truncated"][-1])==(original["terminated"],original["truncated"]),
                own_proprio_equal_query_counts={k:sum(q["equal"][k] for q in matches) for k in matches[0]["equal"]}, own_queries=len(matches),
                correct_stop=correct["stop"], correct_controls=correct["controls"], sham_stop=sham["stop"], sham_controls=sham["controls"],
                gpu_physics_equal=own["gpu_physics_equal"], gpu_post_settle_scene_equal=own["gpu_post_settle_scene_equal"],
                cpu03_physics_equal=own["cpu03_physics_equal"]))
            if a == "Kinova3":
                trials = [cases[a,row,b,i] for i in ROWS]
                oracle.append(dict(layout=row, own=own["success"], recorded=original["success"],
                    donors_succeeding=[r["donor_layout"] for r in trials if r["success"]],
                    reward_by_donor=[r["reward_sum"] for r in trials], any_success=any(r["success"] for r in trials)))
    metrics = dict(wall_seconds=end-start, total_controls=sum(r["controls"] for r in cases.values()),
        stops={s:sum(r["stop"]==s for r in cases.values()) for s in ("success","horizon","tape_end")},
        primary=primary, kinova_oracle=oracle,
        contrasts={group:{base:contrast([r for r in primary if group=="both" or r["target"]==group],base)
                     for base in ("own","sham")} for group in (*ARMS,"both")})
    (EV/"summary.json").write_text(json.dumps(metrics, indent=2)+"\n")
    fields = ["target","layout","donor","donor_layout","group","own_gpu_success","donor_gpu_success","donor_controls",
              "controls","reward_sum","stop","terminal_control","first_success_control"]
    with (EV/"cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cases.values())


if __name__ == "__main__":
    main()
