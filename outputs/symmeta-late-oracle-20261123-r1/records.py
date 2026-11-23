"""Validate native triplets before exposing any timing or reader effect."""
import math
from paired_records import HASH, SETTINGS, atomic, checked, encode, parse, require
from panel import ARMS, IDENTITY, differences

CONTRASTS = (("exact1","exact11"), ("exact11","stream11"), ("exact1","stream11"))


def validate(row, arms, ref, *, cpu=False):
    require(set(arms) == set(ARMS), "Unexpected arms")
    first = arms["exact1"]
    reset = first["reset"]
    require(all(reset[k] == ref["reset"][k] for k in SETTINGS) and
            reset["physics"] != row["support_physics"] and HASH.fullmatch(reset["physics"]) and
            (cpu or "nvidia" in reset["renderer_vendor"].lower()), "Wrong task conditions")
    if not cpu:
        require({k:v for k,v in reset.items() if k != "renderer"} == ref["gpu_reset"],
                "GPU query disagrees with historical native reset")
    for arm, rec in arms.items():
        chunks = rec["chunks"]
        n = rec["controls"]
        require(rec["reset"] == reset and rec["released_prompt"] == reset["prompt"] and
                type(n) is int and 0 < n <= reset["horizon"] and (n+4)//5 == len(chunks)
                and len(rec["native_rewards"]) == n and all(map(math.isfinite,rec["native_rewards"]))
                and rec["native_return"] == sum(rec["native_rewards"]) and type(rec["success"]) is bool
                and type(rec["truncated"]) is bool and type(rec["terminated"]) is bool
                and rec["truncated"] != rec["terminated"] and (not rec["success"] or rec["terminated"])
                and (not rec["truncated"] or n == reset["horizon"]), f"Native outcome {arm}/{row['index']}")
        require(chunks[0]["proposal"] == rec["opening_proposal_sha256"] == first["opening_proposal_sha256"]
                and chunks[0]["input"] == first["chunks"][0]["input"], "Initial motor proposal/input mismatch")
        for j, c in enumerate(chunks):
            yaw, scale, grip = c["correction"]
            require(c["start"] == 5*j+1 and c["steps"] == min(5,n-5*j) and c["noise"] == ref["noise_draws"][j]
                    and all(HASH.fullmatch(c[k]) for k in ("issued","proposal","canonical"))
                    and math.isfinite(yaw) and math.isfinite(scale) and scale > 0 and type(grip) is int
                    and grip in (-1,1) and c["changed_offsets"] == sorted(set(c["changed_offsets"]))
                    and all(type(x) is int and 0 <= x < c["steps"] for x in c["changed_offsets"])
                    and (not c["changed_offsets"] or c["issued"] != c["canonical"]), f"Chunk receipt {arm}/{j}")
            if j < 2:
                require(all(HASH.fullmatch(c[k]) for k in ("input","after")), "Missing opening trace")
            if arm.startswith("exact"):
                expected = row["truth"] if arm == "exact1" or j >= 2 else list(IDENTITY)
                require(c["correction"] == expected, f"Wrong oracle onset {arm}/{j}")
            elif j < 2:
                require(c["correction"] == list(IDENTITY), "Reader acted before control 11")
            elif j > 24:
                require(c["correction"] == chunks[24]["correction"], "Reader updated after chunk 24")
            # A sign flip can produce -0.0 from +0.0 without a numerical action change.
            if tuple(c["correction"]) == IDENTITY:
                require(not c["changed_offsets"] and c["issued"] == c["canonical"], "Identity intervention changed an action")
    require(arms["exact11"]["chunks"][:2] == arms["stream11"]["chunks"][:2], "Unmatched controls 1–10")
    if tuple(row["truth"]) == IDENTITY:
        require(first["chunks"] == arms["exact11"]["chunks"] and
                all(first[k] == arms["exact11"][k] for k in ("native_rewards","success","terminated","truncated")),
                "The two exact interventions diverged on the nominal identity control")


def panel(out):
    return parse(checked(out/"panel.json", (out/"panel.sha256").read_text().strip()))


def summary(out):
    data = panel(out)
    paths = [out/"cases"/f"{i:02d}.json" for i in range(96)]
    require(set((out/"cases").glob("*.json")) == set(paths), "Missing or unexpected native triplets")
    scores, timings = [], []
    for r, path in zip(data["rows"], paths, strict=True):
        payload = parse(path.read_bytes())
        require(payload["row"] == r and payload["panel_sha256"] == (out/"panel.sha256").read_text().strip(),
                "Wrong case identity or inputs")
        arms = payload["arms"]
        validate(r, arms, data["conditions"][r["shard"]])
        scores.append(dict(**r, outcomes={k:a["success"] for k,a in arms.items()}))
        stream = arms["stream11"]["chunks"]
        corr = next((c["start"] for c in stream if tuple(c["correction"]) != IDENTITY), None)
        action = next((c["start"]+c["changed_offsets"][0] for c in stream if c["changed_offsets"]), None)
        timings.append(dict(index=r["index"], first_nonidentity=corr, first_changed_issued=action,
                            control_11_estimate=stream[2]["correction"] if len(stream)>2 else None))
    result = {}
    for key, group in (("all",scores), ("ood",[s for s in scores if s["regime"] != "familiar"])):
        counts = {k:sum(r["outcomes"][k] for r in group) for k in ARMS}
        result[key] = dict(n=len(group), native_successes=counts,
                           contrasts=[differences(group,a,b) for a,b in CONTRASTS])
    result["historical"] = data["provenance"]["archived_counts"]
    result["historical_discordant_indices"] = {
        a:[r["index"] for r in scores if r["outcomes"][a] != r["archived"][a]["success"]]
        for a in ("exact1","stream11")}
    result.update(fresh_episodes=288, archived_episodes_reused=0, streaming_onsets=timings)
    atomic(out/"summary.json", encode(result))
    print(result["all"], flush=True)
