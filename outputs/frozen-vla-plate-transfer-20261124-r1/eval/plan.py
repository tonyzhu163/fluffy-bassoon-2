"""Draw the preregistered placement seeds and freeze unscored physical donor states on Baige CPU."""
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path

from large_panel_protocol import GEOMETRY_FIELDS, bind, mobile_joints, read, require, sha

PFS = Path("/mowenjie/fluffy-bassoon-2/outputs")
PRIOR = PFS / "libero-paligemma-dualarm-20261120-r2"
DOMAIN = "fluffy-bassoon/frozen-vla-plate-transfer-20261124-r1/v1"


def seed(domain, index, draw=0):
    digest = hashlib.sha256(f"{DOMAIN}/{domain}/{index}/{draw}".encode()).digest()
    number = int.from_bytes(digest[:8], "big")
    # Place test seeds in a separate native uint32 namespace from earlier source and test runs.
    return 2**30 + number % 2**30 if domain == "layout" else number


def prior():
    from large_panel_sim import released

    denied = {k: set() for k in (*GEOMETRY_FIELDS, "layout_seed")}
    sources = {}

    def add(rows):
        for row in rows:
            if "sampled_scene" in row:
                denied["mobile_joint_xy_hash"].add(mobile_joints(row["sampled_scene"]))
            for k, values in denied.items():
                if k in row:
                    values.add(row[k])

    public = released()
    add(public)
    require(len(public) == 50, "incomplete released denylist")
    paths = set(PRIOR.glob("large-panel*/*/manifest.json"))
    paths.update(PRIOR.glob("gendemos*/banks/bank*/index.json"))
    paths.update(PRIOR.glob("layouts*/layouts/index.json"))
    paths.update(PRIOR.glob("scale3fit01/data/bank*/index.json"))
    paths.update(PFS.glob("embodiment-source-correction-gate-*/native*/layouts/manifest.json"))
    paths.update(PFS.glob("embodiment-probe-order-*/native*/collection/collection.json"))
    paths.update(PFS.glob("embodiment-command-frame-*/native*/collection/tapes.jsonl*"))
    for path in sorted(paths):
        sources[str(path.relative_to(PFS))] = sha(path)
        if path.name.startswith("tapes.jsonl"):
            with path.open() as stream:
                add(json.loads(line) for line in stream if line.strip())
        else:
            doc = read(path)
            add(doc.get("layouts", []))
            if path.parent.parent.name == "layouts01":
                add(doc["accepted"])
    require(any("scale3fit01/data/bank12" in p for p in sources) and any("large-panel-v2" in p for p in sources)
            and len(denied["scene_hash"]) >= 345, "prior placement archive unavailable")
    return denied, sources


def prepare(out, count, prefix=None):
    import generated_layouts as g
    require(os.uname().sysname == "Linux" and len(os.sched_getaffinity(0)) <= 2 and
            os.environ.get("CUDA_VISIBLE_DEVICES") == "", "use bounded Linux CPU")
    out.mkdir()
    attempts, cases = [], []
    parent_sha = None
    if prefix is None:
        denied, provenance = prior()
        inventory = dict(files=provenance, fingerprints={k: sorted(v) for k, v in denied.items()})
    else:
        previous = read(prefix)
        parent_sha = sha(prefix)
        inventory = read(prefix.with_name("prior.json"))
        require(sha(prefix.with_name("prior.json")) == previous["prior_sha256"] and previous["domain"] == DOMAIN,
                "invalid prior panel")
        denied = {k: set(v) for k,v in inventory["fingerprints"].items()}
        cases = list(previous["cases"])
        attempts = list(previous["rejected"])
        require(len(cases) >= 256 and count == 2*len(cases), "extension must double the frozen prefix")
        for r in attempts:
            require(r["seed"] not in denied["layout_seed"], "previous candidate repeated")
            denied["layout_seed"].add(r["seed"])
        for i, row in enumerate(cases):
            require(row["index"] == i and row["layout_seed"] == seed("layout", i, row["draw"]) and
                    row["noise_seed"] == seed("noise", i), "original enrollment changed")
            for k in (*GEOMETRY_FIELDS, "layout_seed"):
                require(row[k] not in denied[k], "previous physical placement repeated")
                denied[k].add(row[k])
    bind(out / "prior.json", inventory)
    with closing(g.make_sampler()) as sampler:
        public = g.released_fingerprints()
        for index in range(len(cases), count):
            noise = seed("noise", index)
            draw = 0
            while True:
                candidate = seed("layout", index, draw)
                require(candidate not in denied["layout_seed"], f"repeated layout seed {candidate}")
                denied["layout_seed"].add(candidate)
                try:
                    state, origin = g.sample_layout(sampler, candidate, public)
                except g.LayoutRejected as exc:
                    require("duplicate" not in str(exc).lower(), "placement overlap with released bank")
                    attempts.append(dict(index=index, draw=draw, seed=candidate, reason=str(exc)))
                    draw += 1
                    continue
                break
            origin["mobile_joint_xy_hash"] = mobile_joints(origin["sampled_scene"])
            for k in GEOMETRY_FIELDS:
                require(origin[k] not in denied[k], f"placement overlap at {index}: {k}")
                denied[k].add(origin[k])
            # The legacy sampler records a source-use label; the native physical constructor is unchanged.
            origin.pop("source_index")
            origin.update(split="target", index=index, draw=draw, noise_seed=noise, shard=index % 4,
                          donor_state=state.tolist())
            cases.append(origin)
            if len(cases) % 32 == 0:
                print("physical placements", len(cases), flush=True)
    require(len({r["noise_seed"] for r in cases}) == count, "noise seed collision")
    bind(out / "manifest.json", dict(domain=DOMAIN, prior_sha256=sha(out / "prior.json"),
                                     prepare_pin=os.environ["EVAL_PIN"], rejected=attempts, cases=cases,
                                     **({"prefix_sha256": parent_sha} if parent_sha else {})))
    print("manifest", sha(out / "manifest.json"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--count", type=int, default=256)
    p.add_argument("--prefix", type=Path, help="immutable parent manifest for a registered doubling look")
    args = p.parse_args()
    prepare(args.out, args.count, args.prefix)
