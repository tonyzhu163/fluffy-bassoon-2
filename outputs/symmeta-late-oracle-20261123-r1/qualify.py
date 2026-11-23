"""Linux CPU qualification with a real simulator, frozen reader and deterministic surrogate motor."""
import copy
import os
from types import SimpleNamespace

import jax
import numpy as np

from paired_inputs import module
from paired_records import atomic, digest, encode, require
from panel import ARMS, CHECKPOINT, differences
from records import panel, validate
from rollout import ev, run_arm
from driver import RETRY


class CpuMotor:
    _model = SimpleNamespace(action_horizon=10, action_dim=32)

    def __init__(self):
        self.calls = []

    def infer(self, item, *, noise):
        state = np.asarray(item["observation/state"], dtype=np.float32)
        actions = np.tanh(noise[:, :7]*0.35 + state[:7]*0.4).astype(np.float32)
        self.calls.append(dict(input=digest(encode(dict(prompt=item["prompt"],
                      arrays={k:ev.array_digest(v) for k,v in item.items() if k!="prompt"}))),
                      noise=ev.array_digest(noise), proposal=ev.array_digest(actions)))
        return {"actions": actions}


class CpuSim:
    def __init__(self, sim, cutoff):
        self.sim, self.cutoff = sim, cutoff
        self.commands = []
        self.controls = 0

    def call(self, op, value):
        result = self.sim.call(op,value)
        if op == "create":
            require(self.controls == 0, "Post-action restart")
            self.anchor = result[1]["anchor_eef_pos"]
            self.prompt = result[1]["language_instruction"]
        else:
            n = result["steps"]
            self.commands.append(ev.array_digest(value[:n]))
            self.controls += n
            if self.controls == 10:
                self.after10 = result["observation"]
            if self.controls == self.cutoff and not result["terminated"]:
                result = dict(result, truncated=True)
        return result


def qualify(out):
    require(os.environ["CUDA_VISIBLE_DEVICES"] == "" and os.environ["LIBGL_ALWAYS_SOFTWARE"] == "1"
            and all(x.platform == "cpu" for x in jax.devices()), "Require CPU + software rendering")
    data = panel(out)
    weights = ev.load_v2_checkpoint(CHECKPOINT)[:2]
    setup = module("retry_qa_live", RETRY/"live.py")
    evidence = []
    first_bad = []
    with setup.simulator(ev,cpu=True) as sim:
        for shard in range(8):
            r = data["rows"][shard*12+4+shard%7]
            ref = data["conditions"][shard]
            horizon = ref["reset"]["horizon"]
            # One full native horizon tests the production terminal checks and the update freeze after 120.
            cutoff = horizon if shard == 4 else 20
            setup.warmup(sim,r,ref["reset"],None,cpu=True)
            resets, results, traces = {}, {}, {}
            for name in ARMS:
                tap, motor = CpuSim(sim,cutoff), CpuMotor()
                rec = run_arm(tap,motor,r,name,ref,resets,weights)
                results[name] = rec
                traces[name] = (tap.commands,motor.calls)
                require(rec["controls"] >= 15 and rec["controls"] <= cutoff and
                        tap.commands == [c["issued"] for c in rec["chunks"]], "Issued trace")
                if name == "exact11":
                    item = ev.to_openpi_observation(ev.retarget_observation_for_motor(
                          tap.after10,tap.anchor,r["truth"][0]),tap.prompt)
                    noise_seed = ref["reset"]["noise_seed"]
                    bank = np.random.default_rng(noise_seed).standard_normal(((horizon+4)//5,10,32)).astype(np.float32)
                    independent = CpuMotor()
                    proposal = independent.infer(item,noise=bank[2])["actions"][:5]
                    expected = ev.retarget_action_from_motor(np.clip(proposal,-1,1),*r["truth"],
                                                            np.full(7,-1.),np.full(7,1.))
                    require(independent.calls[0] == motor.calls[2] and ev.array_digest(expected) == tap.commands[2],
                            "Control 11 observation/action convention")
            for name, original in (("exact1",ev.run_episode), ("stream11",ev.run_online_streaming_episode)):
                tap, motor = CpuSim(sim,cutoff), CpuMotor()
                kw = (dict(reader_params=weights[0], reader_cfg=weights[1]) if name == "stream11" else
                      dict(zip(("yaw_hat","scale_hat","grip_hat"),r["truth"],strict=True)))
                rec = ev.run_paired_query(original,tap,motor,r["query"],SimpleNamespace(name=r["embodiment"]),
                        arm_name=name,resets=resets,key=(r["suite"],r["task_id"],13),
                        support_physics=r["support_physics"], expected_prompt=ref["reset"]["prompt"],create_attempts=3,**kw)
                require(traces[name] == (tap.commands,motor.calls) and rec["success"] == results[name]["success"]
                        and rec["controls"] == results[name]["controls"], f"Original rollout divergence {shard}/{name}")
                if name == "stream11":
                    yaw,scale,grip=results[name]["chunks"][-1]["correction"]
                    require([round(yaw,2),round(scale,3),grip] ==
                            [rec[k] for k in ("final_yaw_hat","final_scale_hat","final_grip_hat")], "Final reader")
            stream = results["stream11"]["chunks"]
            require(results["exact11"]["chunks"][:2] == stream[:2] and
                    all(c["correction"] == [0.,1.,1] for c in stream[:2]), "Matched opening")
            require(all(results[a]["reset"] == results["exact1"]["reset"] for a in ARMS) and
                    all(results["exact1"]["reset"][k] == v for k,v in ref["reset"].items()), "Historical reset conditions")
            if cutoff == horizon:
                validate(r,results,ref,cpu=True)
                def reject(label, change):
                    bad = copy.deepcopy(results)
                    change(bad)
                    try:
                        validate(r,bad,ref,cpu=True)
                    except ValueError:
                        first_bad.append(label)
                    else:
                        raise AssertionError(f"Accepted corruption {label}")
                reject("noise",lambda a:a["exact1"]["chunks"][0].update(noise="0"*64))
                reject("issued prefix",lambda a:a["exact11"]["chunks"][1].update(after="0"*64))
                reject("early oracle",lambda a:a["exact11"]["chunks"][1].update(correction=r["truth"]))
                reject("native return",lambda a:a["stream11"].update(native_return=1e6))
            evidence.append(dict(index=r["index"],suite=r["suite"],task=r["task_id"],case=r["embodiment"],
                        compared_controls=results["stream11"]["controls"],native_horizon=horizon,
                        stream_first_nonidentity=next((c["start"] for c in stream if c["correction"] != [0.,1.,1]),None),
                        stream_first_changed_action=next((c["start"]+c["changed_offsets"][0]
                            for c in stream if c["changed_offsets"]),None),
                        stream_control11=stream[2]["correction"],renderer=results["exact1"]["reset"]["renderer"],
                        query_noise_sha256=ref["reset"]["noise_sha256"],
                        exact1_original_trace_sha256=digest(encode(traces["exact1"])),
                        stream_original_trace_sha256=digest(encode(traces["stream11"])),
                        matched_first_ten_sha256=digest(encode(stream[:2]))))
            print(f"CPU task {shard} passed through control {results['stream11']['controls']}", flush=True)

    # A known four-positive, four-zero paired task contrast exercises the interval and sign conventions.
    fixture = [dict(**r,outcomes=dict(a=r["shard"]%2 == 0,b=False)) for r in data["rows"]]
    ci = differences(fixture,"a","b")
    require(ci["difference_pp"] == 50 and ci["task_ci95_pp"] == [12.5,87.5] and
            differences(fixture,"b","a")["task_ci95_pp"] == [-87.5,-12.5], "Task interval/sign")
    atomic(out/"cpu.json", encode(dict(backend=jax.default_backend(),host=os.uname().nodename,
           scored_experiment_episodes=0, motor="state/noise-dependent scripted CPU surrogate",
           reader_sha256=data["provenance"]["reader_sha256"], cases=evidence,
           corruption_rejected=first_bad, known_contrast=ci, exact_original_trace_agreement=True,
           nominally_identical_delayed_prefix=True)))
