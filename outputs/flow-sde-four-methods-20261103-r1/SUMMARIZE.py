"""Allocated-CPU audit and matched five-attempt curves; no model/sim calls."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson(wins, n):
    z = 1.959963984540054
    mean = wins / n
    middle = (mean + z*z/(2*n)) / (1+z*z/n)
    radius = z*math.sqrt(mean*(1-mean)/n+z*z/(4*n*n)) / (1+z*z/n)
    return middle-radius, middle+radius


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--readout", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-hostname", required=True)
    a = p.parse_args()
    if (sys.platform != "linux" or socket.gethostname() != a.expected_hostname
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""):
        raise ValueError("Use identified allocated Linux CPU resources with CUDA hidden")
    import numpy as np
    import torch
    torch.set_num_threads(2)
    variants = [m+"-"+mode for m in ("ad", "dpt", "dit", "ic_iql") for mode in ("full", "episode")]
    variants += ["native_initial-full", "native_final-full"]
    assembly_path=Path("/mowenjie/fluffy-bassoon-2/outputs/flow-four-method-source-gpu-20261103-r2/preparation/ASSEMBLY.json")
    if sha(assembly_path)!="12433cf163c03a7874f6028e8bdc2a662f25fdc964e9fd972b7ac4d4f2b1760c":
        raise ValueError("Audited source identity changed")
    interface=json.loads(assembly_path.read_text())["codec"]
    outcomes, end_outcomes, audit = {}, {}, []
    prospective = None
    for variant in variants:
        directory = a.readout / variant
        binding = json.loads((directory / "BINDING.json").read_text())
        protocol=dict(version="frozen_steering_actual_attempt_v1",interface=interface,
            maximum_rgb_delta=None,actual_attempts=5,
            feedback="Observed first-success once; full physical boundary",
            noise="frozen_steering_actual_attempt_noise_v1")
        if variant.endswith("-episode"):protocol["memory_mode"]="episode"
        rows, refs, case_definitions = {}, [], {}
        for marker in sorted(directory.glob("complete-bank-*.json")):
            bank = json.loads(marker.read_text())
            if bank["binding"] != binding: raise ValueError("Mixed selected policy/source bindings")
            for item in bank["cases"]:
                case_definitions[item["case"]["task"], item["case"]["slot"]] = item
            refs += bank["attempt_references"]
            refs += [ref for prefix in bank.get("reused_prefix_references", {}).values() for ref in prefix]
            for row in bank["results"]:
                key = row["case"]["task"], row["case"]["slot"], row["attempt"]
                if key in rows: raise ValueError("A physical attempt was admitted twice")
                rows[key] = row
        if len(case_definitions) != 100 or len(rows) != 500 or len(refs) != 500:
            raise ValueError("Every condition requires exactly100 cases and500 actual attempts")
        if prospective is None: prospective = case_definitions
        if case_definitions != prospective: raise ValueError("Matched native case cohorts differ")
        seen = set()
        for ref in refs:
            if sha(ref["path"]) != ref["sha256"]: raise ValueError("Attempt artifact changed")
            saved = torch.load(ref["path"], map_location="cpu", weights_only=False)
            key = saved["case"]["task"], saved["case"]["slot"], saved["attempt"]
            if (key in seen or any(saved["binding"].get(k)!=v for k,v in binding.items())
                    or saved["binding"].get("native_attempt_protocol")!=protocol
                    or saved["interface"]!=interface
                    or saved["policy_weights_sha256"]!=binding["policy_weights_sha256"]
                    or saved["case"]!=rows[key]["case"]):
                raise ValueError("Duplicate/mixed physical provenance")
            seen.add(key)
            for name in ("truth", "video", "initial"):
                if sha(saved[name]["path"]) != saved[name]["sha256"]: raise ValueError("Native evidence changed")
            truth = json.loads(Path(saved["truth"]["path"]).read_text())
            video = json.loads(Path(saved["video"]["path"]).read_text())
            row = rows[key]; horizon = row["horizon"]
            if (not saved["completed"] or not row["completed"]
                    or saved["success_once"] != row["success_once"]
                    or row["success_once"] != truth["simulator_success_once"]
                    or saved["success_at_end"] != row["success_at_end"]
                    or row["success_at_end"] != truth["simulator_final_success"]
                    or truth["first_success_step"] != row["first_success_step"]
                    or truth["native_counts"]["controls"]-truth["native_counts"]["reset_controls"] != horizon
                    or not video["completed"] or video["frame_clock"]["frame_count"] != horizon+1
                    or video["frame_clock"]["resampled"] is not False
                    or len(saved["chunks"]) * 50 != horizon):
                raise ValueError("Only verified complete native simulator truth may become a curve row")
            for chunk in saved["chunks"]:
                if sha(chunk["path"]) != chunk["sha256"]: raise ValueError("Physical transition changed")
            for item in video["videos"]:
                if sha(item["path"]) != item["sha256"]: raise ValueError("Actual-attempt video changed")
        if seen != set(rows): raise ValueError("Actual artifact/outcome join differs")
        outcomes[variant] = {task: np.asarray([[rows[task, s, k]["success_once"] for k in range(1,6)]
            for s in range(50)], bool) for task in ("OpenDrawer", "CloseFridge")}
        end_outcomes[variant] = {task: np.asarray([[rows[task, s, k]["success_at_end"] for k in range(1,6)]
            for s in range(50)], bool) for task in ("OpenDrawer", "CloseFridge")}
        audit.append(dict(variant=variant, cases=100, physical_attempts=500, checked_chunk_and_video_hashes=True,
            selected_policy=binding["selected"], policy_weights_sha256=binding["policy_weights_sha256"]))
    a.output.mkdir(parents=True, exist_ok=False)
    curves = []
    for variant, groups in outcomes.items():
        for task, values in groups.items():
            for kind, data in (("per_attempt_first_success", values),
                               ("cumulative_ever_first_success", np.maximum.accumulate(values, axis=1)),
                               ("per_attempt_end_success", end_outcomes[variant][task]),
                               ("cumulative_ever_end_success", np.maximum.accumulate(end_outcomes[variant][task], axis=1))):
                for k in range(5):
                    wins = int(data[:,k].sum()); low, high = wilson(wins,50)
                    curves.append(dict(variant=variant, task=task, metric=kind, attempt=k+1,
                        successes=wins, cases=50, rate=wins/50, lower95=low, upper95=high))
    with (a.output / "curves.csv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(curves[0])); writer.writeheader();writer.writerows(curves)
    paired=[]; rng=np.random.default_rng(741230)
    for method in ("ad","dpt","dit","ic_iql"):
        for task in ("OpenDrawer","CloseFridge"):
            full=outcomes[method+"-full"][task]; reset=outcomes[method+"-episode"][task]
            for metric,left,right in (("attempt5",full[:,-1],reset[:,-1]),
                                      ("ever_through5",full.any(1),reset.any(1))):
                difference=left.astype(float)-right.astype(float)
                bootstrap=difference[rng.integers(0,50,size=(10000,50))].mean(1)
                wins=int((left & ~right).sum()); losses=int((~left & right).sum()); n=wins+losses
                exact=1. if not n else min(1.,2*sum(math.comb(n,i) for i in range(min(wins,losses)+1))/2**n)
                paired.append(dict(method=method,task=task,metric=metric,gain=float(difference.mean()),
                    paired_bootstrap95=list(map(float,np.quantile(bootstrap,[.025,.975]))),
                    discordant_wins=wins,discordant_losses=losses,exploratory_exact_p=exact))
    floor=0.
    for rank,index in enumerate(sorted(range(len(paired)),key=lambda i:paired[i]["exploratory_exact_p"])):
        floor=max(floor,min(1.,paired[index]["exploratory_exact_p"]*(len(paired)-rank)))
        paired[index]["Holm16_adjusted_p"]=floor
    (a.output / "paired.json").write_text(json.dumps(paired,indent=2)+"\n")
    (a.output / "AUDIT.json").write_text(json.dumps(dict(completed=True,variants=audit,
        actual_complete_episodes=5000,policy_controls=3250000,model_simulator_GPU_calls=0,
        limitation="Different direct-action interface and source horizon from older288k histories; weak source PPO gain. OOD is relative to source, not necessarily foundation pretraining."),indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors=dict(ad="#1565c0",dpt="#00897b",dit="#d84315",ic_iql="#8e24aa")
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    for row,task in enumerate(("OpenDrawer","CloseFridge")):
        for col,kind in enumerate(("per_attempt_first_success","cumulative_ever_first_success")):
            ax=axes[row,col]
            for variant in variants:
                data=[x for x in curves if x["variant"]==variant and x["task"]==task and x["metric"]==kind]
                method,mode=variant.rsplit("-",1)
                color=colors.get(method,"#263238" if method=="native_final" else "#9e9e9e")
                label=method.upper()+(" reset" if mode=="episode" else "")
                ax.plot(range(1,6),[x["rate"] for x in data],label=label,color=color,
                    linestyle="--" if mode=="episode" else "-",marker="o",markersize=3)
                if mode=="full":ax.fill_between(range(1,6),[x["lower95"] for x in data],[x["upper95"] for x in data],color=color,alpha=.07)
            ax.set_title(task+" · "+("Per-attempt success" if col==0 else "Ever succeeded"))
            ax.set_ylim(0,1);ax.set_xticks(range(1,6));ax.grid(alpha=.15)
            if row==1:ax.set_xlabel("Actual attempt")
            if col==0:ax.set_ylabel("First-success rate")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=5,frameon=False)
    fig.suptitle("Flow-SDE histories: frozen four-method readout · 50 paired cases/task")
    fig.tight_layout(rect=(0,.08,1,.96))
    fig.savefig(a.output/"curves.png",dpi=180);fig.savefig(a.output/"curves.svg")


if __name__=="__main__":main()
