"""Draw offline fitting and native development success on their respective held-out scenes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def draw(data, output):
    xs=np.arange(5)
    epochs=[.5,1,2,4,8]
    figure,axes=plt.subplots(1,2,figsize=(10,3.6),layout="constrained")
    labels={"memoryless":"Memoryless LoRA","own":"DPT, own support","donor":"DPT, swapped support"}
    colors={"memoryless":"#08786b","own":"#244897","donor":"#945900"}
    for arm in ("memoryless","own"):
        records=data["curves"]["records"]
        seed_curves=np.array([[next(r[arm] for r in records if r["epoch"]==epoch and r["seed"]==seed)
                               for epoch in epochs] for seed in (991,1481,2029)])
        axes[0].fill_between(xs,seed_curves.min(0),seed_curves.max(0),alpha=.13,color=colors[arm],linewidth=0)
        axes[0].plot(xs,seed_curves.mean(0),marker="o",lw=2,color=colors[arm],label=labels[arm])
    axes[0].axhline(data["curves"]["records"][0]["frozen"],ls=":",lw=1.4,color="#53565c",label="Frozen π0.5")
    for arm in ("memoryless","own","donor"):
        counts=[data["selection"]["counts"][str(int(epoch*288))][arm] for epoch in epochs]
        axes[1].plot(xs,counts,marker="o",lw=2,color=colors[arm],label=labels[arm],ls="--" if arm=="donor" else "-")
    axes[0].set(title="Offline imitation on 48 scenes",ylabel="Flow-matching error")
    axes[1].set(title="Native retries on 24 development scenes",ylabel="Successful retries (out of 24)",ylim=(0,24),yticks=range(0,25,6))
    for ax in axes:
        ax.set(xticks=xs,xticklabels=["½","1","2","4","8"],xlabel="Training epochs")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",color="#dcdedc",lw=.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False,fontsize=8,loc="lower left")
    figure.savefig(output,dpi=180,facecolor="white")
    plt.close(figure)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("data",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    draw(json.loads(args.data.read_text()),args.output)
