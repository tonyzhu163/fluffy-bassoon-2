"""Show the two factual practice histories for the first strict development triple."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(tapes, scores, dest):
    result = json.loads(scores.read_text())["splits"]["development"]
    cyclic = result["panels"]["cyclic"]["triples"]
    group = next(g for g in result["groups"] if g["strict"] and [g["A"],g["B"],g["query"]] in cyclic)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    for row, case in enumerate(group["cases"]):
        for col, (pose, variant) in enumerate(zip((group["A"],group["B"]),case["variants"])):
            with np.load(tapes / f"p{pose}-g{col}-{variant}.npz") as z:
                ax=axes[row,col]
                ax.imshow(z["frames"][-1,0], interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel("Reached destination" if z["hits"][:,col].any() else "Missed destination",fontsize=10)
                for spine in ax.spines.values():spine.set_visible(False)
    axes[0,0].set_title(f"Cabinet attempt · pose {group['A']}",fontsize=12)
    axes[0,1].set_title(f"Plate attempt · pose {group['B']}",fontsize=12)
    axes[0,0].set_ylabel("Retry cabinet",fontsize=12,fontweight="bold")
    axes[1,0].set_ylabel("Retry plate",fontsize=12,fontweight="bold")
    fig.suptitle("Two zero rewards in each row",fontsize=16,fontweight="bold",y=.98)
    fig.text(.5,.025,f"Final agent-camera views at control 300. Both retries start at pose {group['query']}.",
             ha="center",fontsize=10)
    fig.subplots_adjust(left=.12,right=.95,top=.86,bottom=.14,hspace=.25,wspace=.13)
    fig.savefig(dest,dpi=160,facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main(*map(Path,sys.argv[1:]))
