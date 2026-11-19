"""Show the same prospective query at reset and at the visual controller's decision."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import pilot
from visual import public_rgb


class Frames:
    def start(self,sim):
        self.renderer=mujoco.Renderer(sim.model._model,height=224,width=224)
        self.reset=self.frame(sim)

    def frame(self,sim):
        return public_rgb(sim,self.renderer)

    def choose(self,sim):
        self.decision=self.frame(sim)
        return 'suffix'


def main(openings, output):
    previous=pilot.old.tape
    pilot.old.tape=lambda p,t,c: openings/f'pose{p}' if p>=12 else previous(p,t,c)
    fig,ax=plt.subplots(2,2,figsize=(6.4,6.2),layout='constrained')
    resets=[]
    for i,g in enumerate(pilot.REGIMES):
        camera=Frames()
        r=pilot.simulate((12,g,'suffix'),camera=camera)
        camera.renderer.close()
        resets.append(camera.reset)
        for j,im in enumerate([camera.reset,camera.decision]):
            ax[i,j].imshow(im)
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
        ax[i,0].set_ylabel('Stationary' if i==0 else 'Moving rack')
        ax[i,1].set_xlabel(f'After {r["lift"]+5} controls')
    assert np.array_equal(*resets)
    ax[0,0].set_title('At reset')
    ax[0,1].set_title('At the camera decision')
    fig.savefig(output,dpi=170)
    plt.close(fig)


if __name__=='__main__':
    main(*map(Path,sys.argv[1:]))
