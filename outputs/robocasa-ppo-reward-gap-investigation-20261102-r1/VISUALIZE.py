"""Render retained diagnostics and unchanged video frames on Linux CPUs."""
import hashlib,json,sys
from pathlib import Path
root=Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-ppo-reward-gap-investigation-20261102-r1')
report=json.loads((root/'REPORT.json').read_text());episodes=json.loads((root/'EPISODES.json').read_text())['episodes']
if sys.argv[1]=='plot':
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    training=report['training_batches'];gradients=report['label_only_gradient_diagnostics']
    y=np.asarray([v['success'] for v in episodes if v['arm']=='F'],dtype=bool)
    score=np.asarray([v['reward'] for v in episodes if v['arm']=='F'])
    fig,axes=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
    for arm,color in [('S','#2563eb'),('F','#ea580c')]:
        rows=[v for v in training if v['arm']==arm]
        axes[0].plot(range(1,7),[v['true_success_rate'] for v in rows],'-o',color=color,label=arm+' true success')
    rows=[v for v in training if v['arm']=='F']
    axes[0].plot(range(1,7),[v['reward']['mean'] for v in rows],'--',color='#ea580c',label='F mean reward')
    axes[0].set(xlabel='Completed PPO update',ylabel='Fraction / normalized reward',ylim=(0,1),title='On-policy TRAIN batches (64 each)');axes[0].legend(fontsize=8)
    axes[1].boxplot([score[~y],score[y]],showfliers=True)
    axes[1].set_xticks([1,2],['Terminal failure','Terminal success'])
    axes[1].set(ylabel='Official Flora normalized reward',ylim=(0,1),title='Flora TRAIN discrimination: AUC 0.925')
    axes[2].bar(range(1,7),[v['full_batch_actor_gradient_cosine'] for v in gradients],color='#7c3aed')
    axes[2].axhline(0,color='black',linewidth=.7);axes[2].set(xlabel='PPO update',ylabel='Gradient cosine',ylim=(-1,1),title='Same F records: Flora vs binary labels')
    fig.savefig(root/'reward-alignment.png',dpi=180);plt.close(fig)
    (root/'PLOT-RECEIPT.json').write_text(json.dumps(dict(completed=True,matplotlib=matplotlib.__version__,
        source='REPORT.json and EPISODES.json',model_computation=False,new_robot_actions=0),indent=2)+'\n')
elif sys.argv[1]=='frames':
    from PIL import Image,ImageDraw
    import imageio.v2 as imageio
    chosen=[('high-failure-1',report['highest_reward_failures'][0]),('high-failure-2',report['highest_reward_failures'][1]),
            ('low-success-1',report['lowest_reward_successes'][0])]
    refs=[]
    for name,episode in chosen:
        canvas=Image.new('RGB',(940,880),'white');draw=ImageDraw.Draw(canvas)
        draw.text((12,10),episode['task'],fill='black')
        draw.text((12,30),f"Simulator final: {episode['success']} | Flora reward: {episode['reward']:.5f} | update {episode['update']}",fill='black')
        draw.text((12,50),episode['id'],fill='black')
        for col,frame in enumerate([0,375,750]):draw.text((180+col*256,75),f'Frame {frame} / 750',fill='black')
        for row,video in enumerate(episode['videos']):
            path=Path(video['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==video['sha256']
            draw.text((8,110+row*256),video['camera'],fill='black')
            with imageio.get_reader(str(path),format='ffmpeg') as stream:
                for col,frame in enumerate([0,375,750]):
                    image=Image.fromarray(stream.get_data(frame));assert image.size==(256,256)
                    canvas.paste(image,(170+col*256,100+row*256))
        canvas.save(root/(name+'.png'))
        refs.append(dict(output=name+'.png',episode=episode,frames=[0,375,750],pixel_resizing=False))
    (root/'FRAME-RECEIPT.json').write_text(json.dumps(dict(completed=True,examples=refs,new_robot_actions=0),indent=2)+'\n')
else:raise ValueError('Select plot or frames')
