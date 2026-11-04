"""Render only independently audited final query curves on allocated CPUs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sys.platform=='linux'
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--methods',nargs='*',choices=('ad','dpt','dit','ic_iql'),default=[])
    p.add_argument('--tag',required=True)
    args=p.parse_args()
    assert args.tag.replace('-','').replace('_','').isalnum()
    assert len(set(args.methods))==len(args.methods)
    run=Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-eight-hour-babysit-20261104-r1')
    os.environ['MPLCONFIGDIR']=str(run/'plot-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
        'svg.fonttype':'none','axes.axisbelow':True})
    data,inputs={},[]
    for method in ('native_initial','native_final',*args.methods):
        for split in ('ID','OOD'):
            group=run/'gpu'/f'{method.replace("_","-") if method.startswith("native_") else method}-{split}-final-01'
            result_path,audit_path=group/'RESULT.json',group/'AUDIT-STATUS.json'
            if not result_path.exists() or not audit_path.exists():
                assert method not in ('native_initial','native_final'), 'Both native controls must finish auditing'
                continue
            result=json.loads(result_path.read_text());audit=json.loads(audit_path.read_text())
            assert result['completed'] and result['cases_per_task']==50 and result['stage']=='final'
            assert result['method']==method and result['split']==split
            assert result['deployment_optimizer_updates']==0
            assert audit['passed'] and audit['group_result_sha256']==sha(result_path)
            summary=Path(audit['audit_summary']['path'])
            assert sha(summary)==audit['audit_summary']['sha256']
            assert set(result['summaries'])==({'full'} if method.startswith('native_') else {'full','episode'})
            data[method,split]=result
            inputs.append(dict(result=dict(path=str(result_path),sha256=sha(result_path)),
                audit=dict(path=str(audit_path),sha256=sha(audit_path))))
        if not method.startswith('native_') and (method,'ID') in data and (method,'OOD') in data:
            assert data[method,'ID']['selected_candidate_sha256']==data[method,'OOD']['selected_candidate_sha256']
    tasks=[('ID','OpenDrawer','OpenDrawer'),('ID','CloseToasterOvenDoor','Toaster door'),
        ('ID','CoffeeSetupMug','Coffee mug'),('OOD','CloseFridge','Close fridge'),
        ('OOD','TurnOnMicrowave','Microwave')]
    methods=args.methods or ['controls']
    fig,axes=plt.subplots(len(methods),5,figsize=(16,3*len(methods)+1.4),squeeze=False,sharex=True,sharey=True)
    styles=[('native_initial','full','#89949e','Original SFT control',':','.'),
        ('native_final','full','#9a83ab','Fixed130 PPO control','--','.'),
        (None,'full','#147dba','Full history','-','o'),(None,'episode','#df812e','Episode reset','-','s')]
    for row,method in enumerate(methods):
        for col,(split,task,title) in enumerate(tasks):
            ax=axes[row,col]
            for fixed,mode,color,label,style,marker in styles:
                key=(fixed or method,split)
                if key not in data:continue
                summary=data[key]['summaries'][mode][task]
                assert summary['cases']==50
                counts=summary['first_success_by_support_episodes']
                ax.plot(range(6),[2*counts[str(k)] for k in range(6)],color=color,linestyle=style,
                    marker=marker,linewidth=1.2 if fixed else 2.,markersize=4,label=label,
                    alpha=.75 if fixed else 1.)
            if method!='controls' and (method,split) not in data:
                ax.text(.5,.91,'Final evaluation pending',ha='center',transform=ax.transAxes,color='#606870',fontsize=8)
            if row==0:ax.set_title(f'{title}\n{split}',fontsize=10)
            if col==0:ax.set_ylabel(('Native controls' if method=='controls' else method.upper().replace('_','-'))+'\nQuery success (%)')
            ax.set(xlim=(-.2,5.2),ylim=(0,100),xticks=range(6),yticks=(0,25,50,75,100))
            ax.grid(axis='y',alpha=.18)
    legend_styles=styles[:2] if methods==['controls'] else styles
    handles=[Line2D([0],[0],color=s[2],linestyle=s[4],marker=s[5],label=s[3]) for s in legend_styles]
    fig.suptitle('RoboCasa: actual query success after 0–5 completed episodes',y=.98,fontsize=15)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.93),ncol=len(handles),frameon=False)
    fig.subplots_adjust(top=.79 if len(methods)==1 else .83,bottom=.20 if len(methods)==1 else .16,wspace=.12,hspace=.35)
    fig.supxlabel('Completed prior episodes (0 is the first query)',y=.095,fontsize=10)
    fig.text(.5,.025,'50 paired cases per task and mode; actual query success, not cumulative best-of-six. '
        'Native controls have no learned memory.\nCurrent three-source baseline with two contracted OOD tasks; '
        'single-fit descriptive curves; zero deployment SGD.',ha='center',fontsize=9,color='#505860')
    output=run/'figures'/args.tag;output.mkdir(parents=True,exist_ok=False)
    png,svg=output/'query-curves.png',output/'query-curves.svg'
    fig.savefig(png,dpi=160,facecolor='white');fig.savefig(svg,facecolor='white');plt.close(fig)
    source=Path(__file__).resolve().parents[2]
    manifest=dict(methods=methods,source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        inputs=inputs,images=[dict(path=str(path),sha256=sha(path)) for path in (png,svg)],CPU_only=True,GPU_hours=0)
    (output/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),methods=methods,inputs=len(inputs))))


if __name__=='__main__':
    main()
