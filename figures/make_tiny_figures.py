"""Comparative scientific figures; every number comes from shipped run records."""
import figstyle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from tiny_comparison import SCENES,cells,comparison,spheres
HERE=Path(__file__).resolve().parent
RED='#65717B'; BLUE='#0072B2'; ORANGE='#B87A00'; GREY='#888888'
SHORT=['F','S','G','C1','C2','C3']


def save(fig,name):
    figstyle.save(fig,HERE/name)
    plt.close(fig)


def plot_all(data):
    figstyle.use()
    cc=cells(data);comp=comparison(data)
    fig,axes=plt.subplots(2,2,figsize=(figstyle.TEXT_W,4.9),layout='constrained');ax,bx,cx,dx=axes.flat
    for arm,label,color,marker in [('sdp','TinySDP formulation',RED,'s'),('sdp_layer_off','SDP layer off',ORANGE,'^'),('kkt_one_margin1.5','One-step QP, +1.5 m',BLUE,'o')]:
        a=[cc[s,arm] for s in SCENES]
        ax.scatter([r['plan_ms']['p50'] for r in a],[r['min_dense'] for r in a],s=30,c=color,marker=marker,label=label,zorder=3)
    # Annotate the gate comparison only; all six paired values are tabulated.
    a=cc['vertical_gate','sdp'];b=cc['vertical_gate','kkt_one_margin1.5']
    ax.annotate('',xy=(b['plan_ms']['p50'],b['min_dense']),xytext=(a['plan_ms']['p50'],a['min_dense']),
                arrowprops={'arrowstyle':'->','color':BLUE,'lw':1.2,'connectionstyle':'arc3,rad=-.15'})
    ratio=a['plan_ms']['p50']/b['plan_ms']['p50']
    ax.text(.72,1.09,f'Gate: {ratio:.0f}× faster\n{a["min_dense"]:.3f} → {b["min_dense"]:.3f} m',color=BLUE,fontsize=8)
    ax.set_xscale('log');ax.set_xlim(.035,110);ax.set_ylim(0,1.75)
    ax.set_xlabel('Median planning time (ms)');ax.set_ylabel('Minimum sampled clearance (m)')
    ax.set_title('(a) Planning time and clearance',fontsize=9)
    ax.legend(loc='lower left',frameon=False,fontsize=7,handletextpad=.4)
    ratios=[r['speedup'] for r in comp['revised']];yy=np.arange(6)
    bx.barh(yy,ratios,color=BLUE,height=.61)
    for y,v in zip(yy,ratios):bx.text(v+8,y,f'{v:.0f}×',va='center',fontsize=8,color=BLUE)
    bx.set_yticks(yy,list(SCENES.values()));bx.invert_yaxis();bx.set_xlim(0,720)
    bx.set_xlabel('Planning-time ratio (TinySDP / QP)');bx.set_title('(b) Ratio of scene medians',fontsize=9)
    bx.grid(axis='x',alpha=.15);bx.set_axisbelow(True)
    colors=['#0072B2','#D55E00','#009E73','#CC79A7','#56B4E9','#555555']
    for j,scene in enumerate(SCENES):
        base=cc[scene,'sdp'];mm=[cc[scene,f'kkt_one_margin{v:g}'] for v in [0.,1.,1.5]]
        marker=['o','s','^','D','v','P'][j]
        cx.plot([0,1,1.5],[q['min_dense']-base['min_dense'] for q in mm],marker=marker,ms=3,color=colors[j],label=SCENES[scene])
        dx.plot([0,1,1.5],[q['control_effort']/base['control_effort'] for q in mm],marker=marker,ms=3,color=colors[j])
    cx.axhline(0,color='black',ls='--',lw=.7);dx.axhline(1,color='black',ls='--',lw=.7)
    for a in [cx,dx]:a.set_xticks([0,1,1.5]);a.set_xlabel('QP additional inflation (m)');a.grid(alpha=.12)
    cx.set_ylabel('Clearance change (m)');dx.set_ylabel('Effort / TinySDP effort')
    cx.set_title('(c) Margin-dependent clearance');dx.set_title('(d) Margin-dependent effort')
    cx.legend(ncol=3,loc='lower left',fontsize=9,columnspacing=.5,handlelength=1)
    save(fig,'fig_revision_tiny_overview')

    fig,(ax,bx)=plt.subplots(1,2,figsize=(figstyle.TEXT_W,2.7),layout='constrained')
    chosen=[('sdp','TinySDP',RED),('kkt_one_margin0','QP +0 m',GREY),('kkt_one_margin1.5','QP +1.5 m',BLUE)]
    for arm,label,col in chosen:
        rep=next(r for r in data['revision_tinysdp']['rows'] if r['scene']=='vertical_gate' and r['arm']==arm)['repeats'][0]
        steps=rep['steps'];track=rep['tracking'];vals=[s['dense_clear'] for s in steps];dash='--' if arm=='kkt_one_margin0' else '-';mark='s' if arm=='sdp' else '^' if arm=='kkt_one_margin0' else 'o'
        ax.plot([s['step'] for s in steps],vals,marker=mark,ls=dash,ms=3,lw=1.4,color=col,label=label)
        j=int(np.argmin(vals));ax.scatter([steps[j]['step']],[vals[j]],s=55,facecolors='none',edgecolors=col,zorder=4)
        # Interpolate the modeled constant-acceleration trajectory within each interval.
        xyz=[]
        for k in range(len(track)-1):
            p=np.array([track[k][n] for n in ['x','y','z']]);v=np.array([track[k][n] for n in ['vx','vy','vz']])
            u=np.array([track[k+1][n] for n in ['u1','u2','u3']]);t=np.linspace(0,1,17)
            xyz.extend(p+t[:,None]*v+.5*t[:,None]**2*u)
        xyz=np.array(xyz);bx.plot(xyz[:,0],xyz[:,2],color=col,ls=dash,lw=1.4,label=label,zorder=3)
        bx.scatter([v['x'] for v in track],[v['z'] for v in track],s=6,color=col,zorder=4)
    ax.axhline(0,color=RED,lw=.8,ls='--');ax.set_xlabel('Control interval k');ax.set_ylabel('Sampled minimum clearance (m)')
    ax.set_title('(a) Gate clearance (repeat 0)',fontsize=9);ax.legend(frameon=False,fontsize=7,loc='upper right')
    ax.set_ylim(-.05,2.15)
    for k,alpha in [(2,.10),(4,.18),(6,.28),(8,.4)]:
        c,r=spheres('vertical_gate',np.array(float(k)))[-1]
        bx.add_patch(Circle((c[0],c[2]),r,facecolor=ORANGE,edgecolor=ORANGE,alpha=alpha))
    for c,r in spheres('vertical_gate',np.array(0.))[:2]:bx.add_patch(Circle((c[0],c[2]),r,fill=False,edgecolor=GREY,ls='--',lw=.8))
    bx.scatter([-4.5],[1],marker='s',s=22,color='black',zorder=5);bx.scatter([0],[0],marker='*',s=65,color='black',zorder=5)
    bx.annotate('moving ball\nat k = 2, 4, 6, 8',xy=(-1.35,.65),xytext=(-4.5,-.95),fontsize=7,
                arrowprops={'arrowstyle':'->','color':ORANGE,'lw':.7})
    bx.set_xlim(-5,1.5);bx.set_ylim(-1.2,3.55);bx.set_aspect('equal',adjustable='box')
    bx.set_xlabel('x (m)');bx.set_ylabel('z (m)');bx.set_title('(b) Executed paths in the x-z plane',fontsize=9)
    save(fig,'fig_revision_tiny_gate')

    fig,(ax,bx)=plt.subplots(1,2,figsize=(figstyle.TEXT_W,2.7),layout='constrained')
    colors=['#0072B2','#D55E00','#009E73','#CC79A7','#56B4E9','#555555']
    for j,s in enumerate(SCENES):
        base=cc[s,'sdp'];m=[cc[s,f'kkt_one_margin{v:g}'] for v in [0.,1.,1.5]]
        ax.plot([0,1,1.5],[r['min_dense']-base['min_dense'] for r in m],marker=['o','s','^','D','v','P'][j],ms=3,color=colors[j],label=SCENES[s])
        bx.plot([0,1,1.5],[r['control_effort']/base['control_effort'] for r in m],marker=['o','s','^','D','v','P'][j],ms=3,color=colors[j])
    ax.axhline(0,color='black',lw=.7,ls='--');bx.axhline(1,color='black',lw=.7,ls='--')
    for a in [ax,bx]:a.set_xticks([0,1,1.5]);a.set_xlabel('QP additional inflation (m)');a.grid(alpha=.12)
    ax.set_ylabel('Clearance change from TinySDP (m)');bx.set_ylabel('Executed effort / TinySDP effort')
    ax.set_title('(a) Clearance versus inflation',fontsize=9);bx.set_title('(b) Executed control effort',fontsize=9)
    fig.set_size_inches(figstyle.TEXT_W,2.95);fig.legend(*ax.get_legend_handles_labels(),ncol=6,fontsize=9,frameon=False,loc='outside lower center',columnspacing=.9,handlelength=1.1);bx.set_ylim(0,4)
    save(fig,'fig_revision_tiny_margin')

    hist=data['a83_tinysdp_ladder'];fig,(ax,bx)=plt.subplots(1,2,figsize=(figstyle.TEXT_W,2.65),layout='constrained')
    order=['base','tinympc','cone','kkt','commit'];labels=['SDP','Layer\noff','ADMM\nQP','KKT iter.\n+1 m','KKT one\n+1.5 m']
    for j,s in enumerate(hist['scenes']):
        rr=[hist['table'][s][a] for a in order]
        ax.plot(range(5),[r['ms_per_solve'] for r in rr],marker=['o','s','^','D','v','P'][j],ms=3,color=colors[j],label=SCENES[s])
        bx.plot(range(5),[r['min_seg'] for r in rr],marker=['o','s','^','D','v','P'][j],ms=3,color=colors[j])
    for a in [ax,bx]:a.set_xticks(range(5),labels,fontsize=7);a.grid(alpha=.12)
    ax.set_yscale('log');ax.set_ylabel('Archived solve time per call (ms)');bx.set_ylabel('Archived segment clearance (m)')
    ax.set_title('(a) Archived configurations',fontsize=9);bx.set_title('(b) Archived clearance',fontsize=9)
    ax.legend(frameon=False,fontsize=7);bx.axhline(0,color=RED,lw=.7,ls='--');bx.set_ylim(0,.95)
    save(fig,'fig_revision_tiny_history')
