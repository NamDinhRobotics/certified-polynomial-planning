import argparse
import gzip
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/polynomial-planning-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle
import numpy as np
from math import comb

ROOT = Path(__file__).resolve().parents[1]

def coefficients(value):
    return np.array([int(a) / int(b) for a, b in value['values']]).reshape(value['shape'])

def evaluate(G, times):
    times = np.asarray(times)
    segments = np.minimum((times * len(G)).astype(int), len(G) - 1)
    local = times * len(G) - segments
    d = G.shape[-1] - 1
    weights = np.array([comb(d, j) * local**j * (1-local)**(d-j) for j in range(d+1)]).T
    return np.einsum('tnd,td->tn', G[segments], weights)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(ROOT/'video/v3_execution.mp4'))
    args = parser.parse_args()
    records = json.loads(gzip.decompress((ROOT/'data/planning.json.gz').read_bytes()))['records']
    r = next(r for r in records if r['run_id'] == 'B_s13_r0_conic')
    with np.load(ROOT/'data'/r['tracking']['log'], allow_pickle=False) as z:
        data = z['samples']
        cols = {name:i for i,name in enumerate(z['columns'].tolist())}
    t = data[:,cols['t']]
    pos = data[:,[cols[n] for n in ('x','y','z')]]
    error = data[:,cols['error']]
    clearance = data[:,cols['body_clearance']]
    curve = evaluate(coefficients(r['recovery']['Gamma']), np.linspace(0,1,501))
    projected = evaluate(coefficients(r['projected_Gamma']), np.linspace(0,1,501))
    centers = np.array([o['center'] for o in r['input']['physical_obstacles']])
    radii = np.array([o['radius'] for o in r['input']['physical_obstacles']])
    assert np.allclose(np.min(np.linalg.norm(pos[:,None,:]-centers,axis=2)-radii-.23,axis=1),clearance)
    assert np.isclose(max(error),r['tracking']['max_position_error'])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':'#dae5ed','axes.labelcolor':'#bccdd9','xtick.color':'#98adbf','ytick.color':'#98adbf'})
    fig = plt.figure(figsize=(12.8,7.2),dpi=100,facecolor='#0b1520')
    ax = fig.add_axes([.035,.21,.57,.65],projection='3d',facecolor='#0b1520')
    top = fig.add_axes([.68,.53,.28,.30],facecolor='#101f2d')
    metric = fig.add_axes([.68,.22,.28,.21],facecolor='#101f2d')
    fig.text(.04,.935,'Constructive recovery to recorded flight',fontsize=23,weight='bold')
    fig.text(.04,.893,'MuJoCo telemetry replay  |  Scene 41013  |  Conic arm, repetition 0',fontsize=12,color='#9eb5c8')
    u,v=np.meshgrid(np.linspace(0,2*np.pi,24),np.linspace(0,np.pi,14))
    sphere=np.array([np.cos(u)*np.sin(v),np.sin(u)*np.sin(v),np.cos(v)])
    for o,(_,inflated) in zip(r['input']['physical_obstacles'],r['input']['obstacles']):
        center=np.array(o['center'])
        p=center[:,None,None]+o['radius']*sphere
        ax.plot_surface(*p,color='#8c9dad',alpha=.65,linewidth=0)
        p=center[:,None,None]+inflated*sphere
        ax.plot_wireframe(*p,color='#778895',alpha=.22,rstride=3,cstride=3,linewidth=.4)
        top.add_patch(Circle(center[:2],inflated,fill=False,edgecolor='#8797a3',linestyle=':',linewidth=1))
        top.add_patch(Circle(center[:2],o['radius'],color='#8c9dad',alpha=.7))
    ax.plot(*projected.T,'--',color='#f8a15d',lw=2,label='Projected SDP')
    ax.plot(*curve.T,color='#59d8ce',lw=2,label='Certified recovery')
    path,=ax.plot([],[],[],color='#fafafa',lw=2,label='Recorded execution')
    top.plot(projected[:,0],projected[:,1],'--',color='#f8a15d',lw=1.7)
    top.plot(curve[:,0],curve[:,1],color='#59d8ce',lw=1.7)
    trace,=top.plot([],[],color='white',lw=1.8)
    body2=Circle(pos[0,:2],.23,facecolor='#59d8ce',edgecolor='white',alpha=.8)
    top.add_patch(body2)
    ax.set(xlim=(-3.6,3.6),ylim=(-1.8,1.6),zlim=(0,3),xlabel='x (m)',ylabel='y (m)',zlabel='z (m)')
    ax.set_box_aspect((7.2,3.4,3));ax.view_init(elev=24,azim=-66)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.pane.fill=False
        axis.line.set_color('#567083')
    ax.grid(False)
    legend=ax.legend(loc='upper left',fontsize=9,frameon=False)
    top.set(xlim=(-3.6,3.6),ylim=(-1.5,1.2),xlabel='x (m)',ylabel='y (m)',title='Top view: physical balls + planning envelopes')
    top.set_aspect('equal',adjustable='box')
    top.title.set_color('#dae5ed');top.title.set_fontsize(10)
    metric.plot(t,error*100,color='#59d8ce',lw=1.5,label='Position error')
    metric.plot(t,clearance*100,color='#f8a15d',lw=1.5,label='Body clearance')
    metric.axhline(12,color='#91a5b5',ls=':',lw=1,label='Allocated reserve')
    cursor=metric.axvline(0,color='white',lw=.8)
    metric.set(xlim=(0,t[-1]),ylim=(0,140),xlabel='Simulation time (s)',ylabel='cm')
    metric.legend(fontsize=8,frameon=False,loc='upper right')
    metric.grid(alpha=.12)
    stats=fig.text(.045,.155,'',fontsize=13,weight='bold')
    fig.text(.045,.10,'Projected curve violates inflated planning clearance; the corrected reference was verified before execution.',fontsize=10,color='#becdda')
    fig.text(.045,.062,'Replay of recorded samples, not a new simulation. Tracking is sampled evidence, not a closed-loop safety proof.',fontsize=10,color='#8fa8bd')
    clock=fig.text(.84,.89,'',fontsize=12,color='#59d8ce')
    body=None
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    writer=FFMpegWriter(fps=20,codec='libx264',bitrate=2500,metadata={},extra_args=['-pix_fmt','yuv420p','-map_metadata','-1','-movflags','+faststart'])
    with writer.saving(fig,str(output),dpi=100):
        for frame in range(300):
            play=frame/20
            sim=float(np.clip(play-2,0,t[-1]))
            i=int(np.argmin(abs(t-sim)))
            path.set_data_3d(*pos[:i+1:10].T)
            trace.set_data(pos[:i+1:10,0],pos[:i+1:10,1]);body2.center=pos[i,:2]
            if body is not None:body.remove()
            body=ax.plot_surface(*(pos[i,:,None,None]+.23*sphere),color='#59d8ce',alpha=.8,linewidth=0)
            cursor.set_xdata([sim,sim]);clock.set_text(f't = {sim:05.2f} s')
            if play<2:stats.set_text('SDP  →  endpoint-line recovery  →  exact reference check  →  tracking')
            elif play<12:stats.set_text(f'Error {error[i]*100:.2f} cm   |   Body clearance {clearance[i]*100:.2f} cm   |   Body radius 0.23 m')
            else:stats.set_text(f'This run: maximum error {max(error)*100:.2f} cm   |   minimum body clearance {min(clearance)*100:.2f} cm   |   no contact')
            writer.grab_frame()
    plt.close(fig)
    print(output)

if __name__=='__main__':
    main()
