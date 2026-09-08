"""MuJoCo rigid-body tracking of the exact reference, with four lagged rotors.

All tracking and executed-clearance claims are numerical simulation results.
The 12-cm tracking reserve is an allocated margin, not a proved error bound.
"""
from pathlib import Path
from fractions import Fraction
from math import comb
import hashlib,json,sys,time
import numpy as np
import mujoco

OUT=Path(__file__).resolve().parents[1]/'paper_mpc/demo3d'
MASS=.60;ARM=.14;MAX_ROTOR=3.30;MOTOR_TAU=.018
J=np.diag([.003,.003,.005]);GRAVITY=9.81


def evaluate(G,s,order=0):
    N,_,h=G.shape;P=G.copy()
    for k in range(order):P=np.diff(P,axis=2)*(h-1-k)*N
    i=min(int(max(0.,min(1.,s))*N),N-1);u=max(0.,min(1.,s))*N-i;d=P.shape[-1]-1
    return P[i]@np.array([comb(d,k)*u**k*(1-u)**(d-k) for k in range(d+1)])


def vee(A):return np.array([A[2,1],A[0,2],A[1,0]])


def desired_rotation(a,yaw):
    b3=a/np.linalg.norm(a);b1=np.array([np.cos(yaw),np.sin(yaw),0.])
    b2=np.cross(b3,b1);b2/=np.linalg.norm(b2)
    return np.column_stack([np.cross(b2,b3),b2,b3])


def model_xml(scene,dt):
    obs='\n'.join(f'<geom name="obstacle_{i}" type="sphere" pos="{c[0]} {c[1]} {c[2]}" size="{o["radius"]}"/>'
        for i,o in enumerate(scene['physical_obstacles']) for c in [o['center']])
    p=scene['samples']['recovered'][0]
    return f'''<mujoco model="generic inspection quadrotor">
  <compiler angle="radian"/>
  <option timestep="{dt}" gravity="0 0 -{GRAVITY}" integrator="RK4"/>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 .1"/>
    {obs}
    <body name="quad" pos="{p[0]} {p[1]} {p[2]}">
      <freejoint/>
      <inertial pos="0 0 0" mass="{MASS}" diaginertia=".003 .003 .005"/>
      <geom name="body_envelope" type="sphere" size="{scene['body_radius']}"/>
    </body>
  </worldbody>
</mujoco>'''


def simulate(scene,dt=.001,store=True):
    source=scene['recovered_Gamma'];G=np.array([float(Fraction(int(a),int(b))) for a,b in source['values']]).reshape(source['shape'])
    duration=scene['duration'];pre=1.;post=2.;total=pre+duration+post
    model=mujoco.MjModel.from_xml_string(model_xml(scene,dt));data=mujoco.MjData(model)
    bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'quad');mujoco.mj_forward(model,data)
    a=ARM/np.sqrt(2);xy=np.array([[a,a],[-a,a],[-a,-a],[a,-a]])
    mix=np.array([np.ones(4),xy[:,1],-xy[:,0],.014*np.array([1.,-1.,1.,-1.])])
    rotor=np.ones(4)*MASS*GRAVITY/4;target=rotor.copy();Rd_prev=np.eye(3)
    outer_stride=round(.005/dt);outer_dt=outer_stride*dt
    wrench=np.zeros(6);vel=np.zeros(6);body_vel=np.zeros(6);records=[]
    metrics=dict(max_error=0.,min_body_clearance=float('inf'),min_floor_clearance=float('inf'),
                 max_speed=0.,max_tilt_deg=0.,max_rotor_thrust=0.,saturated_updates=0,contact_steps=0)
    centers=np.array([o['center'] for o in scene['physical_obstacles']]);radii=np.array([o['radius'] for o in scene['physical_obstacles']])
    t0=time.perf_counter()
    for i in range(int(round(total/dt))+1):
        t=i*dt;s=np.clip((t-pre)/duration,0,1)
        ref=evaluate(G,s);vr=evaluate(G,s,1)/duration;ar=evaluate(G,s,2)/duration**2
        R=data.xmat[bid].reshape(3,3);pos=data.xpos[bid].copy()
        mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,bid,vel,0)
        mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,bid,body_vel,1)
        v=vel[3:].copy();omega=body_vel[:3].copy()
        if i%outer_stride==0:
            # Position feedback and gravity compensation; yaw is a smooth chosen heading.
            acc=ar+6.25*(ref-pos)+5.*(vr-v)+.08*v/MASS+np.array([0.,0.,GRAVITY])
            Rd=desired_rotation(acc,.35*np.sin(2*np.pi*s))
            wd=vee(Rd_prev.T@Rd-Rd.T@Rd_prev)/(2*outer_dt) if i else np.zeros(3)
            er=.5*vee(Rd.T@R-R.T@Rd);ew=omega-R.T@Rd@wd
            tau=-.35*er-.065*ew+np.cross(omega,J@omega)
            thrust=max(0.,MASS*np.dot(acc,R[:,2]));raw=np.linalg.solve(mix,np.r_[thrust,tau])
            metrics['saturated_updates']+=int(np.any(raw<0)|np.any(raw>MAX_ROTOR))
            target=np.clip(raw,0,MAX_ROTOR);Rd_prev=Rd
        rotor+=(1-np.exp(-dt/MOTOR_TAU))*(target-rotor)
        force_tau=mix@rotor
        wind=np.array([0.,.12*np.exp(-.5*((t-pre-duration*.52)/.55)**2),0.])
        data.xfrc_applied[bid,:3]=R[:,2]*force_tau[0]-.08*v+wind
        data.xfrc_applied[bid,3:]=R@force_tau[1:]
        error=float(np.linalg.norm(pos-ref));clear=float(np.min(np.linalg.norm(pos-centers,axis=1)-radii-scene['body_radius']))
        tilt=float(np.degrees(np.arccos(np.clip(R[2,2],-1,1))))
        metrics['max_error']=max(metrics['max_error'],error);metrics['min_body_clearance']=min(metrics['min_body_clearance'],clear)
        metrics['min_floor_clearance']=min(metrics['min_floor_clearance'],pos[2]-scene['body_radius'])
        metrics['max_speed']=max(metrics['max_speed'],float(np.linalg.norm(v)))
        metrics['max_tilt_deg']=max(metrics['max_tilt_deg'],tilt)
        metrics['max_rotor_thrust']=max(metrics['max_rotor_thrust'],float(np.max(rotor)))
        metrics['contact_steps']+=int(data.ncon>0)
        if store:records.append([t,*pos,*data.qpos[3:7],*ref,error,clear,float(np.linalg.norm(v)),tilt,*rotor,float(wind[1])])
        if i<int(round(total/dt)):mujoco.mj_step(model,data)
    metrics.update(final_position_error=error,final_speed=float(np.linalg.norm(v)),steps=i+1,dt=dt,
                   within_allocated_tracking_reserve=metrics['max_error']<scene['tracking_reserve'],
                   wall_seconds=time.perf_counter()-t0)
    metrics['passed']=bool(metrics['contact_steps']==0 and metrics['min_body_clearance']>0
                           and metrics['within_allocated_tracking_reserve'] and metrics['final_position_error']<.025)
    return dict(metrics=metrics,columns=['t','x','y','z','qw','qx','qy','qz','ref_x','ref_y','ref_z','error','body_clearance','speed','tilt_deg','f0','f1','f2','f3','wind_y_N'],samples=records,
                duration=duration,total=total,pre=pre,post=post,model_xml=model_xml(scene,dt))


def main():
    path=OUT/'planning.json';planning=json.loads(path.read_text());scene=planning['inspection']
    primary=simulate(scene);check=simulate(scene,dt=.0005,store=False)
    report=dict(mujoco_version=mujoco.__version__,mass=MASS,arm=ARM,max_rotor_thrust=MAX_ROTOR,
        motor_time_constant=MOTOR_TAU,controller_hz=200,physics_hz=1000,
        wind='0.12 N lateral Gaussian pulse, sigma 0.55 s, centered at 52% of flight',
        primary=primary['metrics'],half_step=check['metrics'],
        integration_sensitivity=dict(max_error_difference=abs(primary['metrics']['max_error']-check['metrics']['max_error']),
            min_clearance_difference=abs(primary['metrics']['min_body_clearance']-check['metrics']['min_body_clearance'])),
        planning_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Numerical 6-DOF rigid-body simulation with four lagged, bounded rotors. Sampled executed clearance and tracking error are not formal continuous-time bounds. Generic model, not calibrated hardware.')
    (OUT/'tracking.json').write_text(json.dumps(primary,separators=(',',':'))+'\n')
    (OUT/'tracking_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'quadrotor.xml').write_text(primary['model_xml'])
    print(json.dumps(report,indent=2))
    assert primary['metrics']['passed'] and check['metrics']['passed'],'Demo tracking failed; retain logs and investigate.'


if __name__=='__main__':main()
