"""L-устойчивый SDIRK2 для связанного быстрого ядра."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from generate_port_multirate_fixture import FAST_ACTIVE,slow_affine
from muff_complete_model import Q1_FORWARD,Q1_REVERSE,operating_point
from muff_frontend_model import nonlinear_terms,prepare_frontend
from muff_multirate_model import FAST_NODE_COUNT,Q1_NONLINEAR,Q2_COLLECTOR,SLOW_NODES,_q1_terms,slow_step
from run_port_sparse_linear_experiment import active_terms

GAMMA=1.0-1.0/np.sqrt(2.0)

@dataclass
class ContinuousFast:
    ac:np.ndarray; bias:np.ndarray; input:np.ndarray; port:np.ndarray; active:np.ndarray
    q_bias:np.ndarray; q_state:np.ndarray; q_input:np.ndarray; q_port:np.ndarray; q_active:np.ndarray
    out_bias:float; out_state:np.ndarray; out_input:float; out_port:float; out_active:np.ndarray

def prepare_continuous(p,dc_q,port_g,sustain):
    base=prepare_frontend(sustain,None,p); n=FAST_NODE_COUNT
    g=base.linear_matrix[:n,:n].copy();g[Q2_COLLECTOR,Q2_COLLECTOR]-=1/39_000;g[Q2_COLLECTOR,Q2_COLLECTOR]+=port_g
    d=base.capacitor_incidence[:n,:9];c=base.capacitance[:9]
    inj=base.injection[:n];vmap=base.voltage[:,:n]
    dc_i,dc_g=nonlinear_terms(dc_q,p,"hybrid",dc_q)
    passive=np.ones(9,dtype=bool);passive[FAST_ACTIVE]=False
    slope=np.where(passive,dc_g,0);offset=np.where(passive,dc_i-slope*dc_q,0)
    ge=g+(inj*slope[np.newaxis,:])@vmap
    source=base.source[:n]-inj@offset;bu=base.input_vector[:n]
    bp=np.zeros(n);bp[Q2_COLLECTOR]=-1;bi=inj[:,FAST_ACTIVE]
    kkt=np.block([[ge,d],[d.T,np.zeros((9,9))]]);ki=np.linalg.inv(kkt)
    vr=ki[:n,:n];vx=ki[:n,n:];lr=ki[n:,:n];lx=ki[n:,n:]
    ci=np.diag(1/c);drive=ci@lr
    qv=vmap[FAST_ACTIVE]
    return ContinuousFast(
        ci@lx,drive@source,drive@bu,drive@bp,drive@bi,
        qv@vr@source,qv@vx,qv@vr@bu,qv@vr@bp,qv@vr@bi,
        float((vr@source)[Q2_COLLECTOR]),vx[Q2_COLLECTOR],float((vr@bu)[Q2_COLLECTOR]),
        float((vr@bp)[Q2_COLLECTOR]),vr[Q2_COLLECTOR]@bi)

def solve_stage(system,p,x_base,input_v,port_offset,h,q_guess,corrections=3,inverse=None):
    if inverse is None:
        inverse=np.linalg.inv(np.eye(9)-h*GAMMA*system.ac)
    x_linear=inverse@(x_base+h*GAMMA*(system.bias+system.input*input_v+system.port*port_offset))
    state_active=h*GAMMA*inverse@system.active
    linear_q=system.q_bias+system.q_state@x_linear+system.q_input*input_v+system.q_port*port_offset
    influence=system.q_active+system.q_state@state_active
    q=q_guess.copy()
    for _ in range(corrections):
        current,first=active_terms(q,p)
        residual=q-linear_q+influence@current
        jacobian=np.eye(4)+influence*first[np.newaxis,:]
        correction=np.linalg.solve(jacobian,residual)
        # Ограниченная обратная связь вместо безусловного полного шага Ньютона.
        scale=min(1.0,0.5/max(float(np.max(np.abs(correction)/np.array((.0258649,.0258649,.04914331,.04914331)))),0.5))
        q-=scale*correction
    current,_=active_terms(q,p);x=x_linear-state_active@current
    return x,q,current

def simulate_sdirk(input_v,sustain,tone,level_unused=0,fast_factor=2,corrections=3):
    p=__import__('q3_model').Q3Parameters();nodes,dc_q=operating_point(sustain,tone,0.8,p)
    sr,slow=slow_affine(p,tone,0.8,48_000*2)
    slow_state=sr.incidence.T@np.r_[nodes[Q2_COLLECTOR],nodes[SLOW_NODES]]
    _,_,_,_,pi,pg=slow_step(sr,slow_state,dc_q[Q1_NONLINEAR],float(nodes[Q2_COLLECTOR]),True)
    sys=prepare_continuous(p,dc_q[:9],pg,sustain);front=prepare_frontend(sustain,None,p)
    x=front.capacitor_incidence[:FAST_NODE_COUNT,:9].T@nodes[:FAST_NODE_COUNT]
    q=dc_q[FAST_ACTIVE].copy();slow_q=dc_q[[Q1_FORWARD,Q1_REVERSE]].copy()
    port_offset=pi-pg*nodes[Q2_COLLECTOR];out=np.zeros(len(input_v));h=1/(48_000*fast_factor)
    stage_inverse=np.linalg.inv(np.eye(9)-h*GAMMA*sys.ac)
    for sample in range(1,len(input_v)):
        delta=input_v[sample]-input_v[sample-1]
        for sub in range(fast_factor):
            u1=input_v[sample-1]+(sub+GAMMA)/fast_factor*delta
            y1,q1,i1=solve_stage(sys,p,x,u1,port_offset,h,q,corrections,stage_inverse)
            f1=sys.bias+sys.ac@y1+sys.input*u1+sys.port*port_offset-sys.active@i1
            base=x+h*(1-GAMMA)*f1
            u2=input_v[sample-1]+(sub+1)/fast_factor*delta
            x,q,current=solve_stage(sys,p,base,u2,port_offset,h,q1,corrections,stage_inverse)
            port_v=sys.out_bias+sys.out_state@x+sys.out_input*u2+sys.out_port*port_offset-sys.out_active@current
            linear_q=slow["active_bias"]+slow["active_port"]*port_v+slow["active_state"]@slow_state
            for _ in range(2):
                si,sg=_q1_terms(slow_q,p);res=slow_q-linear_q+slow["active_influence"]@si
                slow_q-=np.linalg.solve(np.eye(2)+slow["active_influence"]*sg[np.newaxis,:],res)
            si,_=_q1_terms(slow_q,p)
            next_slow=slow["state_bias"]+slow["state_port"]*port_v+slow["state_transition"]@slow_state-slow["state_active"]@si
            out[sample]=slow["output_bias"]+slow["output_port"]*port_v+slow["output_state"]@slow_state-slow["output_active"]@si
            port_i=slow["current_bias"]+slow["current_port"]*port_v+slow["current_state"]@slow_state+slow["current_active"]@si
            slow_q=linear_q-slow["active_influence"]@si;slow_state=next_slow;port_offset=port_i-pg*port_v
    return out
