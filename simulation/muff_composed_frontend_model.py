"""Композиция полного Q4+Sustain+Q3 и локального Q2 без глобальной связи."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from muff_frontend_model import Q2_COLLECTOR,Q2_DRIVE,nonlinear_terms,prepare_frontend

@dataclass
class Block:
 matrix:np.ndarray;source:np.ndarray;input:np.ndarray;injection:np.ndarray;voltage:np.ndarray
 influence:np.ndarray;incidence:np.ndarray;conductance:np.ndarray;indices:slice;dc_q:np.ndarray

def prepare_blocks(p,sustain,step,port_g,dc_q):
 base=prepare_frontend(sustain,None,p)
 # Q4 + Sustain + Q3 + единственный C13, выход q2_drive.
 n=Q2_DRIVE+1;g=base.linear_matrix[:n,:n].copy()
 d=base.capacitor_incidence[:n,:7];gc=base.capacitance[:7]/step;g+=(d*gc[np.newaxis,:])@d.T
 inj=base.injection[:n,:6];v=base.voltage[:6,:n]
 first=Block(g,base.source[:n].copy(),base.input_vector[:n].copy(),inj,v,
             v@np.linalg.solve(g,inj),d,gc,slice(0,6),dc_q)
 # Q2 начинается непосредственно R12 от известного q2_drive.
 ids=np.arange(13,17);g2=base.linear_matrix[np.ix_(ids,ids)].copy()
 g2[Q2_COLLECTOR-13,Q2_COLLECTOR-13]-=1/39_000;g2[Q2_COLLECTOR-13,Q2_COLLECTOR-13]+=port_g
 d2=base.capacitor_incidence[np.ix_(ids,[7,8])];gc2=base.capacitance[[7,8]]/step;g2+=(d2*gc2[np.newaxis,:])@d2.T
 inj2=base.injection[np.ix_(ids,np.arange(6,9))];v2=base.voltage[np.ix_(np.arange(6,9),ids)]
 drive=np.zeros(4);drive[0]=1/10_000
 second=Block(g2,base.source[ids].copy(),drive,inj2,v2,
              v2@np.linalg.solve(g2,inj2),d2,gc2,slice(6,9),dc_q)
 return first,second

def terms(q,block,p):
 full=block.dc_q.copy();full[block.indices]=q
 current,first=nonlinear_terms(full,p,"hybrid",block.dc_q)
 return current[block.indices],first[block.indices]

def block_step(block,state,q,input_v,port_offset,p):
 rhs=block.source+block.input*input_v+block.incidence@(block.conductance*state)
 # На границе блоков остаётся точный R12. Первый блок видит
 # напряжение базы Q2 с предыдущего внутреннего отсчёта.
 if len(rhs)==Q2_DRIVE+1:rhs[Q2_DRIVE]+=port_offset/10_000
 # Q2 видит постоянный источник Нортона Tone.
 if len(rhs)==4:rhs[Q2_COLLECTOR-13]-=port_offset
 linear_nodes=np.linalg.solve(block.matrix,rhs);linear_q=block.voltage@linear_nodes
 for _ in range(30):
  current,first=terms(q,block,p);res=q-linear_q+block.influence@current
  norm=float(np.max(np.abs(res)))
  if norm<1e-12:break
  correction=np.linalg.solve(np.eye(len(q))+block.influence*first[np.newaxis,:],res)
  damping=1.0
  while damping>=1/4096:
   candidate=q-damping*correction;ci,_=terms(candidate,block,p)
   if np.max(np.abs(candidate-linear_q+block.influence@ci))<norm:q=candidate;break
   damping*=.5
  else:break
 current,_=terms(q,block,p);nodes=np.linalg.solve(block.matrix,rhs-block.injection@current)
 return block.incidence.T@nodes,q,nodes,norm

def simulate_composed(input_v,p,sustain,tone,step,port_g,port_offset,dc_nodes,dc_q):
 first,second=prepare_blocks(p,sustain,step,port_g,dc_q)
 s1=first.incidence.T@dc_nodes[:Q2_DRIVE+1];q1=dc_q[:6].copy()
 s2=second.incidence.T@dc_nodes[13:17];q2=dc_q[6:9].copy()
 q2_base=float(dc_nodes[13]);q2_base_previous=q2_base
 output=np.empty(len(input_v));maximum=0.0
 for k in range(1,len(input_v)):
  predicted_base=2.0*q2_base-q2_base_previous
  s1,q1,n1,r1=block_step(first,s1,q1,float(input_v[k]),predicted_base,p)
  s2,q2,n2,r2=block_step(second,s2,q2,float(n1[Q2_DRIVE]),port_offset,p)
  q2_base_previous,q2_base=q2_base,float(n2[0])
  output[k]=n2[Q2_COLLECTOR-13];maximum=max(maximum,r1,r2)
 output[0]=dc_nodes[Q2_COLLECTOR]
 return output,maximum
