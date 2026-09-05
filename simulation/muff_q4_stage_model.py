"""Полный локальный Q4 с C1/C10/C4 и Sustain, без обратной нагрузки через C5."""
from __future__ import annotations
import numpy as np
from muff_frontend_model import SUSTAIN_WIPER,nonlinear_terms,operating_point,prepare_frontend

NODE_COUNT=7
def simulate_q4(input_v,step_s,sustain,parameters):
 base=prepare_frontend(sustain,None,parameters);matrix=base.linear_matrix[:NODE_COUNT,:NODE_COUNT].copy()
 incidence=base.capacitor_incidence[:NODE_COUNT,:3].copy();conductance=base.capacitance[:3]/step_s
 matrix+=(incidence*conductance[np.newaxis,:])@incidence.T
 source=base.source[:NODE_COUNT];bu=base.input_vector[:NODE_COUNT]
 injection=base.injection[:NODE_COUNT,:3];voltage=base.voltage[:3,:NODE_COUNT]
 dc_nodes,dc_q=operating_point(sustain,parameters);nodes=dc_nodes[:NODE_COUNT].copy();q=dc_q[:3].copy()
 state=incidence.T@nodes;output=np.empty(len(input_v));output[0]=nodes[SUSTAIN_WIPER]
 maximum=0.0
 for k in range(1,len(input_v)):
  rhs=source+bu*input_v[k]+incidence@(conductance*state)
  linear_nodes=np.linalg.solve(matrix,rhs);linear_q=voltage@linear_nodes
  for _ in range(20):
   full=np.zeros(9);full[:3]=q;dcfull=dc_q[:9]
   current,first=nonlinear_terms(full,parameters,"reference_full",dcfull);current=current[:3];first=first[:3]
   residual=q-linear_q+(voltage@np.linalg.solve(matrix,injection))@current
   norm=float(np.max(np.abs(residual)));maximum=max(maximum,norm)
   if norm<1e-12:break
   jac=np.eye(3)+(voltage@np.linalg.solve(matrix,injection))*first[np.newaxis,:]
   q-=np.linalg.solve(jac,residual)
  full=np.zeros(9);full[:3]=q;current,_=nonlinear_terms(full,parameters,"reference_full",dc_q[:9]);current=current[:3]
  nodes=np.linalg.solve(matrix,rhs-injection@current);state=incidence.T@nodes;output[k]=nodes[SUSTAIN_WIPER]
 return output,maximum
