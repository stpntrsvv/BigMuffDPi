"""Точный переход линейного скелета быстрого ядра при постоянных портовых токах."""
from __future__ import annotations
import numpy as np

from generate_port_multirate_fixture import FAST_ACTIVE
from muff_frontend_model import prepare_frontend, nonlinear_terms
from muff_multirate_model import FAST_NODE_COUNT, Q2_COLLECTOR


def _integral_exponential(a: np.ndarray, step: float) -> tuple[np.ndarray,np.ndarray]:
    value,vector=np.linalg.eig(a)
    inverse=np.linalg.inv(vector)
    transition=vector@np.diag(np.exp(value*step))@inverse
    transition=np.real_if_close(transition, tol=1000).real
    integral=np.linalg.solve(a,transition-np.eye(len(a)))
    return transition,integral


def exact_fast_affine(parameters,dc_q,port_g,rate,sustain=1.0,nonlinear_rule="trapezoid"):
    base=prepare_frontend(sustain,None,parameters)
    g=base.linear_matrix[:FAST_NODE_COUNT,:FAST_NODE_COUNT].copy()
    g[Q2_COLLECTOR,Q2_COLLECTOR]-=1.0/39_000.0
    g[Q2_COLLECTOR,Q2_COLLECTOR]+=port_g
    d=base.capacitor_incidence[:FAST_NODE_COUNT,:9].copy()
    capacitance=base.capacitance[:9]
    injection=base.injection[:FAST_NODE_COUNT].copy()
    voltage=base.voltage[:,:FAST_NODE_COUNT].copy()
    dc_current,dc_first=nonlinear_terms(dc_q,parameters,"hybrid",dc_q)
    passive=np.ones(9,dtype=bool);passive[FAST_ACTIVE]=False
    slope=np.where(passive,dc_first,0.0)
    offset=np.where(passive,dc_current-slope*dc_q,0.0)
    g_effective=g+(injection*slope[np.newaxis,:])@voltage
    source=base.source[:FAST_NODE_COUNT]-injection@offset
    input_vector=base.input_vector[:FAST_NODE_COUNT]
    port_vector=np.zeros(FAST_NODE_COUNT);port_vector[Q2_COLLECTOR]=-1.0
    active_injection=injection[:,FAST_ACTIVE]
    kkt=np.block([[g_effective,d],[d.T,np.zeros((9,9))]])
    inverse=np.linalg.inv(kkt)
    v_rhs=inverse[:FAST_NODE_COUNT,:FAST_NODE_COUNT]
    v_state=inverse[:FAST_NODE_COUNT,FAST_NODE_COUNT:]
    lambda_rhs=inverse[FAST_NODE_COUNT:,:FAST_NODE_COUNT]
    lambda_state=inverse[FAST_NODE_COUNT:,FAST_NODE_COUNT:]
    inv_c=np.diag(1.0/capacitance)
    ac=inv_c@lambda_state
    drive=inv_c@lambda_rhs
    transition,integral=_integral_exponential(ac,1.0/rate)
    state_bias=integral@drive@source
    state_input=integral@drive@input_vector
    state_port=integral@drive@port_vector
    state_active=integral@drive@active_injection
    # Алгебраическое напряжение узлов на конце шага.
    node_bias=v_rhs@source+v_state@state_bias
    node_input=v_rhs@input_vector+v_state@state_input
    node_port=v_rhs@port_vector+v_state@state_port
    node_state=v_state@transition
    node_active=v_rhs@active_injection+v_state@state_active
    q_voltage=voltage[FAST_ACTIVE]
    q_direct=q_voltage@v_rhs@active_injection
    q_dynamic=q_voltage@v_state@state_active
    port_direct=v_rhs[Q2_COLLECTOR]@active_injection
    port_dynamic=v_state[Q2_COLLECTOR]@state_active
    if nonlinear_rule == "trapezoid":
        new_state_active=0.5*state_active; history_state_active=0.5*state_active
        new_q_influence=q_direct+0.5*q_dynamic; history_q_influence=0.5*q_dynamic
        new_port_active=port_direct+0.5*port_dynamic; history_port_active=0.5*port_dynamic
    else:
        new_state_active=state_active; history_state_active=np.zeros_like(state_active)
        new_q_influence=q_direct+q_dynamic; history_q_influence=np.zeros_like(q_dynamic)
        new_port_active=port_direct+port_dynamic; history_port_active=np.zeros_like(port_dynamic)
    return {
        "active_bias":q_voltage@node_bias,
        "active_input":q_voltage@node_input,
        "active_port":q_voltage@node_port,
        "active_state":q_voltage@node_state,
        "active_influence":new_q_influence,
        "active_history":history_q_influence,
        "state_bias":state_bias,"state_input":state_input,
        "state_port":state_port,"state_transition":transition,
        "state_active":new_state_active,"state_active_history":history_state_active,
        "port_bias":node_bias[Q2_COLLECTOR],"port_input":node_input[Q2_COLLECTOR],
        "port_port":node_port[Q2_COLLECTOR],"port_state":node_state[Q2_COLLECTOR],
        "port_active":new_port_active,"port_active_history":history_port_active,
    }
