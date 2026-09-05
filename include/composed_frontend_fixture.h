#ifndef COMPOSED_FRONTEND_FIXTURE_H
#define COMPOSED_FRONTEND_FIXTURE_H

#define COMPOSED_STREAM_COUNT 256U

static const float composed_a_q_bias[3] = {
    2.23978233F,
    5.66349268F,
    0.9932338F
};

static const float composed_a_q_input[3] = {
    0.378262907F,
    0.0443506688F,
    0.039432954F
};

static const float composed_a_q_state[3][7] = {
    {-0.378262907F, -0.281173855F, 0.0780163407F, 0.0552930683F, -0.0347132236F, 0.0F, 0.0187977236F},
    {-0.0443506688F, 0.0741762966F, -0.583027601F, -0.632842302F, -0.230503142F, 0.0F, 0.124820873F},
    {-0.039432954F, 0.0659514368F, -0.518379986F, -0.562671125F, -0.306270748F, -1.0F, -0.328995228F}
};

static const float composed_a_influence[3][3] = {
    {2427.83423F, 195.730606F, 1538.70667F},
    {5010.35693F, 1449.69275F, 10217.3379F},
    {4454.7959F, -3324.63794F, 13596.6641F}
};

static const float composed_a_state_bias[7] = {
    -0.0011958261F,
    4.25534868F,
    0.0030298403F,
    0.0169074051F,
    -0.9932338F,
    0.0F,
    0.0953114033F
};

static const float composed_a_state_input[7] = {
    0.000331947202F,
    -0.319403231F,
    4.16531439e-05F,
    0.00029521127F,
    -0.039432954F,
    0.0F,
    0.000100361583F
};

static const float composed_a_state_transition[7][7] = {
    {0.999668062F, 0.000150119522F, -4.16531439e-05F, -2.95211266e-05F, 1.85334884e-05F, 0.0F, -1.00361585e-05F},
    {0.319403231F, 0.37961641F, 0.148222759F, 0.105050959F, -0.0659514368F, 0.0F, 0.0357136801F},
    {-4.16531439e-05F, 6.96646966e-05F, 0.999452412F, -0.000388080516F, 0.000243638598F, 0.0F, -0.000131933906F},
    {-0.00029521127F, 0.000493739499F, -0.00388080534F, 0.995787621F, 0.00264455425F, 0.0F, -0.00143206527F},
    {0.039432954F, -0.0659514368F, 0.518379986F, 0.562671125F, 0.306270748F, 0.0F, 0.328995228F},
    {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F},
    {-0.000100361583F, 0.00016785429F, -0.00131933903F, -0.00143206527F, 0.00154627755F, 0.0F, 0.988853395F}
};

static const float composed_a_state_active[7][3] = {
    {-1.24283731F, -0.104501128F, -0.821519852F},
    {4321.62549F, 371.867371F, 2923.37915F},
    {4.70561361F, -1.37375689F, -10.7995834F},
    {33.3504295F, -14.9113274F, -117.223145F},
    {-4454.7959F, 3324.63794F, -13575.8311F},
    {0.0F, 0.0F, -20.833334F},
    {11.3379879F, 94.3740921F, -68.5406723F}
};

static const float composed_a_q_boundary[3] = {
    0.0187977236F,
    0.124820873F,
    -0.328995228F
};

static const float composed_a_state_boundary[7] = {
    -1.00361585e-05F,
    0.0357136801F,
    -0.000131933906F,
    -0.00143206527F,
    0.328995228F,
    0.0F,
    -0.0111466106F
};

static const float composed_a_drive_bias[1] = {
    4.57494736F
};

static const float composed_a_drive_input[1] = {
    0.00481735589F
};

static const float composed_a_drive_boundary[1] = {
    0.464962691F
};

static const float composed_a_drive_state[7] = {
    -0.00481735589F,
    0.00805700663F,
    -0.0633282736F,
    -0.0687391311F,
    0.0742213205F,
    0.0F,
    -0.535037279F
};

static const float composed_a_drive_active[3] = {
    544.22345F,
    4529.95654F,
    -3289.95215F
};

static const float composed_b_q_bias[1] = {
    -2.91969848F
};

static const float composed_b_q_input[1] = {
    3.37399125F
};

static const float composed_b_q_state[1][2] = {
    {-0.780138731F, -0.99999994F}
};

static const float composed_b_influence[1][1] = {
    {34601.4492F}
};

static const float composed_b_state_bias[2] = {
    2.91969848F,
    0.0F
};

static const float composed_b_state_input[2] = {
    -3.37399125F,
    0.0F
};

static const float composed_b_state_transition[2][2] = {
    {0.780138731F, 0.0F},
    {0.0F, 1.0F}
};

static const float composed_b_state_active[2][1] = {
    {-34580.6172F},
    {-20.8333321F}
};

static const float composed_b_q_port[1] = {
    840.705505F
};

static const float composed_b_state_port[2] = {
    -840.705505F,
    0.0F
};

static const float composed_b_base_bias[1] = {
    0.650580227F
};

static const float composed_b_base_input[1] = {
    0.137161478F
};

static const float composed_b_base_port[1] = {
    -170.425354F
};

static const float composed_b_base_state[2] = {
    -0.0270988345F,
    0.0F
};

static const float composed_b_base_active[1] = {
    1201.18945F
};

static const float composed_b_collector_bias[1] = {
    3.57027864F
};

static const float composed_b_collector_input[1] = {
    -3.23682976F
};

static const float composed_b_collector_port[1] = {
    -1011.13086F
};

static const float composed_b_collector_state[2] = {
    0.753039896F,
    0.0F
};

static const float composed_b_collector_active[1] = {
    -33379.4297F
};

static const float composed_slow_active_bias[2] = {
    0.0845932066F,
    -7.68007517F
};

static const float composed_slow_active_port[2] = {
    0.739730477F,
    0.739730477F
};

static const float composed_slow_active_state[2][4] = {
    {-0.737819135F, 0.0357804298F, -0.950183988F, 0.0F},
    {-0.737819135F, 0.0357804298F, -0.950183988F, -0.137259051F}
};

static const float composed_slow_active_influence[2][2] = {
    {3310.0791F, -2076.38745F},
    {-12898.7627F, 13614.7266F}
};

static const float composed_slow_state_bias[4] = {
    -0.0804308653F,
    0.00156019325F,
    -0.00414324412F,
    0.0171573814F
};

static const float composed_slow_state_port[4] = {
    0.258354127F,
    0.0641091838F,
    0.00189950166F,
    0.0F
};

static const float composed_slow_state_transition[4][4] = {
    {0.73973006F, -0.0358637944F, -0.0473648421F, 0.0F},
    {-0.0143455174F, 0.931575835F, 0.000918780454F, 0.0F},
    {-0.0018945937F, 9.18780424e-05F, 0.997560084F, 0.0F},
    {0.0F, 0.0F, 0.0F, 0.998093605F}
};

static const float composed_slow_state_active[4][2] = {
    {-9.58306217F, -640.468018F},
    {0.185891688F, 12.4237604F},
    {-0.493653357F, -32.9925003F},
    {28.5243263F, -28.5956364F}
};

static const float composed_slow_output_bias[1] = {
    6.10040236F
};

static const float composed_slow_output_port[1] = {
    0.0F
};

static const float composed_slow_output_state[4] = {
    0.0F,
    0.0F,
    0.0F,
    -0.677822471F
};

static const float composed_slow_output_active[2] = {
    10141.9824F,
    -10167.3379F
};

static const float composed_slow_current_bias[1] = {
    -1.54827303e-05F
};

static const float composed_slow_current_port[1] = {
    7.36011934e-05F
};

static const float composed_slow_current_state[4] = {
    -4.96039938e-05F,
    -3.07724076e-05F,
    -9.11760799e-06F,
    0.0F
};

static const float composed_slow_current_active[2] = {
    0.0018447144F,
    0.123288415F
};

static const float composed_a_dc_state[7] = {
    -0.627634466F,
    6.48222971F,
    7.10986423F,
    -0.700245321F,
    3.80926275F,
    -3.80926275F,
    3.81395531F
};

static const float composed_b_dc_state[2] = {
    3.761024F,
    -3.761024F
};

static const float composed_a_dc_q[3] = {
    0.6100685F,
    0.633938313F,
    0.0F
};

static const float composed_b_dc_q[1] = {
    0.0F
};

static const float composed_slow_dc_state[4] = {
    3.84760737F,
    3.37704015F,
    -1.02709329F,
    4.41319704F
};

static const float composed_slow_dc_q[2] = {
    0.624471009F,
    -2.77710652F
};

static const float composed_port[2] = {
    -0.000302120519F,
    7.40031974e-05F
};

#endif
