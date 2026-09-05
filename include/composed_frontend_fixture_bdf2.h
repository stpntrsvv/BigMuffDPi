#ifndef COMPOSED_FRONTEND_FIXTURE_BDF2_H
#define COMPOSED_FRONTEND_FIXTURE_BDF2_H

#define COMPOSED_STREAM_COUNT 256U

static const float composed_a_q_bias[3] = {
    2.72479916F,
    5.44289589F,
    0.77205801F
};

static const float composed_a_q_input[3] = {
    0.339657336F,
    0.0496088304F,
    0.0420029908F
};

static const float composed_a_q_state[3][7] = {
    {-0.339657336F, -0.356014848F, 0.105316848F, 0.078294538F, -0.0554536059F, 0.0F, 0.0292668603F},
    {-0.0496088304F, 0.0854177028F, -0.536747992F, -0.580981553F, -0.296777844F, 0.0F, 0.156631038F},
    {-0.0420029908F, 0.07232178F, -0.454455793F, -0.491907597F, -0.395523846F, -1.0F, -0.287805319F}
};

static const float composed_a_influence[3][3] = {
    {2857.05688F, 300.819977F, 1638.69995F},
    {4453.30273F, 1759.93506F, 8770.03125F},
    {3770.53882F, -2888.87891F, 11701.9463F}
};

static const float composed_a_state_bias[7] = {
    -0.000970024557F,
    3.55364299F,
    0.00200471655F,
    0.0114163999F,
    -0.77205801F,
    0.0F,
    0.0639840737F
};

static const float composed_a_state_input[7] = {
    0.000235081039F,
    -0.269661218F,
    3.74926458e-05F,
    0.000278727442F,
    -0.0420029908F,
    0.0F,
    0.000104189603F
};

static const float composed_a_state_transition[7][7] = {
    {0.999764919F, 0.000126740779F, -3.74926458e-05F, -2.78727439e-05F, 1.97414047e-05F, 0.0F, -1.04189603e-05F},
    {0.269661218F, 0.476535887F, 0.137352675F, 0.102110587F, -0.07232178F, 0.0F, 0.0381694101F},
    {-3.74926458e-05F, 6.45557593e-05F, 0.999594331F, -0.000301572087F, 0.00021359422F, 0.0F, -0.000112729045F},
    {-0.000278727442F, 0.000479919749F, -0.00301572098F, 0.996735752F, 0.00231196568F, 0.0F, -0.00122019078F},
    {0.0420029908F, -0.07232178F, 0.454455793F, 0.491907597F, 0.395523846F, 0.0F, 0.287805319F},
    {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F},
    {-0.000104189603F, 0.000179396215F, -0.00112729042F, -0.00122019078F, 0.00135268492F, 0.0F, 0.99238956F}
};

static const float composed_a_state_active[7][3] = {
    {-0.981508315F, -0.107091479F, -0.583374858F},
    {3526.37939F, 392.324982F, 2137.16846F},
    {3.3656528F, -1.15868759F, -6.31188583F},
    {25.0209007F, -12.5417538F, -68.3205032F},
    {-3770.53882F, 2888.87891F, -11688.0576F},
    {0.0F, 0.0F, -13.8888884F},
    {9.35292816F, 61.6275864F, -39.9729614F}
};

static const float composed_a_q_boundary[3] = {
    0.0292668603F,
    0.156631038F,
    -0.287805319F
};

static const float composed_a_state_boundary[7] = {
    -1.04189603e-05F,
    0.0381694101F,
    -0.000112729045F,
    -0.00122019078F,
    0.287805319F,
    0.0F,
    -0.00761046074F
};

static const float composed_a_drive_bias[1] = {
    4.60685349F
};

static const float composed_a_drive_input[1] = {
    0.00750165153F
};

static const float composed_a_drive_boundary[1] = {
    0.452046812F
};

static const float composed_a_drive_state[7] = {
    -0.00750165153F,
    0.0129165277F,
    -0.0811649114F,
    -0.0878537372F,
    0.097393319F,
    0.0F,
    -0.547953188F
};

static const float composed_a_drive_active[3] = {
    673.410828F,
    4437.18652F,
    -2878.05322F
};

static const float composed_b_q_bias[1] = {
    -2.09124875F
};

static const float composed_b_q_input[1] = {
    2.41956449F
};

static const float composed_b_q_state[1][2] = {
    {-0.839159548F, -1.0F}
};

static const float composed_b_influence[1][1] = {
    {24811.748F}
};

static const float composed_b_state_bias[2] = {
    2.09124875F,
    0.0F
};

static const float composed_b_state_input[2] = {
    -2.41956449F,
    0.0F
};

static const float composed_b_state_transition[2][2] = {
    {0.839159548F, 0.0F},
    {0.0F, 1.0F}
};

static const float composed_b_state_active[2][1] = {
    {-24797.8594F},
    {-13.8888884F}
};

static const float composed_b_q_port[1] = {
    602.21521F
};

static const float composed_b_state_port[2] = {
    -602.21521F,
    0.0F
};

static const float composed_b_base_bias[1] = {
    0.676337361F
};

static const float composed_b_base_input[1] = {
    0.10653244F
};

static const float composed_b_base_port[1] = {
    -177.858368F
};

static const float composed_b_base_state[2] = {
    -0.0300318506F,
    0.0F
};

static const float composed_b_base_active[1] = {
    887.466003F
};

static const float composed_b_collector_bias[1] = {
    2.76758623F
};

static const float composed_b_collector_input[1] = {
    -2.31303191F
};

static const float composed_b_collector_port[1] = {
    -780.073547F
};

static const float composed_b_collector_state[2] = {
    0.809127688F,
    0.0F
};

static const float composed_b_collector_active[1] = {
    -23910.3926F
};

static const float composed_slow_active_bias[2] = {
    0.0615738593F,
    -7.70230913F
};

static const float composed_slow_active_port[2] = {
    0.809389949F,
    0.809389949F
};

static const float composed_slow_active_state[2][4] = {
    {-0.808437109F, 0.0267556515F, -0.963739812F, 0.0F},
    {-0.808437109F, 0.0267556515F, -0.963739812F, -0.137346327F}
};

static const float composed_slow_active_influence[2][2] = {
    {3307.33643F, -2259.6897F},
    {-12900.2002F, 13430.1152F}
};

static const float composed_slow_state_bias[4] = {
    -0.058752697F,
    0.000777780544F,
    -0.00280156941F,
    0.0114455279F
};

static const float composed_slow_state_port[4] = {
    0.189206839F,
    0.0446624532F,
    0.00138558354F,
    0.0F
};

static const float composed_slow_state_transition[4][4] = {
    {0.80983901F, -0.0267925188F, -0.0345988087F, 0.0F},
    {-0.0107170073F, 0.953188062F, 0.000458026305F, 0.0F},
    {-0.00138395245F, 4.58026334e-05F, 0.998350203F, 0.0F},
    {0.0F, 0.0F, 0.0F, 0.998728275F}
};

static const float composed_slow_state_active[4][2] = {
    {-7.00018263F, -467.845551F},
    {0.0926698893F, 6.19343758F},
    {-0.333797395F, -22.3087921F},
    {19.0283089F, -19.0758801F}
};

static const float composed_slow_output_bias[1] = {
    6.10428143F
};

static const float composed_slow_output_port[1] = {
    0.0F
};

static const float composed_slow_output_state[4] = {
    0.0F,
    0.0F,
    0.0F,
    -0.678253472F
};

static const float composed_slow_output_active[2] = {
    10148.4316F,
    -10173.8027F
};

static const float composed_slow_current_bias[1] = {
    -1.69407194e-05F
};

static const float composed_slow_current_port[1] = {
    7.8987403e-05F
};

static const float composed_slow_current_state[4] = {
    -5.44915674e-05F,
    -3.21569642e-05F,
    -9.97620191e-06F,
    0.0F
};

static const float composed_slow_current_active[2] = {
    0.00201842887F,
    0.13489832F
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
    -0.000326479611F,
    7.94690786e-05F
};

#endif
