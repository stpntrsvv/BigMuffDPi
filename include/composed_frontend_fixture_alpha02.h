#ifndef COMPOSED_FRONTEND_FIXTURE_ALPHA02_H
#define COMPOSED_FRONTEND_FIXTURE_ALPHA02_H

#define COMPOSED_STREAM_COUNT 256U

static const float composed_a_q_bias[3] = {
    2.86304641F,
    5.38289976F,
    0.71182549F
};

static const float composed_a_q_input[3] = {
    0.328469068F,
    0.0507434644F,
    0.0422377996F
};

static const float composed_a_q_state[3][7] = {
    {-0.328469068F, -0.377682626F, 0.113951191F, 0.0857550651F, -0.0624528639F, 0.0F, 0.0327761844F},
    {-0.0507434644F, 0.0879271179F, -0.522818327F, -0.565457523F, -0.316464365F, 0.0F, 0.166085169F},
    {-0.0422377996F, 0.0731886998F, -0.435183078F, -0.470675051F, -0.422129542F, -1.0F, -0.275379092F}
};

static const float composed_a_influence[3][3] = {
    {2974.18677F, 335.935577F, 1647.79797F},
    {4293.7666F, 1852.27002F, 8349.80762F},
    {3574.04175F, -2760.16528F, 11150.1484F}
};

static const float composed_a_state_bias[7] = {
    -0.000910070841F,
    3.35521269F,
    0.00178410357F,
    0.0102086151F,
    -0.71182549F,
    0.0F,
    0.0572155081F
};

static const float composed_a_state_input[7] = {
    0.000213458188F,
    -0.255432397F,
    3.62214378e-05F,
    0.000272587902F,
    -0.0422377996F,
    0.0F,
    0.000104184997F
};

static const float composed_a_state_transition[7][7] = {
    {0.999786556F, 0.000120053221F, -3.62214378e-05F, -2.72587895e-05F, 1.98517664e-05F, 0.0F, -1.04185001e-05F},
    {0.255432397F, 0.504238904F, 0.133539736F, 0.100496612F, -0.0731886998F, 0.0F, 0.0384105071F},
    {-3.62214378e-05F, 6.27636764e-05F, 0.999626815F, -0.000280851847F, 0.000204536045F, 0.0F, -0.000107343534F},
    {-0.000272587902F, 0.000472334068F, -0.00280851824F, 0.996962428F, 0.00221217284F, 0.0F, -0.0011609809F},
    {0.0422377996F, -0.0731886998F, 0.435183078F, 0.470675051F, 0.422129542F, 0.0F, 0.275379092F},
    {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F},
    {-0.000104184997F, 0.000180529387F, -0.0010734353F, -0.0011609809F, 0.0012942818F, 0.0F, 0.993158579F}
};

static const float composed_a_state_active[7][3] = {
    {-0.913612008F, -0.106783167F, -0.523782253F},
    {3305.96826F, 393.683899F, 1931.05945F},
    {3.06495428F, -1.10020471F, -5.39661551F},
    {23.0656071F, -11.899334F, -58.3674431F},
    {-3574.04175F, 2760.16528F, -11137.7471F},
    {0.0F, 0.0F, -12.4007931F},
    {8.81583595F, 54.6599121F, -34.1491928F}
};

static const float composed_a_q_boundary[3] = {
    0.0327761844F,
    0.166085169F,
    -0.275379092F
};

static const float composed_a_state_boundary[7] = {
    -1.04185001e-05F,
    0.0384105071F,
    -0.000107343534F,
    -0.0011609809F,
    0.275379092F,
    0.0F,
    -0.00684144674F
};

static const float composed_a_drive_bias[1] = {
    4.6138587F
};

static const float composed_a_drive_input[1] = {
    0.00840147864F
};

static const float composed_a_drive_boundary[1] = {
    0.448305726F
};

static const float composed_a_drive_state[7] = {
    -0.00840147864F,
    0.0145578897F,
    -0.0865618289F,
    -0.0936214998F,
    0.104370885F,
    0.0F,
    -0.551694274F
};

static const float composed_a_drive_active[3] = {
    710.909058F,
    4407.77539F,
    -2753.79102F
};

static const float composed_b_q_bias[1] = {
    -1.89808416F
};

static const float composed_b_q_input[1] = {
    2.19670844F
};

static const float composed_b_q_state[1][2] = {
    {-0.853286743F, -1.0F}
};

static const float composed_b_influence[1][1] = {
    {22526.0879F}
};

static const float composed_b_state_bias[2] = {
    1.89808416F,
    0.0F
};

static const float composed_b_state_input[2] = {
    -2.19670844F,
    0.0F
};

static const float composed_b_state_transition[2][2] = {
    {0.853286743F, 0.0F},
    {0.0F, 1.0F}
};

static const float composed_b_state_active[2][1] = {
    {-22513.6875F},
    {-12.4007931F}
};

static const float composed_b_q_port[1] = {
    546.601807F
};

static const float composed_b_state_port[2] = {
    -546.601807F,
    0.0F
};

static const float composed_b_base_bias[1] = {
    0.682578444F
};

static const float composed_b_base_input[1] = {
    0.0991029888F
};

static const float composed_b_base_port[1] = {
    -179.659576F
};

static const float composed_b_base_state[2] = {
    -0.0307515841F,
    0.0F
};

static const float composed_b_base_active[1] = {
    811.3703F
};

static const float composed_b_collector_bias[1] = {
    2.58066273F
};

static const float composed_b_collector_input[1] = {
    -2.09760547F
};

static const float composed_b_collector_port[1] = {
    -726.261414F
};

static const float composed_b_collector_state[2] = {
    0.822535157F,
    0.0F
};

static const float composed_b_collector_active[1] = {
    -21702.3164F
};

static const float composed_slow_active_bias[2] = {
    0.0560859218F,
    -7.70762873F
};

static const float composed_slow_active_port[2] = {
    0.826148212F,
    0.826148212F
};

static const float composed_slow_active_state[2][4] = {
    {-0.825368702F, 0.0245140214F, -0.966971636F, 0.0F},
    {-0.825368702F, 0.0245140214F, -0.966971636F, -0.137365043F}
};

static const float composed_slow_active_influence[2][2] = {
    {3306.68237F, -2303.38989F},
    {-12900.5732F, 13386.1348F}
};

static const float composed_slow_state_bias[4] = {
    -0.0535564236F,
    0.000636265089F,
    -0.00250978931F,
    0.0102206143F
};

static const float composed_slow_state_port[4] = {
    0.172571003F,
    0.0402772948F,
    0.00126274268F,
    0.0F
};

static const float composed_slow_state_transition[4][4] = {
    {0.826648653F, -0.0245424584F, -0.0315387808F, 0.0F},
    {-0.00981698371F, 0.957964599F, 0.000374689436F, 0.0F},
    {-0.00126155128F, 3.74689444e-05F, 0.998521984F, 0.0F},
    {0.0F, 0.0F, 0.0F, 0.998864353F}
};

static const float composed_slow_state_active[4][2] = {
    {-6.38106441F, -426.467804F},
    {0.0758088082F, 5.0665555F},
    {-0.299032807F, -19.9853592F},
    {16.9918766F, -17.0343571F}
};

static const float composed_slow_output_bias[1] = {
    6.10511351F
};

static const float composed_slow_output_port[1] = {
    0.0F
};

static const float composed_slow_output_state[4] = {
    0.0F,
    0.0F,
    0.0F,
    -0.678345919F
};

static const float composed_slow_output_active[2] = {
    10149.8145F,
    -10175.1885F
};

static const float composed_slow_current_bias[1] = {
    -1.72914733e-05F
};

static const float composed_slow_current_port[1] = {
    8.02727736e-05F
};

static const float composed_slow_current_state[4] = {
    -5.56645027e-05F,
    -3.24796092e-05F,
    -1.01827573e-05F,
    0.0F
};

static const float composed_slow_current_active[2] = {
    0.00206021988F,
    0.137691364F
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
    -0.000332298223F,
    8.07746983e-05F
};

#endif
