#ifndef COMPOSED_FRONTEND_FIXTURE_BDF2_H
#define COMPOSED_FRONTEND_FIXTURE_BDF2_H

#define COMPOSED_STREAM_COUNT 256U

static const float composed_a_q_bias[3] = {
    2.97384286F,
    3.71544027F,
    -0.690550387F
};

static const float composed_a_q_input[3] = {
    0.344664335F,
    0.0206211917F,
    0.0174596272F
};

static const float composed_a_q_state[3][7] = {
    {-0.344664335F, -0.347393692F, 0.0511432551F, 0.0325451456F, -0.0230507236F, 0.0F, 0.0121655259F},
    {-0.0206211917F, 0.0355060734F, -0.223113164F, -0.399856985F, -0.425062805F, 0.0F, 0.224336237F},
    {-0.0174596272F, 0.0300624166F, -0.188906282F, -0.338552386F, -0.504140615F, -1.0F, -0.230480418F}
};

static const float composed_a_influence[3][3] = {
    {3306.52539F, 125.043594F, 681.167969F},
    {1851.13025F, 2455.84424F, 12560.957F},
    {1567.3219F, -2299.66382F, 14911.6621F}
};

static const float composed_a_state_bias[7] = {
    -0.00105868385F,
    3.87844229F,
    0.00104545814F,
    0.00171068357F,
    0.690550387F,
    0.0F,
    0.0603560358F
};

static const float composed_a_state_input[7] = {
    0.00023329856F,
    -0.263131201F,
    1.82069252e-05F,
    0.000115860261F,
    -0.0174596272F,
    0.0F,
    4.33091009e-05F
};

static const float composed_a_state_transition[7][7] = {
    {0.999766707F, 0.000123671663F, -1.82069252e-05F, -1.15860257e-05F, 8.20602509e-06F, 0.0F, -4.33091009e-06F},
    {0.263131201F, 0.487779468F, 0.0667002797F, 0.0424449034F, -0.0300624166F, 0.0F, 0.0158661008F},
    {-1.82069252e-05F, 3.13491328e-05F, 0.999803007F, -0.000125356237F, 8.87859569e-05F, 0.0F, -4.68587386e-05F},
    {-0.000115860261F, 0.000199491042F, -0.00125356228F, 0.997753382F, 0.00159119628F, 0.0F, -0.000839788816F},
    {0.0174596272F, -0.0300624166F, 0.188906282F, 0.338552386F, 0.504140615F, 0.0F, 0.230480418F},
    {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F},
    {-4.33091009e-05F, 7.45706711e-05F, -0.000468587386F, -0.000839788816F, 0.00108325796F, 0.0F, 0.992531717F}
};

static const float composed_a_state_active[7][3] = {
    {-1.14151847F, -0.0445153415F, -0.242494822F},
    {4112.56982F, 163.080017F, 888.369263F},
    {1.63440549F, -0.481638432F, -2.62369847F},
    {10.4005833F, -8.63178539F, -47.0211639F},
    {-1567.3219F, 2299.66382F, -14897.7725F},
    {0.0F, 0.0F, -13.8888884F},
    {3.88778615F, 63.0891533F, -32.0111694F}
};

static const float composed_a_q_boundary[3] = {
    0.0121655259F,
    0.224336237F,
    -0.230480418F
};

static const float composed_a_state_boundary[7] = {
    -4.33091009e-06F,
    0.0158661008F,
    -4.68587386e-05F,
    -0.000839788816F,
    0.230480418F,
    0.0F,
    -0.00746826502F
};

static const float composed_a_drive_bias[1] = {
    4.34563446F
};

static const float composed_a_drive_input[1] = {
    0.00311825518F
};

static const float composed_a_drive_boundary[1] = {
    0.462284923F
};

static const float composed_a_drive_state[7] = {
    -0.00311825518F,
    0.00536908861F,
    -0.0337382928F,
    -0.060464792F,
    0.0779945776F,
    0.0F,
    -0.537715077F
};

static const float composed_a_drive_active[3] = {
    279.920593F,
    4542.41895F,
    -2304.8042F
};

static const float composed_b_q_bias[1] = {
    -2.10344672F
};

static const float composed_b_q_input[1] = {
    2.42975903F
};

static const float composed_b_q_state[1][2] = {
    {-0.842725754F, -1.0F}
};

static const float composed_b_influence[1][1] = {
    {24917.1309F}
};

static const float composed_b_state_bias[2] = {
    2.10344672F,
    0.0F
};

static const float composed_b_state_input[2] = {
    -2.42975903F,
    0.0F
};

static const float composed_b_state_transition[2][2] = {
    {0.842725754F, 0.0F},
    {0.0F, 1.0F}
};

static const float composed_b_state_active[2][1] = {
    {-24903.2422F},
    {-13.8888884F}
};

static const float composed_b_q_port[1] = {
    605.653381F
};

static const float composed_b_state_port[2] = {
    -605.653381F,
    0.0F
};

static const float composed_b_base_bias[1] = {
    0.679939926F
};

static const float composed_b_base_input[1] = {
    0.103521593F
};

static const float composed_b_base_port[1] = {
    -178.873795F
};

static const float composed_b_base_state[2] = {
    -0.0289786179F,
    0.0F
};

static const float composed_b_base_active[1] = {
    856.342163F
};

static const float composed_b_collector_bias[1] = {
    2.78338671F
};

static const float composed_b_collector_input[1] = {
    -2.32623744F
};

static const float composed_b_collector_port[1] = {
    -784.527161F
};

static const float composed_b_collector_state[2] = {
    0.813747108F,
    0.0F
};

static const float composed_b_collector_active[1] = {
    -24046.9004F
};

static const float composed_slow_active_bias[2] = {
    0.410894632F,
    -7.39185047F
};

static const float composed_slow_active_port[2] = {
    0.305264741F,
    0.305264741F
};

static const float composed_slow_active_state[2][4] = {
    {-0.290775031F, 0.406871438F, -0.758028686F, 0.0F},
    {-0.290775031F, 0.406871438F, -0.758028686F, -0.133028328F}
};

static const float composed_slow_active_influence[2][2] = {
    {3348.95679F, 521.938843F},
    {-12923.1885F, 16276.5137F}
};

static const float composed_slow_state_bias[4] = {
    -0.0211319067F,
    0.0118276579F,
    -0.00220357184F,
    0.0110856937F
};

static const float composed_slow_state_port[4] = {
    0.165160939F,
    0.0422422886F,
    0.000522578543F,
    0.0F
};

static const float composed_slow_state_transition[4][4] = {
    {0.834088564F, -0.0210738331F, -0.0124443453F, 0.0F},
    {-0.0084295338F, 0.949462116F, 0.0069651762F, 0.0F},
    {-0.000497773814F, 0.000696517644F, 0.998702347F, 0.0F},
    {0.0F, 0.0F, 0.0F, 0.99876827F}
};

static const float composed_slow_state_active[4][2] = {
    {-2.51779437F, -168.272583F},
    {1.40922499F, 94.1832047F},
    {-0.262548059F, -17.5469608F},
    {18.4300823F, -18.4761562F}
};

static const float composed_slow_output_bias[1] = {
    3.80080938F
};

static const float composed_slow_output_port[1] = {
    0.0F
};

static const float composed_slow_output_state[4] = {
    0.0F,
    0.0F,
    0.0F,
    -0.42231217F
};

static const float composed_slow_output_active[2] = {
    6318.88525F,
    -6334.68262F
};

static const float composed_slow_current_bias[1] = {
    -6.38926213e-06F
};

static const float composed_slow_current_port[1] = {
    7.21242395e-05F
};

static const float composed_slow_current_state[4] = {
    -4.7566351e-05F,
    -3.04144469e-05F,
    -3.76256548e-06F,
    0.0F
};

static const float composed_slow_current_active[2] = {
    0.000761258765F,
    0.0508774593F
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
    3.76102352F,
    -3.76102352F
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
    3.84760332F,
    3.37703323F,
    0.49531576F,
    4.41319704F
};

static const float composed_slow_dc_q[2] = {
    0.624471009F,
    -2.77710652F
};

static const float composed_port[2] = {
    -0.000294048252F,
    7.21919278e-05F
};

#endif
