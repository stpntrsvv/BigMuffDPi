#ifndef FULL_PEDAL_WORKING_H
#define FULL_PEDAL_WORKING_H

extern float working_active_bias[6];
extern float working_active_input[6];
extern float working_active_state[6][13];
extern float working_active_influence[6][6];
extern float working_state_bias[13];
extern float working_state_input[13];
extern float working_state_pole[13];
extern float working_state_active[13][6];
extern float working_output_state[13];
extern float working_output_active[6];
extern float working_output_bias;
extern float working_output_input;

#endif
