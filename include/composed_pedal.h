#ifndef COMPOSED_PEDAL_H
#define COMPOSED_PEDAL_H

#include <stdint.h>

/* Fixed 48 kHz model generated for Sustain=1, Tone=1, Volume=0.8. */
void composed_pedal_init(void);
float composed_pedal_process_sample(float input);
uint32_t composed_pedal_fallback_count(void);
uint32_t composed_pedal_adaptive_count(void);
uint32_t composed_pedal_fault_stage(void);
uint32_t composed_pedal_hold_count(void);
float composed_pedal_last_port(void);
float composed_pedal_q2_linear_min(void);
float composed_pedal_q2_linear_max(void);

#endif
