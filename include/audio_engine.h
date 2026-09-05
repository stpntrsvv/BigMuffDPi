#ifndef AUDIO_ENGINE_H
#define AUDIO_ENGINE_H

#include <stddef.h>

typedef struct {
    float sustain;
    float tone;
    float volume;
} audio_engine_controls_t;

void audio_engine_init(void);
void audio_engine_set_controls(const audio_engine_controls_t *controls);
void audio_engine_process(const float *input, float *output, size_t sample_count);

#endif
