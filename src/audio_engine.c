#include "audio_engine.h"

#ifdef COMPOSED_PEDAL_RUNTIME
#include "composed_pedal.h"
#endif

static audio_engine_controls_t s_controls;

void audio_engine_init(void)
{
    s_controls.sustain = 0.60F;
    s_controls.tone = 0.45F;
    s_controls.volume = 0.50F;
#ifdef COMPOSED_PEDAL_RUNTIME
    composed_pedal_init();
#endif
}

void audio_engine_set_controls(const audio_engine_controls_t *controls)
{
    if (controls != NULL) {
        s_controls = *controls;
    }
}

void audio_engine_process(const float *input, float *output, size_t sample_count)
{
    if ((input == NULL) || (output == NULL)) {
        return;
    }

    for (size_t index = 0U; index < sample_count; ++index) {
#ifdef COMPOSED_PEDAL_RUNTIME
        output[index] = composed_pedal_process_sample(input[index]);
#else
        output[index] = input[index];
#endif
    }
}
