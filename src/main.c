#include "audio_engine.h"
#include "audio_io.h"
#include "board.h"

#include "stm32g4xx.h"

#if !defined(Q3_BENCHMARK) && !defined(FULL_PEDAL_BENCHMARK) && !defined(PORT_MULTIRATE_BENCHMARK) && !defined(FMAC_BENCHMARK) && !defined(COMPOSED_FRONTEND_BENCHMARK)
int main(void)
{
    board_init();
    audio_engine_init();

    const bool audio_ready = audio_io_init() && audio_io_start();
    uint32_t previous_blink_ms = board_millis();
    const uint32_t blink_period_ms = audio_ready ? 1000U : 250U;

    for (;;) {
        audio_io_poll();

        const uint32_t now_ms = board_millis();
        if ((uint32_t)(now_ms - previous_blink_ms) >= blink_period_ms) {
            previous_blink_ms = now_ms;
            board_status_led_toggle();
        }

        __WFI();
    }
}
#endif
