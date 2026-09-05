#ifndef BOARD_H
#define BOARD_H

#include <stdbool.h>
#include <stdint.h>

void board_init(void);
uint32_t board_millis(void);
void board_status_led_set(bool enabled);
void board_status_led_toggle(void);

#endif
