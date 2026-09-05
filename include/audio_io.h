#ifndef AUDIO_IO_H
#define AUDIO_IO_H

#include <stdbool.h>

bool audio_io_init(void);
bool audio_io_start(void);
void audio_io_poll(void);

#endif
