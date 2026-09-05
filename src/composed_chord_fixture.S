#ifdef COMPOSED_BLOCK_BENCHMARK
    .section .rodata.composed_chord,"a",%progbits
    .balign 4
    .global composed_chord_wav_start
    .global composed_chord_wav_end
composed_chord_wav_start:
    .incbin "simulation/samples/e_major_chord/e_major_dry.wav"
composed_chord_wav_end:
#endif
