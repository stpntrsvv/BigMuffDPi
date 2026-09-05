* Big Muff Pi V3 topology reconstructed from the archived ElectroSmash page.
* Required external nodes: in, out, vcc, ground 0.
* Required parameters: SUSTAIN, TONE, VOLUME in the closed interval [0, 1].

* Input booster Q4
R2 in c1_left 39k
C1 c1_left q4_b 1u
R14 q4_b 0 47k
R13 vcc q4_c 10k
R22 q4_e 0 100
R9 q4_c q4_b 470k
C10 q4_c q4_b 470p
Q4 q4_c q4_b q4_e BC239

* Sustain control and coupling to first clipping stage
C4 q4_c sustain_top 1u
R_SUSTAIN_TOP sustain_top sustain_w {max(1, 100k*(1-SUSTAIN))}
R_SUSTAIN_BOTTOM sustain_w sustain_bottom {max(1, 100k*SUSTAIN)}
R23 sustain_bottom 0 1k
C5 sustain_w q3_drive 100n
R19 q3_drive q3_b 10k

* First clipping stage Q3
R20 q3_b 0 100k
R18 vcc q3_c 10k
R21 q3_e 0 150
R17 q3_c q3_b 470k
C12 q3_c q3_b 470p
Q3 q3_c q3_b q3_e BC239
C6 q3_b q3_diode 1u
D3 q3_diode q3_c D1N914
D4 q3_c q3_diode D1N914

* Coupling to second clipping stage
C13 q3_c q2_drive 100n
R12 q2_drive q2_b 10k

* Second clipping stage Q2
R16 q2_b 0 100k
R11 vcc q2_c 10k
R10 q2_e 0 150
R15 q2_c q2_b 470k
C11 q2_c q2_b 470p
Q2 q2_c q2_b q2_e BC239
C7 q2_b q2_diode 1u
D1 q2_diode q2_c D1N914
D2 q2_c q2_diode D1N914

* Passive tone control: upper branch is high-pass, lower branch is low-pass
C9 q2_c tone_high 4n
R5 tone_high 0 22k
R8 q2_c tone_low 39k
C8 tone_low 0 10n
R_TONE_TOP tone_high tone_w {max(1, 100k*(1-TONE))}
R_TONE_BOTTOM tone_w tone_low {max(1, 100k*TONE)}

* Output booster Q1
C3 tone_w q1_b 100n
R7 vcc q1_b 430k
R3 q1_b 0 100k
R6 vcc q1_c 15k
R4 q1_e 0 3.3k
Q1 q1_c q1_b q1_e BC239

* Volume control and output load
C2 q1_c volume_top 100n
R_VOLUME_TOP volume_top out {max(1, 100k*(1-VOLUME))}
R_VOLUME_BOTTOM out 0 {max(1, 100k*VOLUME)}
R_LOAD out 0 1Meg
