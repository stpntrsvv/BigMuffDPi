"""Единообразно включает FPU Cortex-M4F на всех стадиях сборки."""

Import("env")

fpu_flags = ["-mfpu=fpv4-sp-d16", "-mfloat-abi=hard"]
env.Append(CCFLAGS=fpu_flags, ASFLAGS=fpu_flags, LINKFLAGS=fpu_flags)
