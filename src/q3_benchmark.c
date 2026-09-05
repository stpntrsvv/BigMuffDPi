#ifdef Q3_BENCHMARK

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include "stm32g4xx.h"
#include "q3_stream_fixture.h"

#define BENCHMARK_REPETITIONS 1000U
#if defined(Q3_BREAKDOWN) || defined(Q3_INLINE_STREAM)
#define STREAM_PERIOD_REPETITIONS 1U
#else
#define STREAM_PERIOD_REPETITIONS 1000U
#endif
#define INCREMENTAL_FALLBACK_FLAG 1U
#define INCREMENTAL_LARGE_STEP_FLAG 2U
/* При SCALE=1 вход хранит истинный аргумент 0,75, делённый на два. */
#define Q31_THREE_EIGHTHS ((int32_t)0x30000000)

typedef void (*benchmark_function_t)(void);
typedef void (*nonlinear_function_t)(
    const float q[3], float current[3], float first[3], float second[3]);

typedef struct {
    float q[3];
} nonlinear_state_t;

typedef struct {
    float q[3];
    float forward_exp;
    float diode_sinh;
    float diode_cosh;
} incremental_state_t;

#ifdef Q3_SCALAR_BENCHMARK
typedef struct {
    float diode_v;
    float diode_sinh;
    float diode_cosh;
} scalar_diode_state_t;
#endif

typedef struct {
    incremental_state_t nonlinear;
    float reciprocal_determinant;
} reciprocal_state_t;

typedef struct {
    incremental_state_t nonlinear;
    float m00;
    float m02;
    float m10;
    float m12;
    float m20;
    float m22;
} frozen_inverse_state_t;

typedef struct {
    uint32_t steps;
    uint32_t fallback_count;
    uint32_t large_step_count;
    uint32_t periodic_refresh_count;
    uint32_t average_cycles;
    uint32_t maximum_cycles;
    uint32_t maximum_cache_error_ppb;
    uint32_t nonfinite_count;
    uint32_t final_q_bits[3];
} stream_metrics_t;

typedef struct {
    uint32_t marker_cycles;
    uint32_t nonlinear_cycles;
    uint32_t residual_cycles;
    uint32_t newton_cycles;
    uint32_t range_check_cycles;
    uint32_t cordic_cycles;
    uint32_t state_update_cycles;
    uint32_t measured_sum_cycles;
} breakdown_metrics_t;

typedef struct {
    uint32_t average_cycles;
    uint32_t maximum_cycles;
    uint32_t final_q_bits[3];
    uint32_t final_cache_bits[3];
} kernel_loop_metrics_t;

typedef struct {
    uint32_t average_cycles;
    uint32_t fallback_count;
    uint32_t large_step_count;
    uint32_t periodic_refresh_count;
    uint32_t final_q_bits[3];
} block_stream_metrics_t;

static volatile float s_sink;
#ifdef Q3_FIXED_SOLVE_BENCHMARK
static volatile int32_t s_fixed_sink;
#endif
static volatile int32_t s_cordic_cosh;
static volatile int32_t s_cordic_sinh;
static volatile float s_forward_exp_seed;
static volatile float s_diode_sinh_seed;
static volatile float s_diode_cosh_seed;
static volatile float s_reciprocal_determinant_seed;
static volatile float s_inverse_seed[6];
static uint32_t s_strict_profile_overhead;

static volatile float s_q_seed[3] = {0.63128723F, -4.17381172F, -0.36478485F};
static volatile float s_matrix_seed[3][3] = {
    {64.5797553F, 0.0F, 0.0812277982F},
    {-18.4330564F, 1.0F, 0.163036476F},
    {-18.4330564F, 0.0F, 1.16312518F},
};
static volatile float s_right_seed[3] = {
    7.71502129e-8F,
    -2.22788409e-8F,
    -2.33069836e-8F,
};
#ifdef Q3_FIXED_SOLVE_BENCHMARK
static volatile int32_t s_fixed_matrix_seed[3][3] = {
    {541734252, 0, 681388},
    {-154627685, 8388608, 1367649},
    {-154627685, 0, 9757001},
};
static volatile int32_t s_fixed_right_seed[3] = {83, -24, -25};
#endif
static float s_linear_q[3] = {2.27177952F, -4.65859305F, -0.84957054F};
static const float s_influence[3][3] = {
    {4121.69579F, -3233.53448F, 1192.40418F},
    {-1194.96294F, 1399.66539F, 2393.33554F},
    {-1194.96294F, 1399.66539F, 2394.63762F},
};

__attribute__((noinline)) static uint32_t strict_profile_counter(void)
{
    __asm volatile("" ::: "memory");
    const uint32_t value = DWT->CYCCNT;
    __asm volatile("" ::: "memory");
    return value;
}

static uint32_t measure_strict_profile_overhead(void)
{
    uint64_t total = 0U;
    __disable_irq();
    for (uint32_t index = 0U; index < BENCHMARK_REPETITIONS; ++index) {
        const uint32_t start = strict_profile_counter();
        const uint32_t end = strict_profile_counter();
        total += end - start;
    }
    __enable_irq();
    return (uint32_t)(total / BENCHMARK_REPETITIONS);
}

static float limited_expf(float argument)
{
    if (argument > 80.0F) {
        argument = 80.0F;
    } else if (argument < -80.0F) {
        argument = -80.0F;
    }
    return expf(argument);
}

static void nonlinear_terms_libm(
    const float q[3], float current[3], float first[3], float second[3])
{
    const float forward_scale = 25.8649e-3F;
    const float reverse_scale = 25.8649e-3F;
    const float diode_scale = 1.9F * 25.8649e-3F;
    const float forward_is = 10.025e-15F;
    const float reverse_is = 12.0e-15F;
    const float diode_is_twice = 4.0e-9F;

    const float forward_exp = limited_expf(q[0] / forward_scale);
    const float reverse_exp = limited_expf(q[1] / reverse_scale);
    float diode_argument = q[2] / diode_scale;
    if (diode_argument > 80.0F) {
        diode_argument = 80.0F;
    } else if (diode_argument < -80.0F) {
        diode_argument = -80.0F;
    }
    const float diode_sinh = sinhf(diode_argument);
    const float diode_cosh = coshf(diode_argument);

    current[0] = forward_is * (forward_exp - 1.0F);
    current[1] = reverse_is * (reverse_exp - 1.0F);
    current[2] = diode_is_twice * diode_sinh;
    first[0] = forward_is * forward_exp / forward_scale;
    first[1] = reverse_is * reverse_exp / reverse_scale;
    first[2] = diode_is_twice * diode_cosh / diode_scale;
    second[0] = first[0] / forward_scale;
    second[1] = first[1] / reverse_scale;
    second[2] = diode_is_twice * diode_sinh / (diode_scale * diode_scale);
}

typedef union {
    float value;
    uint32_t bits;
} float_bits_t;

__attribute__((always_inline)) static inline float scale_power_of_two(
    float value, int32_t exponent_delta)
{
    float_bits_t representation = {.value = value};
    const int32_t exponent =
        (int32_t)((representation.bits >> 23U) & 0xFFU) + exponent_delta;
    if (exponent <= 0) {
        return 0.0F;
    }
    if (exponent >= 255) {
        representation.bits = 0x7F800000U;
        return representation.value;
    }
    representation.bits = (representation.bits & 0x807FFFFFU) |
                          ((uint32_t)exponent << 23U);
    return representation.value;
}

__attribute__((always_inline)) static inline float cordic_positive_exponential(
    float argument)
{
    const float inverse_ln2 = 1.4426950408889634F;
    const float ln2 = 0.6931471805599453F;
    const float q31_per_unit_at_scale_1 = 1073741824.0F;
    const float float_per_q31_at_scale_1 = 9.313225746154785e-10F;

    if (argument > 80.0F) {
        argument = 80.0F;
    } else if (argument <= -80.0F) {
        /* После умножения на ток насыщения это значение всё равно
         * округляется до нуля в float. */
        return 0.0F;
    }
    const float logarithm = argument * inverse_ln2;
    const int32_t exponent = (int32_t)(
        logarithm + (logarithm >= 0.0F ? 0.5F : -0.5F));
    const float remainder = argument - (float)exponent * ln2;
    const int32_t input_q31 = (int32_t)(remainder * q31_per_unit_at_scale_1);

    CORDIC->WDATA = (uint32_t)input_q31;
    const int32_t cosh_q31 = (int32_t)CORDIC->RDATA;
    const int32_t sinh_q31 = (int32_t)CORDIC->RDATA;
    const float exponential_remainder =
        (float)(cosh_q31 + sinh_q31) * float_per_q31_at_scale_1;
    return scale_power_of_two(exponential_remainder, exponent);
}

__attribute__((always_inline)) static inline void cordic_sinh_cosh(
    float argument, float *hyperbolic_sine, float *hyperbolic_cosine)
{
    const float inverse_ln2 = 1.4426950408889634F;
    const float ln2 = 0.6931471805599453F;
    const float q31_per_unit_at_scale_1 = 1073741824.0F;
    const float float_per_q31_at_scale_1 = 9.313225746154785e-10F;

    if (argument > 80.0F) {
        argument = 80.0F;
    } else if (argument < -80.0F) {
        argument = -80.0F;
    }
    const float logarithm = argument * inverse_ln2;
    const int32_t exponent = (int32_t)(
        logarithm + (logarithm >= 0.0F ? 0.5F : -0.5F));
    const float remainder = argument - (float)exponent * ln2;
    const int32_t input_q31 = (int32_t)(remainder * q31_per_unit_at_scale_1);

    CORDIC->WDATA = (uint32_t)input_q31;
    const int32_t cosh_q31 = (int32_t)CORDIC->RDATA;
    const int32_t sinh_q31 = (int32_t)CORDIC->RDATA;
    const float positive_remainder =
        (float)(cosh_q31 + sinh_q31) * float_per_q31_at_scale_1;
    const float negative_remainder =
        (float)(cosh_q31 - sinh_q31) * float_per_q31_at_scale_1;
    const float positive = scale_power_of_two(positive_remainder, exponent);
    const float negative = scale_power_of_two(negative_remainder, -exponent);
    *hyperbolic_sine = 0.5F * (positive - negative);
    *hyperbolic_cosine = 0.5F * (positive + negative);
}

__attribute__((always_inline)) static inline void cordic_direct_sinh_cosh(
    float argument, float *hyperbolic_sine, float *hyperbolic_cosine)
{
    const float q31_per_unit_at_scale_1 = 1073741824.0F;
    const float float_per_q31_at_scale_1 = 9.313225746154785e-10F;
    const int32_t input_q31 = (int32_t)(argument * q31_per_unit_at_scale_1);
    CORDIC->WDATA = (uint32_t)input_q31;
    const int32_t cosh_q31 = (int32_t)CORDIC->RDATA;
    const int32_t sinh_q31 = (int32_t)CORDIC->RDATA;
    *hyperbolic_cosine = (float)cosh_q31 * float_per_q31_at_scale_1;
    *hyperbolic_sine = (float)sinh_q31 * float_per_q31_at_scale_1;
}

static void cordic_exponential_pair(
    float argument, float *positive_exponential, float *negative_exponential)
{
    const float inverse_ln2 = 1.4426950408889634F;
    const float ln2 = 0.6931471805599453F;
    const float q31_per_unit_at_scale_1 = 1073741824.0F;
    const float float_per_q31_at_scale_1 = 9.313225746154785e-10F;

    if (argument > 80.0F) {
        argument = 80.0F;
    } else if (argument < -80.0F) {
        argument = -80.0F;
    }
    const float logarithm = argument * inverse_ln2;
    const int32_t exponent = (int32_t)(
        logarithm + (logarithm >= 0.0F ? 0.5F : -0.5F));
    const float remainder = argument - (float)exponent * ln2;
    const int32_t input_q31 = (int32_t)(remainder * q31_per_unit_at_scale_1);

    CORDIC->WDATA = (uint32_t)input_q31;
    const int32_t cosh_q31 = (int32_t)CORDIC->RDATA;
    const int32_t sinh_q31 = (int32_t)CORDIC->RDATA;
    const float cosh_remainder =
        (float)cosh_q31 * float_per_q31_at_scale_1;
    const float sinh_remainder =
        (float)sinh_q31 * float_per_q31_at_scale_1;
    *positive_exponential = scale_power_of_two(
        cosh_remainder + sinh_remainder, exponent);
    *negative_exponential = scale_power_of_two(
        cosh_remainder - sinh_remainder, -exponent);
}

static void nonlinear_terms_cordic(
    const float q[3], float current[3], float first[3], float second[3])
{
    const float forward_scale = 25.8649e-3F;
    const float reverse_scale = 25.8649e-3F;
    const float diode_scale = 1.9F * 25.8649e-3F;
    const float forward_is = 10.025e-15F;
    const float reverse_is = 12.0e-15F;
    const float diode_is_twice = 4.0e-9F;
    float forward_exp;
    float unused_forward_inverse;
    float reverse_exp;
    float unused_reverse_inverse;
    float diode_exp;
    float diode_inverse_exp;

    cordic_exponential_pair(
        q[0] / forward_scale, &forward_exp, &unused_forward_inverse);
    cordic_exponential_pair(
        q[1] / reverse_scale, &reverse_exp, &unused_reverse_inverse);
    cordic_exponential_pair(
        q[2] / diode_scale, &diode_exp, &diode_inverse_exp);
    const float diode_sinh = 0.5F * (diode_exp - diode_inverse_exp);
    const float diode_cosh = 0.5F * (diode_exp + diode_inverse_exp);

    current[0] = forward_is * (forward_exp - 1.0F);
    current[1] = reverse_is * (reverse_exp - 1.0F);
    current[2] = diode_is_twice * diode_sinh;
    first[0] = forward_is * forward_exp / forward_scale;
    first[1] = reverse_is * reverse_exp / reverse_scale;
    first[2] = diode_is_twice * diode_cosh / diode_scale;
    second[0] = first[0] / forward_scale;
    second[1] = first[1] / reverse_scale;
    second[2] = diode_is_twice * diode_sinh / (diode_scale * diode_scale);
}

static void solve_3x3(float matrix[3][3], float right[3], float result[3])
{
    const float a = matrix[0][0];
    const float b = matrix[0][1];
    const float c = matrix[0][2];
    const float d = matrix[1][0];
    const float e = matrix[1][1];
    const float f = matrix[1][2];
    const float g = matrix[2][0];
    const float h = matrix[2][1];
    const float i = matrix[2][2];
    const float cofactor_00 = e * i - f * h;
    const float cofactor_01 = f * g - d * i;
    const float cofactor_02 = d * h - e * g;
    const float inverse_determinant =
        1.0F / (a * cofactor_00 + b * cofactor_01 + c * cofactor_02);

    result[0] = inverse_determinant *
                (cofactor_00 * right[0] + (c * h - b * i) * right[1] +
                 (b * f - c * e) * right[2]);
    result[1] = inverse_determinant *
                (cofactor_01 * right[0] + (a * i - c * g) * right[1] +
                 (c * d - a * f) * right[2]);
    result[2] = inverse_determinant *
                (cofactor_02 * right[0] + (b * g - a * h) * right[1] +
                 (a * e - b * d) * right[2]);
}

#ifdef Q3_FIXED_SOLVE_BENCHMARK
static int32_t rounded_shift_i64(int64_t value, uint32_t bits)
{
    const int64_t half = (int64_t)1 << (bits - 1U);
    if (value >= 0) {
        return (int32_t)((value + half) >> bits);
    }
    return (int32_t)(-((-value + half) >> bits));
}

static void solve_3x3_fixed(
    int32_t matrix[3][3], int32_t right[3], int32_t result[3])
{
    const uint32_t matrix_fractional_bits = 23U;
    for (uint32_t column = 0U; column < 3U; ++column) {
        uint32_t pivot_row = column;
        int64_t pivot_magnitude = matrix[column][column];
        if (pivot_magnitude < 0) {
            pivot_magnitude = -pivot_magnitude;
        }
        for (uint32_t row = column + 1U; row < 3U; ++row) {
            int64_t magnitude = matrix[row][column];
            if (magnitude < 0) {
                magnitude = -magnitude;
            }
            if (magnitude > pivot_magnitude) {
                pivot_magnitude = magnitude;
                pivot_row = row;
            }
        }
        if (pivot_row != column) {
            const int32_t right_temporary = right[column];
            right[column] = right[pivot_row];
            right[pivot_row] = right_temporary;
            for (uint32_t index = column; index < 3U; ++index) {
                const int32_t temporary = matrix[column][index];
                matrix[column][index] = matrix[pivot_row][index];
                matrix[pivot_row][index] = temporary;
            }
        }
        const int32_t pivot = matrix[column][column];
        for (uint32_t row = column + 1U; row < 3U; ++row) {
            const int32_t factor = (int32_t)(
                ((int64_t)matrix[row][column] << matrix_fractional_bits) /
                pivot);
            for (uint32_t index = column; index < 3U; ++index) {
                matrix[row][index] -= rounded_shift_i64(
                    (int64_t)factor * matrix[column][index],
                    matrix_fractional_bits);
            }
            right[row] -= rounded_shift_i64(
                (int64_t)factor * right[column], matrix_fractional_bits);
        }
    }
    for (int32_t row = 2; row >= 0; --row) {
        int32_t numerator = right[row];
        for (int32_t column = row + 1; column < 3; ++column) {
            numerator -= rounded_shift_i64(
                (int64_t)matrix[row][column] * result[column],
                matrix_fractional_bits);
        }
        result[row] = (int32_t)(
            ((int64_t)numerator << matrix_fractional_bits) /
            matrix[row][row]);
    }
}
#endif

__attribute__((noinline)) static void newton_cordic_optimized_kernel(
    nonlinear_state_t *state)
{
    const float inverse_transistor_scale = 38.66243441884562F;
    const float inverse_diode_scale = 20.348649694129275F;
    const float forward_is = 10.025e-15F;
    const float reverse_is = 12.0e-15F;
    const float diode_is_twice = 4.0e-9F;
    const float q0 = state->q[0];
    const float q1 = state->q[1];
    const float q2 = state->q[2];
    const float forward_exp = cordic_positive_exponential(
        q0 * inverse_transistor_scale);
    const float reverse_exp = cordic_positive_exponential(
        q1 * inverse_transistor_scale);
    float diode_sinh;
    float diode_cosh;
    cordic_sinh_cosh(
        q2 * inverse_diode_scale, &diode_sinh, &diode_cosh);

    const float current0 = forward_is * (forward_exp - 1.0F);
    const float current1 = reverse_is * (reverse_exp - 1.0F);
    const float current2 = diode_is_twice * diode_sinh;
    const float first0 =
        forward_is * forward_exp * inverse_transistor_scale;
    const float first1 =
        reverse_is * reverse_exp * inverse_transistor_scale;
    const float first2 =
        diode_is_twice * diode_cosh * inverse_diode_scale;

    const float residual0 =
        q0 - s_linear_q[0] + s_influence[0][0] * current0 +
        s_influence[0][1] * current1 + s_influence[0][2] * current2;
    const float residual1 =
        q1 - s_linear_q[1] + s_influence[1][0] * current0 +
        s_influence[1][1] * current1 + s_influence[1][2] * current2;
    const float residual2 =
        q2 - s_linear_q[2] + s_influence[2][0] * current0 +
        s_influence[2][1] * current1 + s_influence[2][2] * current2;

    const float a = 1.0F + s_influence[0][0] * first0;
    const float b = s_influence[0][1] * first1;
    const float c = s_influence[0][2] * first2;
    const float d = s_influence[1][0] * first0;
    const float e = 1.0F + s_influence[1][1] * first1;
    const float f = s_influence[1][2] * first2;
    const float g = s_influence[2][0] * first0;
    const float h = s_influence[2][1] * first1;
    const float i = 1.0F + s_influence[2][2] * first2;
    const float right0 = -residual0;
    const float right1 = -residual1;
    const float right2 = -residual2;
    float correction0;
    float correction1;
    float correction2;
    if (first1 == 0.0F) {
        const float inverse_determinant_2x2 = 1.0F / (a * i - c * g);
        correction0 =
            (right0 * i - c * right2) * inverse_determinant_2x2;
        correction2 =
            (a * right2 - g * right0) * inverse_determinant_2x2;
        correction1 = right1 - d * correction0 - f * correction2;
    } else {
        const float cofactor00 = e * i - f * h;
        const float cofactor01 = f * g - d * i;
        const float cofactor02 = d * h - e * g;
        const float inverse_determinant =
            1.0F / (a * cofactor00 + b * cofactor01 + c * cofactor02);
        correction0 = inverse_determinant *
            (cofactor00 * right0 + (c * h - b * i) * right1 +
             (b * f - c * e) * right2);
        correction1 = inverse_determinant *
            (cofactor01 * right0 + (a * i - c * g) * right1 +
             (c * d - a * f) * right2);
        correction2 = inverse_determinant *
            (cofactor02 * right0 + (b * g - a * h) * right1 +
             (a * e - b * d) * right2);
    }

    state->q[0] = q0 + correction0;
    state->q[1] = q1 + correction1;
    state->q[2] = q2 + correction2;
}

__attribute__((always_inline)) static inline uint32_t
newton_cordic_incremental_inline(
    incremental_state_t *state)
{
    const float inverse_transistor_scale = 38.66243441884562F;
    const float inverse_diode_scale = 20.348649694129275F;
    const float forward_is = 10.025e-15F;
    const float reverse_is = 12.0e-15F;
    const float diode_is_twice = 4.0e-9F;
    const float q0 = state->q[0];
    const float q1 = state->q[1];
    const float q2 = state->q[2];
    const float forward_exp = state->forward_exp;
    const float diode_sinh = state->diode_sinh;
    const float diode_cosh = state->diode_cosh;
    const float current0 = forward_is * (forward_exp - 1.0F);
    const float current1 = -reverse_is;
    const float current2 = diode_is_twice * diode_sinh;
    const float first0 =
        forward_is * forward_exp * inverse_transistor_scale;
    const float first2 =
        diode_is_twice * diode_cosh * inverse_diode_scale;

#if defined(Q3_LOCAL_RESIDUAL)
    /* Явно просим одно округление на каждое умножение со сложением. */
    const float residual0 = fmaf(
        s_influence[0][2], current2,
        fmaf(s_influence[0][1], current1,
             fmaf(s_influence[0][0], current0, q0 - s_linear_q[0])));
    const float residual1 = fmaf(
        s_influence[1][2], current2,
        fmaf(s_influence[1][1], current1,
             fmaf(s_influence[1][0], current0, q1 - s_linear_q[1])));
    const float residual2 = fmaf(
        s_influence[2][2], current2,
        fmaf(s_influence[2][1], current1,
             fmaf(s_influence[2][0], current0, q2 - s_linear_q[2])));
#else
    const float residual0 =
        q0 - s_linear_q[0] + s_influence[0][0] * current0 +
        s_influence[0][1] * current1 + s_influence[0][2] * current2;
    const float residual1 =
        q1 - s_linear_q[1] + s_influence[1][0] * current0 +
        s_influence[1][1] * current1 + s_influence[1][2] * current2;
    const float residual2 =
        q2 - s_linear_q[2] + s_influence[2][0] * current0 +
        s_influence[2][1] * current1 + s_influence[2][2] * current2;
#endif

#if defined(Q3_LOCAL_NEWTON)
    const float a = fmaf(s_influence[0][0], first0, 1.0F);
    const float c = s_influence[0][2] * first2;
    const float d = s_influence[1][0] * first0;
    const float f = s_influence[1][2] * first2;
    /* Строки 1 и 2 имеют один коэффициент: g в точности равен d. */
    const float g = d;
    const float i = fmaf(s_influence[2][2], first2, 1.0F);
#elif defined(Q3_REUSE_EQUAL_DERIVATIVE)
    const float a = 1.0F + s_influence[0][0] * first0;
    const float c = s_influence[0][2] * first2;
    const float d = s_influence[1][0] * first0;
    const float f = s_influence[1][2] * first2;
    const float g = d;
    const float i = 1.0F + s_influence[2][2] * first2;
#else
    const float a = 1.0F + s_influence[0][0] * first0;
    const float c = s_influence[0][2] * first2;
    const float d = s_influence[1][0] * first0;
    const float f = s_influence[1][2] * first2;
    const float g = s_influence[2][0] * first0;
    const float i = 1.0F + s_influence[2][2] * first2;
#endif
    const float right0 = -residual0;
    const float right1 = -residual1;
    const float right2 = -residual2;
#if defined(Q3_LOCAL_NEWTON)
    const float inverse_determinant = 1.0F / fmaf(-c, g, a * i);
    const float correction0 =
        fmaf(-c, right2, right0 * i) * inverse_determinant;
    const float correction2 =
        fmaf(-g, right0, a * right2) * inverse_determinant;
    const float correction1 =
        fmaf(-f, correction2, fmaf(-d, correction0, right1));
#else
    const float inverse_determinant = 1.0F / (a * i - c * g);
    const float correction0 =
        (right0 * i - c * right2) * inverse_determinant;
    const float correction2 =
        (a * right2 - g * right0) * inverse_determinant;
    const float correction1 = right1 - d * correction0 - f * correction2;
#endif
    const float next_q0 = q0 + correction0;
    const float next_q1 = q1 + correction1;
    const float next_q2 = q2 + correction2;
    const float forward_delta = correction0 * inverse_transistor_scale;
    const float diode_delta = correction2 * inverse_diode_scale;
    const uint32_t direct_cordic_limit_bits = 0x3F8F1AA0U; /* 1.118F */
    const float_bits_t forward_delta_bits = {.value = forward_delta};
    const float_bits_t diode_delta_bits = {.value = diode_delta};

    float forward_delta_sinh;
    float forward_delta_cosh;
    float diode_delta_sinh;
    float diode_delta_cosh;
    uint32_t result_flags = 0U;
    if (((forward_delta_bits.bits & 0x7FFFFFFFU) > 0x3E800000U) ||
        ((diode_delta_bits.bits & 0x7FFFFFFFU) > 0x3E800000U)) {
        result_flags |= INCREMENTAL_LARGE_STEP_FLAG;
    }
    if (((forward_delta_bits.bits & 0x7FFFFFFFU) <=
         direct_cordic_limit_bits) &&
        ((diode_delta_bits.bits & 0x7FFFFFFFU) <=
         direct_cordic_limit_bits)) {
        cordic_direct_sinh_cosh(
            forward_delta, &forward_delta_sinh, &forward_delta_cosh);
        cordic_direct_sinh_cosh(
            diode_delta, &diode_delta_sinh, &diode_delta_cosh);

        state->forward_exp =
            forward_exp * (forward_delta_cosh + forward_delta_sinh);
        state->diode_sinh =
            diode_sinh * diode_delta_cosh + diode_cosh * diode_delta_sinh;
        state->diode_cosh =
            diode_cosh * diode_delta_cosh + diode_sinh * diode_delta_sinh;
    } else {
        result_flags |= INCREMENTAL_FALLBACK_FLAG;
        state->forward_exp = cordic_positive_exponential(
            next_q0 * inverse_transistor_scale);
        cordic_sinh_cosh(
            next_q2 * inverse_diode_scale,
            &state->diode_sinh,
            &state->diode_cosh);
    }

    state->q[0] = next_q0;
    state->q[1] = next_q1;
    state->q[2] = next_q2;
    return result_flags;
}

__attribute__((noinline)) static uint32_t newton_cordic_incremental_kernel(
    incremental_state_t *state)
{
    return newton_cordic_incremental_inline(state);
}

__attribute__((always_inline)) static inline void refresh_incremental_state(
    incremental_state_t *state)
{
    state->forward_exp = cordic_positive_exponential(
        state->q[0] * 38.66243441884562F);
    cordic_sinh_cosh(
        state->q[2] * 20.348649694129275F,
        &state->diode_sinh,
        &state->diode_cosh);
}

#ifdef Q3_SCALAR_BENCHMARK
__attribute__((always_inline)) static inline void refresh_scalar_diode_state(
    scalar_diode_state_t *state)
{
    cordic_sinh_cosh(
        state->diode_v * 20.348649694129275F,
        &state->diode_sinh,
        &state->diode_cosh);
}

__attribute__((always_inline)) static inline uint32_t scalar_diode_inline(
    scalar_diode_state_t *state, float linear_diode_v)
{
    const float inverse_diode_scale = 20.348649694129275F;
    const float diode_is_twice = 4.0e-9F;
    const float diode_v = state->diode_v;
    const float current = diode_is_twice * state->diode_sinh;
    const float first =
        diode_is_twice * state->diode_cosh * inverse_diode_scale;
    const float residual =
        diode_v - linear_diode_v + Q3_SCALAR_DIODE_INFLUENCE * current;
    const float jacobian =
        1.0F + Q3_SCALAR_DIODE_INFLUENCE * first;
    const float correction = -residual / jacobian;
    const float next_diode_v = diode_v + correction;
    const float diode_delta = correction * inverse_diode_scale;
    const float_bits_t diode_delta_bits = {.value = diode_delta};
    const uint32_t magnitude = diode_delta_bits.bits & 0x7FFFFFFFU;
    uint32_t result_flags = 0U;

    if (magnitude > 0x3E800000U) {
        result_flags |= INCREMENTAL_LARGE_STEP_FLAG;
    }
    if (magnitude <= 0x3F8F1AA0U) {
        float delta_sinh;
        float delta_cosh;
        cordic_direct_sinh_cosh(diode_delta, &delta_sinh, &delta_cosh);
        const float old_sinh = state->diode_sinh;
        const float old_cosh = state->diode_cosh;
        state->diode_sinh =
            old_sinh * delta_cosh + old_cosh * delta_sinh;
        state->diode_cosh =
            old_cosh * delta_cosh + old_sinh * delta_sinh;
    } else {
        result_flags |= INCREMENTAL_FALLBACK_FLAG;
        cordic_sinh_cosh(
            next_diode_v * inverse_diode_scale,
            &state->diode_sinh,
            &state->diode_cosh);
    }
    state->diode_v = next_diode_v;
    return result_flags;
}
#endif

__attribute__((noinline)) static uint32_t
newton_cordic_incremental_reciprocal_kernel(reciprocal_state_t *state)
{
    const float inverse_transistor_scale = 38.66243441884562F;
    const float inverse_diode_scale = 20.348649694129275F;
    const float forward_is = 10.025e-15F;
    const float reverse_is = 12.0e-15F;
    const float diode_is_twice = 4.0e-9F;
    incremental_state_t *const nonlinear = &state->nonlinear;
    const float q0 = nonlinear->q[0];
    const float q1 = nonlinear->q[1];
    const float q2 = nonlinear->q[2];
    const float forward_exp = nonlinear->forward_exp;
    const float diode_sinh = nonlinear->diode_sinh;
    const float diode_cosh = nonlinear->diode_cosh;
    const float current0 = forward_is * (forward_exp - 1.0F);
    const float current1 = -reverse_is;
    const float current2 = diode_is_twice * diode_sinh;
    const float first0 = forward_is * forward_exp * inverse_transistor_scale;
    const float first2 = diode_is_twice * diode_cosh * inverse_diode_scale;
    const float residual0 =
        q0 - s_linear_q[0] + s_influence[0][0] * current0 +
        s_influence[0][1] * current1 + s_influence[0][2] * current2;
    const float residual1 =
        q1 - s_linear_q[1] + s_influence[1][0] * current0 +
        s_influence[1][1] * current1 + s_influence[1][2] * current2;
    const float residual2 =
        q2 - s_linear_q[2] + s_influence[2][0] * current0 +
        s_influence[2][1] * current1 + s_influence[2][2] * current2;
    const float a = 1.0F + s_influence[0][0] * first0;
    const float c = s_influence[0][2] * first2;
    const float d = s_influence[1][0] * first0;
    const float f = s_influence[1][2] * first2;
    const float g = s_influence[2][0] * first0;
    const float i = 1.0F + s_influence[2][2] * first2;
    const float right0 = -residual0;
    const float right1 = -residual1;
    const float right2 = -residual2;
    const float determinant = a * i - c * g;
    const float reciprocal = state->reciprocal_determinant *
        (2.0F - determinant * state->reciprocal_determinant);
    const float correction0 = (right0 * i - c * right2) * reciprocal;
    const float correction2 = (a * right2 - g * right0) * reciprocal;
    const float correction1 = right1 - d * correction0 - f * correction2;
    const float next_q0 = q0 + correction0;
    const float next_q1 = q1 + correction1;
    const float next_q2 = q2 + correction2;
    const float forward_delta = correction0 * inverse_transistor_scale;
    const float diode_delta = correction2 * inverse_diode_scale;
    const float_bits_t forward_delta_bits = {.value = forward_delta};
    const float_bits_t diode_delta_bits = {.value = diode_delta};
    uint32_t result_flags = 0U;
    if (((forward_delta_bits.bits & 0x7FFFFFFFU) > 0x3E800000U) ||
        ((diode_delta_bits.bits & 0x7FFFFFFFU) > 0x3E800000U)) {
        result_flags |= INCREMENTAL_LARGE_STEP_FLAG;
    }
    if (((forward_delta_bits.bits & 0x7FFFFFFFU) <= 0x3F8F1AA0U) &&
        ((diode_delta_bits.bits & 0x7FFFFFFFU) <= 0x3F8F1AA0U)) {
        float forward_delta_sinh;
        float forward_delta_cosh;
        float diode_delta_sinh;
        float diode_delta_cosh;
        cordic_direct_sinh_cosh(
            forward_delta, &forward_delta_sinh, &forward_delta_cosh);
        cordic_direct_sinh_cosh(
            diode_delta, &diode_delta_sinh, &diode_delta_cosh);
        nonlinear->forward_exp =
            forward_exp * (forward_delta_cosh + forward_delta_sinh);
        nonlinear->diode_sinh =
            diode_sinh * diode_delta_cosh + diode_cosh * diode_delta_sinh;
        nonlinear->diode_cosh =
            diode_cosh * diode_delta_cosh + diode_sinh * diode_delta_sinh;
    } else {
        result_flags |= INCREMENTAL_FALLBACK_FLAG;
        nonlinear->forward_exp = cordic_positive_exponential(
            next_q0 * inverse_transistor_scale);
        cordic_sinh_cosh(
            next_q2 * inverse_diode_scale,
            &nonlinear->diode_sinh,
            &nonlinear->diode_cosh);
    }
    nonlinear->q[0] = next_q0;
    nonlinear->q[1] = next_q1;
    nonlinear->q[2] = next_q2;
    state->reciprocal_determinant = reciprocal;
    return result_flags;
}

__attribute__((noinline)) static uint32_t
newton_cordic_incremental_frozen_kernel(frozen_inverse_state_t *state)
{
    const float inverse_transistor_scale = 38.66243441884562F;
    const float inverse_diode_scale = 20.348649694129275F;
    const float forward_is = 10.025e-15F;
    const float reverse_is = 12.0e-15F;
    const float diode_is_twice = 4.0e-9F;
    incremental_state_t *const nonlinear = &state->nonlinear;
    const float q0 = nonlinear->q[0];
    const float q1 = nonlinear->q[1];
    const float q2 = nonlinear->q[2];
    const float forward_exp = nonlinear->forward_exp;
    const float diode_sinh = nonlinear->diode_sinh;
    const float diode_cosh = nonlinear->diode_cosh;
    const float current0 = forward_is * (forward_exp - 1.0F);
    const float current1 = -reverse_is;
    const float current2 = diode_is_twice * diode_sinh;
    const float residual0 =
        q0 - s_linear_q[0] + s_influence[0][0] * current0 +
        s_influence[0][1] * current1 + s_influence[0][2] * current2;
    const float residual1 =
        q1 - s_linear_q[1] + s_influence[1][0] * current0 +
        s_influence[1][1] * current1 + s_influence[1][2] * current2;
    const float residual2 =
        q2 - s_linear_q[2] + s_influence[2][0] * current0 +
        s_influence[2][1] * current1 + s_influence[2][2] * current2;
    const float right0 = -residual0;
    const float right1 = -residual1;
    const float right2 = -residual2;
    const float correction0 = state->m00 * right0 + state->m02 * right2;
    const float correction1 =
        right1 + state->m10 * right0 + state->m12 * right2;
    const float correction2 = state->m20 * right0 + state->m22 * right2;
    const float next_q0 = q0 + correction0;
    const float next_q1 = q1 + correction1;
    const float next_q2 = q2 + correction2;
    const float forward_delta = correction0 * inverse_transistor_scale;
    const float diode_delta = correction2 * inverse_diode_scale;
    const float_bits_t forward_delta_bits = {.value = forward_delta};
    const float_bits_t diode_delta_bits = {.value = diode_delta};
    uint32_t result_flags = 0U;
    if (((forward_delta_bits.bits & 0x7FFFFFFFU) > 0x3E800000U) ||
        ((diode_delta_bits.bits & 0x7FFFFFFFU) > 0x3E800000U)) {
        result_flags |= INCREMENTAL_LARGE_STEP_FLAG;
    }
    if (((forward_delta_bits.bits & 0x7FFFFFFFU) <= 0x3F8F1AA0U) &&
        ((diode_delta_bits.bits & 0x7FFFFFFFU) <= 0x3F8F1AA0U)) {
        float forward_delta_sinh;
        float forward_delta_cosh;
        float diode_delta_sinh;
        float diode_delta_cosh;
        cordic_direct_sinh_cosh(
            forward_delta, &forward_delta_sinh, &forward_delta_cosh);
        cordic_direct_sinh_cosh(
            diode_delta, &diode_delta_sinh, &diode_delta_cosh);
        nonlinear->forward_exp =
            forward_exp * (forward_delta_cosh + forward_delta_sinh);
        nonlinear->diode_sinh =
            diode_sinh * diode_delta_cosh + diode_cosh * diode_delta_sinh;
        nonlinear->diode_cosh =
            diode_cosh * diode_delta_cosh + diode_sinh * diode_delta_sinh;
    } else {
        result_flags |= INCREMENTAL_FALLBACK_FLAG;
        nonlinear->forward_exp = cordic_positive_exponential(
            next_q0 * inverse_transistor_scale);
        cordic_sinh_cosh(
            next_q2 * inverse_diode_scale,
            &nonlinear->diode_sinh,
            &nonlinear->diode_cosh);
    }
    nonlinear->q[0] = next_q0;
    nonlinear->q[1] = next_q1;
    nonlinear->q[2] = next_q2;
    return result_flags;
}

static void refresh_inverse_state(frozen_inverse_state_t *state)
{
    const float first0 = 10.025e-15F * state->nonlinear.forward_exp *
        38.66243441884562F;
    const float first2 = 4.0e-9F * state->nonlinear.diode_cosh *
        20.348649694129275F;
    const float a = 1.0F + s_influence[0][0] * first0;
    const float c = s_influence[0][2] * first2;
    const float d = s_influence[1][0] * first0;
    const float f = s_influence[1][2] * first2;
    const float g = s_influence[2][0] * first0;
    const float i = 1.0F + s_influence[2][2] * first2;
    const float reciprocal = 1.0F / (a * i - c * g);
    state->m00 = i * reciprocal;
    state->m02 = -c * reciprocal;
    state->m20 = -g * reciprocal;
    state->m22 = a * reciprocal;
    state->m10 = -d * state->m00 - f * state->m20;
    state->m12 = -d * state->m02 - f * state->m22;
}

__attribute__((always_inline)) static inline void refresh_reciprocal_state(
    reciprocal_state_t *state)
{
    const float first0 = 10.025e-15F * state->nonlinear.forward_exp *
        38.66243441884562F;
    const float first2 = 4.0e-9F * state->nonlinear.diode_cosh *
        20.348649694129275F;
    const float a = 1.0F + s_influence[0][0] * first0;
    const float c = s_influence[0][2] * first2;
    const float g = s_influence[2][0] * first0;
    const float i = 1.0F + s_influence[2][2] * first2;
    state->reciprocal_determinant = 1.0F / (a * i - c * g);
}

static void form_residual_and_jacobian(
    const float q[3],
    float residual[3],
    float jacobian[3][3],
    float second[3],
    nonlinear_function_t nonlinear_function)
{
    float current[3];
    float first[3];
    nonlinear_function(q, current, first, second);
    for (uint32_t row = 0U; row < 3U; ++row) {
        residual[row] = q[row] - s_linear_q[row];
        for (uint32_t column = 0U; column < 3U; ++column) {
            residual[row] += s_influence[row][column] * current[column];
            jacobian[row][column] = s_influence[row][column] * first[column];
        }
        jacobian[row][row] += 1.0F;
    }
}

__attribute__((noinline)) static void newton_kernel(
    nonlinear_state_t *state, nonlinear_function_t nonlinear_function)
{
    float residual[3];
    float jacobian[3][3];
    float second[3];
    float correction[3];
    form_residual_and_jacobian(
        state->q, residual, jacobian, second, nonlinear_function);
    for (uint32_t index = 0U; index < 3U; ++index) {
        residual[index] = -residual[index];
    }
    solve_3x3(jacobian, residual, correction);
    for (uint32_t index = 0U; index < 3U; ++index) {
        state->q[index] += correction[index];
    }
}

__attribute__((noinline)) static void halley_kernel(
    nonlinear_state_t *state, nonlinear_function_t nonlinear_function)
{
    float residual[3];
    float jacobian[3][3];
    float newton_matrix[3][3];
    float newton_right[3];
    float second[3];
    float newton_correction[3];
    float halley_correction[3];
    form_residual_and_jacobian(
        state->q, residual, jacobian, second, nonlinear_function);

    for (uint32_t row = 0U; row < 3U; ++row) {
        newton_right[row] = -residual[row];
        for (uint32_t column = 0U; column < 3U; ++column) {
            newton_matrix[row][column] = jacobian[row][column];
        }
    }
    solve_3x3(newton_matrix, newton_right, newton_correction);

    for (uint32_t row = 0U; row < 3U; ++row) {
        for (uint32_t column = 0U; column < 3U; ++column) {
            jacobian[row][column] += 0.5F * s_influence[row][column] *
                                     second[column] * newton_correction[column];
        }
        residual[row] = -residual[row];
    }
    solve_3x3(jacobian, residual, halley_correction);
    for (uint32_t index = 0U; index < 3U; ++index) {
        state->q[index] += halley_correction[index];
    }
}

__attribute__((noinline)) static void benchmark_empty(void)
{
    __asm volatile("" ::: "memory");
}

__attribute__((noinline)) static void benchmark_nonlinear_libm(void)
{
    const float q[3] = {s_q_seed[0], s_q_seed[1], s_q_seed[2]};
    float current[3];
    float first[3];
    float second[3];
    nonlinear_terms_libm(q, current, first, second);
    s_sink = current[0] + first[2] + second[2];
}

__attribute__((noinline)) static void benchmark_nonlinear_cordic(void)
{
    const float q[3] = {s_q_seed[0], s_q_seed[1], s_q_seed[2]};
    float current[3];
    float first[3];
    float second[3];
    nonlinear_terms_cordic(q, current, first, second);
    s_sink = current[0] + first[2] + second[2];
}

__attribute__((noinline)) static void benchmark_solve(void)
{
    float matrix[3][3];
    float right[3];
    float result[3];
    for (uint32_t row = 0U; row < 3U; ++row) {
        right[row] = s_right_seed[row];
        for (uint32_t column = 0U; column < 3U; ++column) {
            matrix[row][column] = s_matrix_seed[row][column];
        }
    }
    solve_3x3(matrix, right, result);
    s_sink = result[0];
}

#ifdef Q3_FIXED_SOLVE_BENCHMARK
__attribute__((noinline)) static void benchmark_solve_fixed(void)
{
    int32_t matrix[3][3];
    int32_t right[3];
    int32_t result[3];
    for (uint32_t row = 0U; row < 3U; ++row) {
        right[row] = s_fixed_right_seed[row];
        result[row] = 0;
        for (uint32_t column = 0U; column < 3U; ++column) {
            matrix[row][column] = s_fixed_matrix_seed[row][column];
        }
    }
    solve_3x3_fixed(matrix, right, result);
    s_fixed_sink = result[0];
}
#endif

__attribute__((noinline)) static void benchmark_form_cordic(void)
{
    const float q[3] = {s_q_seed[0], s_q_seed[1], s_q_seed[2]};
    float residual[3];
    float jacobian[3][3];
    float second[3];
    float checksum = 0.0F;
    form_residual_and_jacobian(
        q, residual, jacobian, second, nonlinear_terms_cordic);
    for (uint32_t row = 0U; row < 3U; ++row) {
        checksum += residual[row] + second[row];
        for (uint32_t column = 0U; column < 3U; ++column) {
            checksum += jacobian[row][column];
        }
    }
    s_sink = checksum;
}

__attribute__((noinline)) static void benchmark_newton_libm(void)
{
    nonlinear_state_t state = {{s_q_seed[0], s_q_seed[1], s_q_seed[2]}};
    newton_kernel(&state, nonlinear_terms_libm);
    s_sink = state.q[0];
}

__attribute__((noinline)) static void benchmark_newton_cordic(void)
{
    nonlinear_state_t state = {{s_q_seed[0], s_q_seed[1], s_q_seed[2]}};
    newton_kernel(&state, nonlinear_terms_cordic);
    s_sink = state.q[0];
}

__attribute__((noinline)) static void benchmark_newton_cordic_optimized(void)
{
    nonlinear_state_t state = {{s_q_seed[0], s_q_seed[1], s_q_seed[2]}};
    newton_cordic_optimized_kernel(&state);
    s_sink = state.q[0] + state.q[1] + state.q[2];
}

__attribute__((noinline)) static void benchmark_newton_cordic_incremental(void)
{
    incremental_state_t state = {
        {s_q_seed[0], s_q_seed[1], s_q_seed[2]},
        s_forward_exp_seed,
        s_diode_sinh_seed,
        s_diode_cosh_seed,
    };
    newton_cordic_incremental_kernel(&state);
    s_sink = state.q[0] + state.q[1] + state.q[2] + state.forward_exp +
             state.diode_sinh + state.diode_cosh;
}

__attribute__((noinline)) static void
benchmark_newton_cordic_incremental_reciprocal(void)
{
    reciprocal_state_t state = {
        {
            {s_q_seed[0], s_q_seed[1], s_q_seed[2]},
            s_forward_exp_seed,
            s_diode_sinh_seed,
            s_diode_cosh_seed,
        },
        s_reciprocal_determinant_seed,
    };
    newton_cordic_incremental_reciprocal_kernel(&state);
    s_sink = state.nonlinear.q[0] + state.nonlinear.q[1] +
             state.nonlinear.q[2] + state.nonlinear.forward_exp +
             state.nonlinear.diode_sinh + state.nonlinear.diode_cosh +
             state.reciprocal_determinant;
}

__attribute__((noinline)) static void
benchmark_newton_cordic_incremental_frozen(void)
{
    frozen_inverse_state_t state = {
        {
            {s_q_seed[0], s_q_seed[1], s_q_seed[2]},
            s_forward_exp_seed,
            s_diode_sinh_seed,
            s_diode_cosh_seed,
        },
        s_inverse_seed[0], s_inverse_seed[1], s_inverse_seed[2],
        s_inverse_seed[3], s_inverse_seed[4], s_inverse_seed[5],
    };
    newton_cordic_incremental_frozen_kernel(&state);
    s_sink = state.nonlinear.q[0] + state.nonlinear.q[1] +
             state.nonlinear.q[2] + state.nonlinear.forward_exp +
             state.nonlinear.diode_sinh + state.nonlinear.diode_cosh;
}

__attribute__((noinline)) static void benchmark_inverse_refresh(void)
{
    frozen_inverse_state_t state = {
        {
            {s_q_seed[0], s_q_seed[1], s_q_seed[2]},
            s_forward_exp_seed,
            s_diode_sinh_seed,
            s_diode_cosh_seed,
        },
        0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
    };
    refresh_inverse_state(&state);
    s_sink = state.m00 + state.m02 + state.m10 +
             state.m12 + state.m20 + state.m22;
}

__attribute__((noinline)) static void benchmark_reciprocal_refresh(void)
{
    const float first0 = 10.025e-15F * s_forward_exp_seed *
        38.66243441884562F;
    const float first2 = 4.0e-9F * s_diode_cosh_seed *
        20.348649694129275F;
    const float a = 1.0F + s_influence[0][0] * first0;
    const float c = s_influence[0][2] * first2;
    const float g = s_influence[2][0] * first0;
    const float i = 1.0F + s_influence[2][2] * first2;
    s_sink = 1.0F / (a * i - c * g);
}

__attribute__((noinline)) static void benchmark_incremental_refresh(void)
{
    const float q0 = s_q_seed[0];
    const float q2 = s_q_seed[2];
    float diode_sinh;
    float diode_cosh;
    const float forward_exp = cordic_positive_exponential(
        q0 * 38.66243441884562F);
    cordic_sinh_cosh(q2 * 20.348649694129275F, &diode_sinh, &diode_cosh);
    s_sink = forward_exp + diode_sinh + diode_cosh;
}

static float absolute_float(float value)
{
    float_bits_t representation = {.value = value};
    representation.bits &= 0x7FFFFFFFU;
    return representation.value;
}

static float relative_state_error(float actual, float expected)
{
    float scale = absolute_float(expected);
    if (scale < 1.0F) {
        scale = 1.0F;
    }
    return absolute_float(actual - expected) / scale;
}

static bool finite_float(float value)
{
    const float_bits_t representation = {.value = value};
    return (representation.bits & 0x7F800000U) != 0x7F800000U;
}

__attribute__((always_inline)) static inline kernel_loop_metrics_t
measure_kernel_loop_impl(bool inline_version)
{
    const uint32_t repetitions = 16U;
    uint64_t total_cycles = 0U;
    uint32_t maximum_cycles = 0U;
    incremental_state_t state = {{0.0F, 0.0F, 0.0F}, 0.0F, 0.0F, 0.0F};

    for (uint32_t repetition = 0U; repetition < repetitions; ++repetition) {
        state.q[0] = q3_stream_nominal_seed_q[0];
        state.q[1] = q3_stream_nominal_seed_q[1];
        state.q[2] = q3_stream_nominal_seed_q[2];
        refresh_incremental_state(&state);
        for (uint32_t sample = 0U; sample < Q3_STREAM_SAMPLE_COUNT; ++sample) {
            s_linear_q[0] = q3_stream_nominal_linear_q[sample][0];
            s_linear_q[1] = q3_stream_nominal_linear_q[sample][1];
            s_linear_q[2] = q3_stream_nominal_linear_q[sample][2];
            __disable_irq();
            const uint32_t start = strict_profile_counter();
            if (inline_version) {
                (void)newton_cordic_incremental_inline(&state);
            } else {
                (void)newton_cordic_incremental_kernel(&state);
            }
            const uint32_t end = strict_profile_counter();
            const uint32_t raw_elapsed = end - start;
            const uint32_t elapsed = raw_elapsed > s_strict_profile_overhead ?
                raw_elapsed - s_strict_profile_overhead : 0U;
            __enable_irq();
            total_cycles += elapsed;
            if (elapsed > maximum_cycles) {
                maximum_cycles = elapsed;
            }
        }
    }

    kernel_loop_metrics_t result = {
        (uint32_t)(total_cycles /
                   (repetitions * Q3_STREAM_SAMPLE_COUNT)),
        maximum_cycles,
        {0U, 0U, 0U},
        {0U, 0U, 0U},
    };
    for (uint32_t index = 0U; index < 3U; ++index) {
        const float_bits_t q_bits = {.value = state.q[index]};
        result.final_q_bits[index] = q_bits.bits;
    }
    const float_bits_t forward_bits = {.value = state.forward_exp};
    const float_bits_t sinh_bits = {.value = state.diode_sinh};
    const float_bits_t cosh_bits = {.value = state.diode_cosh};
    result.final_cache_bits[0] = forward_bits.bits;
    result.final_cache_bits[1] = sinh_bits.bits;
    result.final_cache_bits[2] = cosh_bits.bits;
    return result;
}

__attribute__((noinline)) static kernel_loop_metrics_t
measure_kernel_loop_called(void)
{
    return measure_kernel_loop_impl(false);
}

__attribute__((noinline)) static kernel_loop_metrics_t
measure_kernel_loop_inlined(void)
{
    return measure_kernel_loop_impl(true);
}

static stream_metrics_t run_incremental_stream(
    const float seed_q[3],
    const float linear_q[Q3_STREAM_SAMPLE_COUNT][3],
    uint32_t period_repetitions,
    uint32_t refresh_period)
{
    incremental_state_t state = {
        {seed_q[0], seed_q[1], seed_q[2]}, 0.0F, 0.0F, 0.0F,
    };
    refresh_incremental_state(&state);
    stream_metrics_t metrics = {0};
    uint64_t total_cycles = 0U;
    float maximum_cache_error = 0.0F;
    uint32_t fast_refresh_steps = 0U;

    for (uint32_t repetition = 0U; repetition < period_repetitions; ++repetition) {
        for (uint32_t sample = 0U; sample < Q3_STREAM_SAMPLE_COUNT; ++sample) {
            for (uint32_t index = 0U; index < 3U; ++index) {
                s_linear_q[index] = linear_q[sample][index];
            }

            __disable_irq();
            const uint32_t start = strict_profile_counter();
            const uint32_t result_flags =
#ifdef Q3_CALLED_STREAM
                newton_cordic_incremental_kernel(&state);
#else
                newton_cordic_incremental_inline(&state);
#endif
            const bool fallback =
                (result_flags & INCREMENTAL_FALLBACK_FLAG) != 0U;
            const bool large_step =
                (result_flags & INCREMENTAL_LARGE_STEP_FLAG) != 0U;
            if (large_step) {
                fast_refresh_steps = 32U;
            }
            const uint32_t active_refresh_period =
                refresh_period != 0U ? refresh_period :
                (fast_refresh_steps != 0U ? 8U : 32U);
            const bool periodic = ((metrics.steps + 1U) &
                                   (active_refresh_period - 1U)) == 0U;
            if (periodic && !fallback) {
                refresh_incremental_state(&state);
                ++metrics.periodic_refresh_count;
            }
            const uint32_t end = strict_profile_counter();
            const uint32_t raw_elapsed = end - start;
            const uint32_t elapsed = raw_elapsed > s_strict_profile_overhead ?
                raw_elapsed - s_strict_profile_overhead : 0U;
            __enable_irq();

            total_cycles += elapsed;
            if (elapsed > metrics.maximum_cycles) {
                metrics.maximum_cycles = elapsed;
            }
            metrics.fallback_count += fallback ? 1U : 0U;
            metrics.large_step_count += large_step ? 1U : 0U;
            ++metrics.steps;
            if (fast_refresh_steps != 0U) {
                --fast_refresh_steps;
            }

            float reference_sinh;
            float reference_cosh;
            const float reference_exp = cordic_positive_exponential(
                state.q[0] * 38.66243441884562F);
            cordic_sinh_cosh(
                state.q[2] * 20.348649694129275F,
                &reference_sinh,
                &reference_cosh);
            float error = relative_state_error(state.forward_exp, reference_exp);
            const float sinh_error = relative_state_error(
                state.diode_sinh, reference_sinh);
            const float cosh_error = relative_state_error(
                state.diode_cosh, reference_cosh);
            if (sinh_error > error) {
                error = sinh_error;
            }
            if (cosh_error > error) {
                error = cosh_error;
            }
            if (error > maximum_cache_error) {
                maximum_cache_error = error;
            }
            if (!finite_float(state.q[0]) || !finite_float(state.q[1]) ||
                !finite_float(state.q[2]) || !finite_float(state.forward_exp) ||
                !finite_float(state.diode_sinh) ||
                !finite_float(state.diode_cosh)) {
                ++metrics.nonfinite_count;
            }
        }
    }

    metrics.average_cycles = (uint32_t)(total_cycles / metrics.steps);
    metrics.maximum_cache_error_ppb = (uint32_t)(
        maximum_cache_error * 1.0e9F + 0.5F);
    for (uint32_t index = 0U; index < 3U; ++index) {
        const float_bits_t bits = {.value = state.q[index]};
        metrics.final_q_bits[index] = bits.bits;
    }
    return metrics;
}

static block_stream_metrics_t run_incremental_block_stream(
    const float seed_q[3],
    const float linear_q[Q3_STREAM_SAMPLE_COUNT][3],
    uint32_t refresh_period)
{
    const uint32_t repetitions = 16U;
    uint64_t total_cycles = 0U;
    block_stream_metrics_t metrics = {0};
    incremental_state_t state = {{0.0F, 0.0F, 0.0F}, 0.0F, 0.0F, 0.0F};

    for (uint32_t repetition = 0U; repetition < repetitions; ++repetition) {
        state.q[0] = seed_q[0];
        state.q[1] = seed_q[1];
        state.q[2] = seed_q[2];
        refresh_incremental_state(&state);
        uint32_t fast_refresh_steps = 0U;
        __disable_irq();
        const uint32_t start = strict_profile_counter();
        for (uint32_t sample = 0U; sample < Q3_STREAM_SAMPLE_COUNT; ++sample) {
            s_linear_q[0] = linear_q[sample][0];
            s_linear_q[1] = linear_q[sample][1];
            s_linear_q[2] = linear_q[sample][2];
            const uint32_t result_flags =
                newton_cordic_incremental_inline(&state);
            const bool fallback =
                (result_flags & INCREMENTAL_FALLBACK_FLAG) != 0U;
            const bool large_step =
                (result_flags & INCREMENTAL_LARGE_STEP_FLAG) != 0U;
            if (large_step) {
                fast_refresh_steps = 32U;
            }
            const uint32_t active_refresh_period =
                refresh_period != 0U ? refresh_period :
                (fast_refresh_steps != 0U ? 8U : 32U);
            const bool periodic = ((sample + 1U) &
                                   (active_refresh_period - 1U)) == 0U;
            if (periodic && !fallback) {
                refresh_incremental_state(&state);
                ++metrics.periodic_refresh_count;
            }
            metrics.fallback_count += fallback ? 1U : 0U;
            metrics.large_step_count += large_step ? 1U : 0U;
            if (fast_refresh_steps != 0U) {
                --fast_refresh_steps;
            }
        }
        const uint32_t end = strict_profile_counter();
        __enable_irq();
        const uint32_t raw_elapsed = end - start;
        total_cycles += raw_elapsed > s_strict_profile_overhead ?
            raw_elapsed - s_strict_profile_overhead : 0U;
    }

    metrics.average_cycles = (uint32_t)(total_cycles /
        (repetitions * Q3_STREAM_SAMPLE_COUNT));
    for (uint32_t index = 0U; index < 3U; ++index) {
        const float_bits_t bits = {.value = state.q[index]};
        metrics.final_q_bits[index] = bits.bits;
    }
    return metrics;
}

#ifdef Q3_SCALAR_BENCHMARK
static block_stream_metrics_t run_scalar_diode_block_stream(
    float seed_diode_v,
    const float linear_diode_v[Q3_STREAM_SAMPLE_COUNT],
    uint32_t refresh_period)
{
    const uint32_t repetitions = 16U;
    uint64_t total_cycles = 0U;
    block_stream_metrics_t metrics = {0};
    scalar_diode_state_t state = {0.0F, 0.0F, 0.0F};

    for (uint32_t repetition = 0U; repetition < repetitions; ++repetition) {
        state.diode_v = seed_diode_v;
        refresh_scalar_diode_state(&state);
        uint32_t fast_refresh_steps = 0U;
        __disable_irq();
        const uint32_t start = strict_profile_counter();
        for (uint32_t sample = 0U; sample < Q3_STREAM_SAMPLE_COUNT; ++sample) {
            const uint32_t result_flags = scalar_diode_inline(
                &state, linear_diode_v[sample]);
            const bool fallback =
                (result_flags & INCREMENTAL_FALLBACK_FLAG) != 0U;
            const bool large_step =
                (result_flags & INCREMENTAL_LARGE_STEP_FLAG) != 0U;
            if (large_step) {
                fast_refresh_steps = 32U;
            }
            const uint32_t active_refresh_period =
                refresh_period != 0U ? refresh_period :
                (fast_refresh_steps != 0U ? 8U : 32U);
            const bool periodic = ((sample + 1U) &
                                   (active_refresh_period - 1U)) == 0U;
            if (periodic && !fallback) {
                refresh_scalar_diode_state(&state);
                ++metrics.periodic_refresh_count;
            }
            metrics.fallback_count += fallback ? 1U : 0U;
            metrics.large_step_count += large_step ? 1U : 0U;
            if (fast_refresh_steps != 0U) {
                --fast_refresh_steps;
            }
        }
        const uint32_t end = strict_profile_counter();
        __enable_irq();
        const uint32_t raw_elapsed = end - start;
        total_cycles += raw_elapsed > s_strict_profile_overhead ?
            raw_elapsed - s_strict_profile_overhead : 0U;
    }

    metrics.average_cycles = (uint32_t)(total_cycles /
        (repetitions * Q3_STREAM_SAMPLE_COUNT));
    const float_bits_t bits = {.value = state.diode_v};
    metrics.final_q_bits[2] = bits.bits;
    return metrics;
}
#endif

static stream_metrics_t run_reciprocal_stream(
    const float seed_q[3],
    const float linear_q[Q3_STREAM_SAMPLE_COUNT][3],
    uint32_t period_repetitions)
{
    reciprocal_state_t state = {
        {
            {seed_q[0], seed_q[1], seed_q[2]},
            0.0F, 0.0F, 0.0F,
        },
        0.0F,
    };
    refresh_incremental_state(&state.nonlinear);
    refresh_reciprocal_state(&state);
    stream_metrics_t metrics = {0};
    uint64_t total_cycles = 0U;
    float maximum_cache_error = 0.0F;
    uint32_t fast_refresh_steps = 0U;

    for (uint32_t repetition = 0U; repetition < period_repetitions; ++repetition) {
        for (uint32_t sample = 0U; sample < Q3_STREAM_SAMPLE_COUNT; ++sample) {
            for (uint32_t index = 0U; index < 3U; ++index) {
                s_linear_q[index] = linear_q[sample][index];
            }

            __disable_irq();
            const uint32_t start = strict_profile_counter();
            const uint32_t result_flags =
                newton_cordic_incremental_reciprocal_kernel(&state);
            const bool fallback =
                (result_flags & INCREMENTAL_FALLBACK_FLAG) != 0U;
            const bool large_step =
                (result_flags & INCREMENTAL_LARGE_STEP_FLAG) != 0U;
            if (large_step) {
                fast_refresh_steps = 32U;
            }
            const uint32_t active_refresh_period =
                fast_refresh_steps != 0U ? 8U : 32U;
            const bool nonlinear_refresh = ((metrics.steps + 1U) &
                (active_refresh_period - 1U)) == 0U;
            if (nonlinear_refresh && !fallback) {
                refresh_incremental_state(&state.nonlinear);
                ++metrics.periodic_refresh_count;
            }
            const bool reciprocal_refresh =
                (((metrics.steps + 1U) & 31U) == 0U) || large_step || fallback;
            if (reciprocal_refresh) {
                refresh_reciprocal_state(&state);
            }
            const uint32_t end = strict_profile_counter();
            const uint32_t raw_elapsed = end - start;
            const uint32_t elapsed = raw_elapsed > s_strict_profile_overhead ?
                raw_elapsed - s_strict_profile_overhead : 0U;
            __enable_irq();

            total_cycles += elapsed;
            if (elapsed > metrics.maximum_cycles) {
                metrics.maximum_cycles = elapsed;
            }
            metrics.fallback_count += fallback ? 1U : 0U;
            metrics.large_step_count += large_step ? 1U : 0U;
            ++metrics.steps;
            if (fast_refresh_steps != 0U) {
                --fast_refresh_steps;
            }

            float reference_sinh;
            float reference_cosh;
            const float reference_exp = cordic_positive_exponential(
                state.nonlinear.q[0] * 38.66243441884562F);
            cordic_sinh_cosh(
                state.nonlinear.q[2] * 20.348649694129275F,
                &reference_sinh,
                &reference_cosh);
            float error = relative_state_error(
                state.nonlinear.forward_exp, reference_exp);
            const float sinh_error = relative_state_error(
                state.nonlinear.diode_sinh, reference_sinh);
            const float cosh_error = relative_state_error(
                state.nonlinear.diode_cosh, reference_cosh);
            if (sinh_error > error) {
                error = sinh_error;
            }
            if (cosh_error > error) {
                error = cosh_error;
            }
            if (error > maximum_cache_error) {
                maximum_cache_error = error;
            }
            if (!finite_float(state.nonlinear.q[0]) ||
                !finite_float(state.nonlinear.q[1]) ||
                !finite_float(state.nonlinear.q[2]) ||
                !finite_float(state.nonlinear.forward_exp) ||
                !finite_float(state.nonlinear.diode_sinh) ||
                !finite_float(state.nonlinear.diode_cosh) ||
                !finite_float(state.reciprocal_determinant)) {
                ++metrics.nonfinite_count;
            }
        }
    }

    metrics.average_cycles = (uint32_t)(total_cycles / metrics.steps);
    metrics.maximum_cache_error_ppb = (uint32_t)(
        maximum_cache_error * 1.0e9F + 0.5F);
    for (uint32_t index = 0U; index < 3U; ++index) {
        const float_bits_t bits = {.value = state.nonlinear.q[index]};
        metrics.final_q_bits[index] = bits.bits;
    }
    return metrics;
}

__attribute__((noinline)) static void benchmark_halley_libm(void)
{
    nonlinear_state_t state = {{s_q_seed[0], s_q_seed[1], s_q_seed[2]}};
    halley_kernel(&state, nonlinear_terms_libm);
    s_sink = state.q[0];
}

__attribute__((noinline)) static void benchmark_halley_cordic(void)
{
    nonlinear_state_t state = {{s_q_seed[0], s_q_seed[1], s_q_seed[2]}};
    halley_kernel(&state, nonlinear_terms_cordic);
    s_sink = state.q[0];
}

__attribute__((noinline)) static void benchmark_cordic(void)
{
    CORDIC->WDATA = (uint32_t)Q31_THREE_EIGHTHS;
    s_cordic_cosh = (int32_t)CORDIC->RDATA;
    s_cordic_sinh = (int32_t)CORDIC->RDATA;
}

static uint32_t measure_average(benchmark_function_t function)
{
    __disable_irq();
    const uint32_t start = DWT->CYCCNT;
    for (uint32_t index = 0U; index < BENCHMARK_REPETITIONS; ++index) {
        function();
    }
    const uint32_t elapsed = DWT->CYCCNT - start;
    __enable_irq();
    return elapsed / BENCHMARK_REPETITIONS;
}

__attribute__((always_inline)) static inline uint32_t profile_counter(void)
{
    __asm volatile("" ::: "memory");
    const uint32_t value = DWT->CYCCNT;
    __asm volatile("" ::: "memory");
    return value;
}

static breakdown_metrics_t measure_incremental_breakdown(void)
{
    uint64_t marker_total = 0U;
    uint64_t nonlinear_total = 0U;
    uint64_t residual_total = 0U;
    uint64_t newton_total = 0U;
    uint64_t range_total = 0U;
    uint64_t cordic_total = 0U;
    uint64_t update_total = 0U;

    __disable_irq();
    for (uint32_t iteration = 0U; iteration < BENCHMARK_REPETITIONS; ++iteration) {
        const uint32_t start = profile_counter();
        const uint32_t end = profile_counter();
        marker_total += end - start;
    }

    for (uint32_t iteration = 0U; iteration < BENCHMARK_REPETITIONS; ++iteration) {
        incremental_state_t state = {
            {s_q_seed[0], s_q_seed[1], s_q_seed[2]},
            s_forward_exp_seed,
            s_diode_sinh_seed,
            s_diode_cosh_seed,
        };
        float q0;
        float q1;
        float q2;
        float forward_exp;
        float diode_sinh;
        float diode_cosh;
        float current0;
        float current1;
        float current2;
        float first0;
        float first2;
        float residual0;
        float residual1;
        float residual2;
        float correction0;
        float correction1;
        float correction2;
        float next_q0;
        float next_q1;
        float next_q2;
        float forward_delta;
        float diode_delta;
        uint32_t result_flags;
        float forward_delta_sinh;
        float forward_delta_cosh;
        float diode_delta_sinh;
        float diode_delta_cosh;

        uint32_t start = profile_counter();
        q0 = state.q[0];
        q1 = state.q[1];
        q2 = state.q[2];
        forward_exp = state.forward_exp;
        diode_sinh = state.diode_sinh;
        diode_cosh = state.diode_cosh;
        current0 = 10.025e-15F * (forward_exp - 1.0F);
        current1 = -12.0e-15F;
        current2 = 4.0e-9F * diode_sinh;
        first0 = 10.025e-15F * forward_exp * 38.66243441884562F;
        first2 = 4.0e-9F * diode_cosh * 20.348649694129275F;
        uint32_t end = profile_counter();
        nonlinear_total += end - start;

        start = profile_counter();
        residual0 = q0 - s_linear_q[0] +
            s_influence[0][0] * current0 + s_influence[0][1] * current1 +
            s_influence[0][2] * current2;
        residual1 = q1 - s_linear_q[1] +
            s_influence[1][0] * current0 + s_influence[1][1] * current1 +
            s_influence[1][2] * current2;
        residual2 = q2 - s_linear_q[2] +
            s_influence[2][0] * current0 + s_influence[2][1] * current1 +
            s_influence[2][2] * current2;
        end = profile_counter();
        residual_total += end - start;

        start = profile_counter();
        const float a = 1.0F + s_influence[0][0] * first0;
        const float c = s_influence[0][2] * first2;
        const float d = s_influence[1][0] * first0;
        const float f = s_influence[1][2] * first2;
        const float g = s_influence[2][0] * first0;
        const float i = 1.0F + s_influence[2][2] * first2;
        const float right0 = -residual0;
        const float right1 = -residual1;
        const float right2 = -residual2;
        const float reciprocal = 1.0F / (a * i - c * g);
        correction0 = (right0 * i - c * right2) * reciprocal;
        correction2 = (a * right2 - g * right0) * reciprocal;
        correction1 = right1 - d * correction0 - f * correction2;
        next_q0 = q0 + correction0;
        next_q1 = q1 + correction1;
        next_q2 = q2 + correction2;
        end = profile_counter();
        newton_total += end - start;

        start = profile_counter();
        forward_delta = correction0 * 38.66243441884562F;
        diode_delta = correction2 * 20.348649694129275F;
        const float_bits_t forward_bits = {.value = forward_delta};
        const float_bits_t diode_bits = {.value = diode_delta};
        result_flags = 0U;
        if (((forward_bits.bits & 0x7FFFFFFFU) > 0x3E800000U) ||
            ((diode_bits.bits & 0x7FFFFFFFU) > 0x3E800000U)) {
            result_flags |= INCREMENTAL_LARGE_STEP_FLAG;
        }
        if (((forward_bits.bits & 0x7FFFFFFFU) > 0x3F8F1AA0U) ||
            ((diode_bits.bits & 0x7FFFFFFFU) > 0x3F8F1AA0U)) {
            result_flags |= INCREMENTAL_FALLBACK_FLAG;
        }
        end = profile_counter();
        range_total += end - start;

        start = profile_counter();
        cordic_direct_sinh_cosh(
            forward_delta, &forward_delta_sinh, &forward_delta_cosh);
        cordic_direct_sinh_cosh(
            diode_delta, &diode_delta_sinh, &diode_delta_cosh);
        end = profile_counter();
        cordic_total += end - start;

        start = profile_counter();
        state.forward_exp = forward_exp *
            (forward_delta_cosh + forward_delta_sinh);
        state.diode_sinh = diode_sinh * diode_delta_cosh +
            diode_cosh * diode_delta_sinh;
        state.diode_cosh = diode_cosh * diode_delta_cosh +
            diode_sinh * diode_delta_sinh;
        state.q[0] = next_q0;
        state.q[1] = next_q1;
        state.q[2] = next_q2;
        s_sink = state.q[0] + state.q[1] + state.q[2] +
            state.forward_exp + state.diode_sinh + state.diode_cosh +
            (float)result_flags;
        end = profile_counter();
        update_total += end - start;
    }
    __enable_irq();

    breakdown_metrics_t result;
    result.marker_cycles = (uint32_t)(marker_total / BENCHMARK_REPETITIONS);
#define PROFILE_AVERAGE(total) \
    ((uint32_t)((total) / BENCHMARK_REPETITIONS) > result.marker_cycles ? \
     (uint32_t)((total) / BENCHMARK_REPETITIONS) - result.marker_cycles : 0U)
    result.nonlinear_cycles = PROFILE_AVERAGE(nonlinear_total);
    result.residual_cycles = PROFILE_AVERAGE(residual_total);
    result.newton_cycles = PROFILE_AVERAGE(newton_total);
    result.range_check_cycles = PROFILE_AVERAGE(range_total);
    result.cordic_cycles = PROFILE_AVERAGE(cordic_total);
    result.state_update_cycles = PROFILE_AVERAGE(update_total);
#undef PROFILE_AVERAGE
    result.measured_sum_cycles = result.nonlinear_cycles +
        result.residual_cycles + result.newton_cycles +
        result.range_check_cycles + result.cordic_cycles +
        result.state_update_cycles;
    return result;
}

static void clock_init_170mhz(void)
{
    RCC->APB1ENR1 |= RCC_APB1ENR1_PWREN;
    (void)RCC->APB1ENR1;
    PWR->CR5 &= ~PWR_CR5_R1MODE;
    PWR->CR1 = (PWR->CR1 & ~PWR_CR1_VOS) | PWR_CR1_VOS_0;
    while ((PWR->SR2 & PWR_SR2_VOSF) != 0U) {
    }

    FLASH->ACR = FLASH_ACR_PRFTEN | FLASH_ACR_ICEN | FLASH_ACR_DCEN |
                 FLASH_ACR_LATENCY_4WS;
    while ((FLASH->ACR & FLASH_ACR_LATENCY) != FLASH_ACR_LATENCY_4WS) {
    }

    RCC->CR |= RCC_CR_HSION;
    while ((RCC->CR & RCC_CR_HSIRDY) == 0U) {
    }
    RCC->CR &= ~RCC_CR_PLLON;
    while ((RCC->CR & RCC_CR_PLLRDY) != 0U) {
    }
    RCC->PLLCFGR = RCC_PLLCFGR_PLLSRC_HSI |
                   (3UL << RCC_PLLCFGR_PLLM_Pos) |
                   (85UL << RCC_PLLCFGR_PLLN_Pos) | RCC_PLLCFGR_PLLREN;
    RCC->CR |= RCC_CR_PLLON;
    while ((RCC->CR & RCC_CR_PLLRDY) == 0U) {
    }
    RCC->CFGR = (RCC->CFGR & ~(RCC_CFGR_SW | RCC_CFGR_HPRE | RCC_CFGR_PPRE1 |
                               RCC_CFGR_PPRE2)) |
                RCC_CFGR_SW_PLL;
    while ((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_PLL) {
    }
    SystemCoreClock = 170000000U;
}

static void uart_init(void)
{
    RCC->AHB2ENR |= RCC_AHB2ENR_GPIOAEN;
    RCC->APB1ENR1 |= RCC_APB1ENR1_USART2EN;
    (void)RCC->AHB2ENR;
    GPIOA->MODER = (GPIOA->MODER & ~(3UL << (2U * 2U))) | (2UL << (2U * 2U));
    GPIOA->AFR[0] = (GPIOA->AFR[0] & ~(0xFUL << (2U * 4U))) |
                    (7UL << (2U * 4U));
    GPIOA->OSPEEDR |= 3UL << (2U * 2U);
    USART2->BRR = (SystemCoreClock + 57600U) / 115200U;
    USART2->CR1 = USART_CR1_TE | USART_CR1_UE;
    while ((USART2->ISR & USART_ISR_TEACK) == 0U) {
    }
}

static void uart_character(char character)
{
    while ((USART2->ISR & USART_ISR_TXE) == 0U) {
    }
    USART2->TDR = (uint8_t)character;
}

static void uart_text(const char *text)
{
    while (*text != '\0') {
        uart_character(*text++);
    }
}

static void uart_unsigned(uint32_t value)
{
    char digits[10];
    uint32_t count = 0U;
    do {
        digits[count++] = (char)('0' + (value % 10U));
        value /= 10U;
    } while (value != 0U);
    while (count != 0U) {
        uart_character(digits[--count]);
    }
}

static void uart_signed(int32_t value)
{
    if (value < 0) {
        uart_character('-');
        uart_unsigned((uint32_t)(-(int64_t)value));
    } else {
        uart_unsigned((uint32_t)value);
    }
}

static void report_cycles(const char *name, uint32_t measured, uint32_t overhead)
{
    uart_text(name);
    uart_character('=');
    uart_unsigned(measured > overhead ? measured - overhead : 0U);
    uart_text("\r\n");
}

static void report_stream_metrics(
    const char *prefix, const stream_metrics_t *metrics)
{
    const char *const names[] = {
        "steps", "fallback_count", "large_step_count", "periodic_refresh_count",
        "average_cycles", "maximum_cycles", "maximum_cache_error_ppb",
        "nonfinite_count",
    };
    const uint32_t values[] = {
        metrics->steps,
        metrics->fallback_count,
        metrics->large_step_count,
        metrics->periodic_refresh_count,
        metrics->average_cycles,
        metrics->maximum_cycles,
        metrics->maximum_cache_error_ppb,
        metrics->nonfinite_count,
    };
    for (uint32_t index = 0U; index < 8U; ++index) {
        uart_text(prefix);
        uart_character('_');
        uart_text(names[index]);
        uart_character('=');
        uart_unsigned(values[index]);
        uart_text("\r\n");
    }
    for (uint32_t index = 0U; index < 3U; ++index) {
        uart_text(prefix);
        uart_text("_final_q");
        uart_unsigned(index);
        uart_text("_bits=");
        uart_unsigned(metrics->final_q_bits[index]);
        uart_text("\r\n");
    }
}

int main(void)
{
    clock_init_170mhz();
    uart_init();
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0U;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    s_strict_profile_overhead = measure_strict_profile_overhead();

    RCC->AHB1ENR |= RCC_AHB1ENR_CORDICEN;
    (void)RCC->AHB1ENR;
    RCC->AHB1RSTR |= RCC_AHB1RSTR_CORDICRST;
    RCC->AHB1RSTR &= ~RCC_AHB1RSTR_CORDICRST;
    (void)RCC->AHB1RSTR;
    CORDIC->CSR = CORDIC_CSR_FUNC_2 | CORDIC_CSR_FUNC_0 |
                  CORDIC_CSR_PRECISION_2 | CORDIC_CSR_PRECISION_1 |
                  CORDIC_CSR_SCALE_0 | CORDIC_CSR_NRES;

    s_forward_exp_seed = cordic_positive_exponential(
        s_q_seed[0] * 38.66243441884562F);
    float diode_sinh_seed;
    float diode_cosh_seed;
    cordic_sinh_cosh(
        s_q_seed[2] * 20.348649694129275F,
        &diode_sinh_seed,
        &diode_cosh_seed);
    s_diode_sinh_seed = diode_sinh_seed;
    s_diode_cosh_seed = diode_cosh_seed;
    frozen_inverse_state_t inverse_seed_state = {
        {
            {s_q_seed[0], s_q_seed[1], s_q_seed[2]},
            s_forward_exp_seed,
            s_diode_sinh_seed,
            s_diode_cosh_seed,
        },
        0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
    };
    refresh_inverse_state(&inverse_seed_state);
    s_inverse_seed[0] = inverse_seed_state.m00;
    s_inverse_seed[1] = inverse_seed_state.m02;
    s_inverse_seed[2] = inverse_seed_state.m10;
    s_inverse_seed[3] = inverse_seed_state.m12;
    s_inverse_seed[4] = inverse_seed_state.m20;
    s_inverse_seed[5] = inverse_seed_state.m22;
    const float seed_first0 = 10.025e-15F * s_forward_exp_seed *
        38.66243441884562F;
    const float seed_first2 = 4.0e-9F * s_diode_cosh_seed *
        20.348649694129275F;
    const float seed_a = 1.0F + s_influence[0][0] * seed_first0;
    const float seed_c = s_influence[0][2] * seed_first2;
    const float seed_g = s_influence[2][0] * seed_first0;
    const float seed_i = 1.0F + s_influence[2][2] * seed_first2;
    s_reciprocal_determinant_seed =
        1.0F / (seed_a * seed_i - seed_c * seed_g);

    const uint32_t overhead = measure_average(benchmark_empty);
    const uint32_t nonlinear_libm = measure_average(benchmark_nonlinear_libm);
    const uint32_t nonlinear_cordic = measure_average(benchmark_nonlinear_cordic);
    const uint32_t form_cordic = measure_average(benchmark_form_cordic);
    const uint32_t solve = measure_average(benchmark_solve);
#ifdef Q3_FIXED_SOLVE_BENCHMARK
    const uint32_t solve_fixed = measure_average(benchmark_solve_fixed);
#endif
    const uint32_t newton_libm = measure_average(benchmark_newton_libm);
    const uint32_t newton_cordic = measure_average(benchmark_newton_cordic);
    const uint32_t newton_cordic_optimized =
        measure_average(benchmark_newton_cordic_optimized);
    const uint32_t newton_cordic_incremental =
        measure_average(benchmark_newton_cordic_incremental);
    const uint32_t newton_cordic_incremental_reciprocal =
        measure_average(benchmark_newton_cordic_incremental_reciprocal);
    const uint32_t newton_cordic_incremental_frozen =
        measure_average(benchmark_newton_cordic_incremental_frozen);
    const uint32_t incremental_refresh =
        measure_average(benchmark_incremental_refresh);
    const uint32_t inverse_refresh =
        measure_average(benchmark_inverse_refresh);
    const uint32_t reciprocal_refresh =
        measure_average(benchmark_reciprocal_refresh);
    const uint32_t halley_libm = measure_average(benchmark_halley_libm);
    const uint32_t halley_cordic = measure_average(benchmark_halley_cordic);
    const uint32_t cordic = measure_average(benchmark_cordic);
    const breakdown_metrics_t breakdown = measure_incremental_breakdown();
    const kernel_loop_metrics_t called_kernel_loop =
        measure_kernel_loop_called();
    const kernel_loop_metrics_t inlined_kernel_loop =
        measure_kernel_loop_inlined();

    const float check_q[3] = {s_q_seed[0], s_q_seed[1], s_q_seed[2]};
    float libm_current[3];
    float libm_first[3];
    float libm_second[3];
    float cordic_current[3];
    float cordic_first[3];
    float cordic_second[3];
    nonlinear_terms_libm(check_q, libm_current, libm_first, libm_second);
    nonlinear_terms_cordic(check_q, cordic_current, cordic_first, cordic_second);
    nonlinear_state_t generic_newton_state = {
        {s_q_seed[0], s_q_seed[1], s_q_seed[2]},
    };
    nonlinear_state_t optimized_newton_state = {
        {s_q_seed[0], s_q_seed[1], s_q_seed[2]},
    };
    incremental_state_t incremental_newton_state = {
        {s_q_seed[0], s_q_seed[1], s_q_seed[2]},
        s_forward_exp_seed,
        s_diode_sinh_seed,
        s_diode_cosh_seed,
    };
    reciprocal_state_t reciprocal_newton_state = {
        incremental_newton_state,
        s_reciprocal_determinant_seed,
    };
    frozen_inverse_state_t frozen_newton_state = {
        incremental_newton_state,
        s_inverse_seed[0], s_inverse_seed[1], s_inverse_seed[2],
        s_inverse_seed[3], s_inverse_seed[4], s_inverse_seed[5],
    };
    newton_kernel(&generic_newton_state, nonlinear_terms_cordic);
    newton_cordic_optimized_kernel(&optimized_newton_state);
    newton_cordic_incremental_kernel(&incremental_newton_state);
    newton_cordic_incremental_reciprocal_kernel(&reciprocal_newton_state);
    newton_cordic_incremental_frozen_kernel(&frozen_newton_state);
    const stream_metrics_t nominal_stream = run_incremental_stream(
        q3_stream_nominal_seed_q,
        q3_stream_nominal_linear_q,
        STREAM_PERIOD_REPETITIONS,
        32U);
    const stream_metrics_t strong_stream_r1 = run_incremental_stream(
        q3_stream_strong_seed_q,
        q3_stream_strong_linear_q,
        STREAM_PERIOD_REPETITIONS,
        1U);
    const stream_metrics_t strong_stream_r8 = run_incremental_stream(
        q3_stream_strong_seed_q,
        q3_stream_strong_linear_q,
        STREAM_PERIOD_REPETITIONS,
        8U);
    const stream_metrics_t strong_stream_r16 = run_incremental_stream(
        q3_stream_strong_seed_q,
        q3_stream_strong_linear_q,
        STREAM_PERIOD_REPETITIONS,
        16U);
    const stream_metrics_t strong_stream_r32 = run_incremental_stream(
        q3_stream_strong_seed_q,
        q3_stream_strong_linear_q,
        STREAM_PERIOD_REPETITIONS,
        32U);
    const stream_metrics_t strong_stream_adaptive = run_incremental_stream(
        q3_stream_strong_seed_q,
        q3_stream_strong_linear_q,
        STREAM_PERIOD_REPETITIONS,
        0U);
    const block_stream_metrics_t nominal_block_stream =
        run_incremental_block_stream(
            q3_stream_nominal_seed_q,
            q3_stream_nominal_linear_q,
            32U);
    const block_stream_metrics_t strong_block_stream =
        run_incremental_block_stream(
            q3_stream_strong_seed_q,
            q3_stream_strong_linear_q,
            0U);
#ifdef Q3_SCALAR_BENCHMARK
    const block_stream_metrics_t nominal_scalar_block_stream =
        run_scalar_diode_block_stream(
            q3_stream_nominal_seed_q[2],
            q3_stream_nominal_scalar_linear_q,
            32U);
    const block_stream_metrics_t strong_scalar_block_stream =
        run_scalar_diode_block_stream(
            q3_stream_strong_seed_q[2],
            q3_stream_strong_scalar_linear_q,
            0U);
#endif
    const stream_metrics_t nominal_stream_reciprocal = run_reciprocal_stream(
        q3_stream_nominal_seed_q,
        q3_stream_nominal_linear_q,
        STREAM_PERIOD_REPETITIONS);
    const stream_metrics_t strong_stream_reciprocal = run_reciprocal_stream(
        q3_stream_strong_seed_q,
        q3_stream_strong_linear_q,
        STREAM_PERIOD_REPETITIONS);
    const char *const term_names[9] = {
        "forward_current",
        "reverse_current",
        "diode_current",
        "forward_first",
        "reverse_first",
        "diode_first",
        "forward_second",
        "reverse_second",
        "diode_second",
    };
    const float *const libm_groups[3] = {libm_current, libm_first, libm_second};
    const float *const cordic_groups[3] = {
        cordic_current,
        cordic_first,
        cordic_second,
    };

#ifdef Q3_FIXED_SOLVE_BENCHMARK
    uart_text("Q3_FIXED_SOLVE_BENCHMARK_V1\r\n");
#else
    uart_text("Q3_BENCHMARK_V12\r\n");
#endif
    uart_text("core_hz=");
    uart_unsigned(SystemCoreClock);
    uart_text("\r\nrepetitions=");
    uart_unsigned(BENCHMARK_REPETITIONS);
    uart_text("\r\noverhead_cycles=");
    uart_unsigned(overhead);
    uart_text("\r\nstrict_profile_overhead_cycles=");
    uart_unsigned(s_strict_profile_overhead);
    uart_text("\r\n");
    report_cycles("nonlinear_libm_cycles", nonlinear_libm, overhead);
    report_cycles("nonlinear_cordic_cycles", nonlinear_cordic, overhead);
    report_cycles("form_cordic_cycles", form_cordic, overhead);
    report_cycles("solve_3x3_cycles", solve, overhead);
#ifdef Q3_FIXED_SOLVE_BENCHMARK
    report_cycles("solve_3x3_fixed_cycles", solve_fixed, overhead);
#endif
    report_cycles("newton_libm_cycles", newton_libm, overhead);
    report_cycles("newton_cordic_cycles", newton_cordic, overhead);
    report_cycles(
        "newton_cordic_optimized_cycles", newton_cordic_optimized, overhead);
    report_cycles(
        "newton_cordic_incremental_cycles",
        newton_cordic_incremental,
        overhead);
    report_cycles(
        "newton_cordic_incremental_reciprocal_cycles",
        newton_cordic_incremental_reciprocal,
        overhead);
    report_cycles(
        "newton_cordic_incremental_frozen_cycles",
        newton_cordic_incremental_frozen,
        overhead);
    report_cycles("incremental_refresh_cycles", incremental_refresh, overhead);
    report_cycles("inverse_refresh_cycles", inverse_refresh, overhead);
    report_cycles("reciprocal_refresh_cycles", reciprocal_refresh, overhead);
    report_cycles("halley_libm_cycles", halley_libm, overhead);
    report_cycles("halley_cordic_cycles", halley_cordic, overhead);
    report_cycles("cordic_cosh_sinh_cycles", cordic, overhead);
    uart_text("breakdown_marker_cycles=");
    uart_unsigned(breakdown.marker_cycles);
    uart_text("\r\nbreakdown_nonlinear_cycles=");
    uart_unsigned(breakdown.nonlinear_cycles);
    uart_text("\r\nbreakdown_residual_cycles=");
    uart_unsigned(breakdown.residual_cycles);
    uart_text("\r\nbreakdown_newton_cycles=");
    uart_unsigned(breakdown.newton_cycles);
    uart_text("\r\nbreakdown_range_check_cycles=");
    uart_unsigned(breakdown.range_check_cycles);
    uart_text("\r\nbreakdown_cordic_cycles=");
    uart_unsigned(breakdown.cordic_cycles);
    uart_text("\r\nbreakdown_state_update_cycles=");
    uart_unsigned(breakdown.state_update_cycles);
    uart_text("\r\nbreakdown_measured_sum_cycles=");
    uart_unsigned(breakdown.measured_sum_cycles);
    uart_text("\r\n");
    uart_text("kernel_called_average_cycles=");
    uart_unsigned(called_kernel_loop.average_cycles);
    uart_text("\r\nkernel_called_maximum_cycles=");
    uart_unsigned(called_kernel_loop.maximum_cycles);
    uart_text("\r\nkernel_inlined_average_cycles=");
    uart_unsigned(inlined_kernel_loop.average_cycles);
    uart_text("\r\nkernel_inlined_maximum_cycles=");
    uart_unsigned(inlined_kernel_loop.maximum_cycles);
    uart_text("\r\n");
    for (uint32_t index = 0U; index < 3U; ++index) {
        uart_text("kernel_called_final_q");
        uart_unsigned(index);
        uart_text("_bits=");
        uart_unsigned(called_kernel_loop.final_q_bits[index]);
        uart_text("\r\nkernel_inlined_final_q");
        uart_unsigned(index);
        uart_text("_bits=");
        uart_unsigned(inlined_kernel_loop.final_q_bits[index]);
        uart_text("\r\nkernel_called_final_cache");
        uart_unsigned(index);
        uart_text("_bits=");
        uart_unsigned(called_kernel_loop.final_cache_bits[index]);
        uart_text("\r\nkernel_inlined_final_cache");
        uart_unsigned(index);
        uart_text("_bits=");
        uart_unsigned(inlined_kernel_loop.final_cache_bits[index]);
        uart_text("\r\n");
    }
#ifdef Q3_SCALAR_BENCHMARK
    uart_text("scalar_block_nominal_average_cycles=");
    uart_unsigned(nominal_scalar_block_stream.average_cycles);
    uart_text("\r\nscalar_block_strong_adaptive_average_cycles=");
    uart_unsigned(strong_scalar_block_stream.average_cycles);
    uart_text("\r\nscalar_block_nominal_fallback_count=");
    uart_unsigned(nominal_scalar_block_stream.fallback_count);
    uart_text("\r\nscalar_block_strong_fallback_count=");
    uart_unsigned(strong_scalar_block_stream.fallback_count);
    uart_text("\r\nscalar_block_nominal_final_q2_bits=");
    uart_unsigned(nominal_scalar_block_stream.final_q_bits[2]);
    uart_text("\r\nscalar_block_strong_final_q2_bits=");
    uart_unsigned(strong_scalar_block_stream.final_q_bits[2]);
    uart_text("\r\n");
#endif
    uart_text("stream_factor=");
    uart_unsigned(Q3_STREAM_FACTOR);
    uart_text("\r\nstream_signal_hz=");
    uart_unsigned(Q3_STREAM_SIGNAL_HZ);
    uart_text("\r\nstream_period_repetitions=");
    uart_unsigned(STREAM_PERIOD_REPETITIONS);
    uart_text("\r\n");
    report_stream_metrics("stream_nominal", &nominal_stream);
    report_stream_metrics("stream_strong_r1", &strong_stream_r1);
    report_stream_metrics("stream_strong_r8", &strong_stream_r8);
    report_stream_metrics("stream_strong_r16", &strong_stream_r16);
    report_stream_metrics("stream_strong_r32", &strong_stream_r32);
    report_stream_metrics("stream_strong_adaptive", &strong_stream_adaptive);
    uart_text("block_nominal_average_cycles=");
    uart_unsigned(nominal_block_stream.average_cycles);
    uart_text("\r\nblock_strong_adaptive_average_cycles=");
    uart_unsigned(strong_block_stream.average_cycles);
    uart_text("\r\n");
    for (uint32_t index = 0U; index < 3U; ++index) {
        uart_text("block_nominal_final_q");
        uart_unsigned(index);
        uart_text("_bits=");
        uart_unsigned(nominal_block_stream.final_q_bits[index]);
        uart_text("\r\nblock_strong_adaptive_final_q");
        uart_unsigned(index);
        uart_text("_bits=");
        uart_unsigned(strong_block_stream.final_q_bits[index]);
        uart_text("\r\n");
    }
    report_stream_metrics(
        "stream_nominal_reciprocal", &nominal_stream_reciprocal);
    report_stream_metrics(
        "stream_strong_reciprocal", &strong_stream_reciprocal);
    uart_text("cordic_cosh_q31=");
    uart_signed(s_cordic_cosh);
    uart_text("\r\ncordic_cosh_raw=");
    uart_unsigned((uint32_t)s_cordic_cosh);
    uart_text("\r\ncordic_sinh_q31=");
    uart_signed(s_cordic_sinh);
    uart_text("\r\ncordic_sinh_raw=");
    uart_unsigned((uint32_t)s_cordic_sinh);
    uart_text("\r\ncordic_csr=");
    uart_unsigned(CORDIC->CSR);
    uart_text("\r\n");
    for (uint32_t group = 0U; group < 3U; ++group) {
        for (uint32_t index = 0U; index < 3U; ++index) {
            const uint32_t flat_index = group * 3U + index;
            const float_bits_t libm_bits = {.value = libm_groups[group][index]};
            const float_bits_t cordic_bits = {
                .value = cordic_groups[group][index],
            };
            uart_text("libm_");
            uart_text(term_names[flat_index]);
            uart_text("_bits=");
            uart_unsigned(libm_bits.bits);
            uart_text("\r\ncordic_");
            uart_text(term_names[flat_index]);
            uart_text("_bits=");
            uart_unsigned(cordic_bits.bits);
            uart_text("\r\n");
        }
    }
    for (uint32_t index = 0U; index < 3U; ++index) {
        const float_bits_t generic_bits = {
            .value = generic_newton_state.q[index],
        };
        const float_bits_t optimized_bits = {
            .value = optimized_newton_state.q[index],
        };
        uart_text("newton_generic_q");
        uart_unsigned(index);
        uart_text("_bits=");
        uart_unsigned(generic_bits.bits);
        uart_text("\r\nnewton_optimized_q");
        uart_unsigned(index);
        uart_text("_bits=");
        uart_unsigned(optimized_bits.bits);
        const float_bits_t incremental_bits = {
            .value = incremental_newton_state.q[index],
        };
        uart_text("\r\nnewton_incremental_q");
        uart_unsigned(index);
        uart_text("_bits=");
        uart_unsigned(incremental_bits.bits);
        const float_bits_t reciprocal_bits = {
            .value = reciprocal_newton_state.nonlinear.q[index],
        };
        const float_bits_t frozen_bits = {
            .value = frozen_newton_state.nonlinear.q[index],
        };
        uart_text("\r\nnewton_reciprocal_q");
        uart_unsigned(index);
        uart_text("_bits=");
        uart_unsigned(reciprocal_bits.bits);
        uart_text("\r\nnewton_frozen_q");
        uart_unsigned(index);
        uart_text("_bits=");
        uart_unsigned(frozen_bits.bits);
        uart_text("\r\n");
    }
    uart_text("END\r\n");
    while ((USART2->ISR & USART_ISR_TC) == 0U) {
    }

    for (;;) {
        __WFI();
    }
}

#endif
