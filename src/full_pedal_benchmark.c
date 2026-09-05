#ifdef FULL_PEDAL_BENCHMARK

#include <stdint.h>

#include "stm32g4xx.h"
#include "full_pedal_fixture.h"
#include "full_pedal_working.h"

#define MEASUREMENT_PASSES 8U
#define FORWARD_SCALE 25.8649e-3F
#define DIODE_SCALE (1.9F * 25.8649e-3F)
#define FORWARD_IS 10.025e-15F
#define REVERSE_IS 12.0e-15F
#define DIODE_IS_TWICE 4.0e-9F

typedef union {
    float value;
    uint32_t bits;
} float_bits_t;

typedef struct {
    float state[FULL_PEDAL_STATE_COUNT];
    float q[FULL_PEDAL_ACTIVE_COUNT];
    float nonlinear_value[FULL_PEDAL_ACTIVE_COUNT];
    float nonlinear_cosh[FULL_PEDAL_ACTIVE_COUNT];
    float output;
    uint32_t sample_index;
    uint32_t fallback_count;
} pedal_state_t;

typedef struct {
    uint32_t average_cycles;
    uint32_t maximum_cycles;
    uint32_t nonfinite_count;
    uint32_t final_output_bits;
    uint32_t fallback_count;
} stream_metrics_t;

static volatile float s_sink;

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
    const float q31_per_unit = 1073741824.0F;
    const float unit_per_q31 = 9.313225746154785e-10F;
    if (argument > 80.0F) {
        argument = 80.0F;
    } else if (argument <= -80.0F) {
        return 0.0F;
    }
    const float logarithm = argument * inverse_ln2;
    const int32_t exponent = (int32_t)(
        logarithm + (logarithm >= 0.0F ? 0.5F : -0.5F));
    const float remainder = argument - (float)exponent * ln2;
    CORDIC->WDATA = (uint32_t)(int32_t)(remainder * q31_per_unit);
    const int32_t cosh_q31 = (int32_t)CORDIC->RDATA;
    const int32_t sinh_q31 = (int32_t)CORDIC->RDATA;
    return scale_power_of_two(
        (float)(cosh_q31 + sinh_q31) * unit_per_q31, exponent);
}

__attribute__((always_inline)) static inline void cordic_sinh_cosh(
    float argument, float *sinh_value, float *cosh_value)
{
    const float inverse_ln2 = 1.4426950408889634F;
    const float ln2 = 0.6931471805599453F;
    const float q31_per_unit = 1073741824.0F;
    const float unit_per_q31 = 9.313225746154785e-10F;
    if (argument > 80.0F) {
        argument = 80.0F;
    } else if (argument < -80.0F) {
        argument = -80.0F;
    }
    const float logarithm = argument * inverse_ln2;
    const int32_t exponent = (int32_t)(
        logarithm + (logarithm >= 0.0F ? 0.5F : -0.5F));
    const float remainder = argument - (float)exponent * ln2;
    CORDIC->WDATA = (uint32_t)(int32_t)(remainder * q31_per_unit);
    const int32_t cosh_q31 = (int32_t)CORDIC->RDATA;
    const int32_t sinh_q31 = (int32_t)CORDIC->RDATA;
    const float positive = scale_power_of_two(
        (float)(cosh_q31 + sinh_q31) * unit_per_q31, exponent);
    const float negative = scale_power_of_two(
        (float)(cosh_q31 - sinh_q31) * unit_per_q31, -exponent);
    *sinh_value = 0.5F * (positive - negative);
    *cosh_value = 0.5F * (positive + negative);
}

__attribute__((always_inline)) static inline void nonlinear_from_cache(
    uint32_t index, const pedal_state_t *pedal, float *current, float *first)
{
    if ((index == 2U) || (index == 3U)) {
        *current = DIODE_IS_TWICE * pedal->nonlinear_value[index];
        *first = DIODE_IS_TWICE * pedal->nonlinear_cosh[index] / DIODE_SCALE;
    } else {
        const float saturation = index == 5U ? REVERSE_IS : FORWARD_IS;
        const float exponential = pedal->nonlinear_value[index];
        *current = saturation * (exponential - 1.0F);
        *first = saturation * exponential / FORWARD_SCALE;
    }
}

static void nonlinear_all_cached(
    const pedal_state_t *pedal, float current[6], float first[6])
{
    for (uint32_t index = 0U; index < 6U; ++index) {
        nonlinear_from_cache(index, pedal, &current[index], &first[index]);
    }
}

static void nonlinear_refresh_one(pedal_state_t *pedal, uint32_t index)
{
    if ((index == 2U) || (index == 3U)) {
        cordic_sinh_cosh(
            pedal->q[index] / DIODE_SCALE,
            &pedal->nonlinear_value[index], &pedal->nonlinear_cosh[index]);
    } else {
        pedal->nonlinear_value[index] = cordic_positive_exponential(
            pedal->q[index] / FORWARD_SCALE);
        pedal->nonlinear_cosh[index] = 0.0F;
    }
}

static void nonlinear_refresh_all(pedal_state_t *pedal)
{
    for (uint32_t index = 0U; index < 6U; ++index) {
        nonlinear_refresh_one(pedal, index);
    }
}

__attribute__((always_inline)) static inline void nonlinear_cache_update(
    pedal_state_t *pedal, uint32_t index, float old_q)
{
    const float inverse_scale =
        ((index == 2U) || (index == 3U)) ?
        (1.0F / DIODE_SCALE) : (1.0F / FORWARD_SCALE);
    const float delta = (pedal->q[index] - old_q) * inverse_scale;
    if (__builtin_fabsf(delta) > 0.25F) {
        nonlinear_refresh_one(pedal, index);
        ++pedal->fallback_count;
        return;
    }
    const float square = delta * delta;
    if ((index == 2U) || (index == 3U)) {
        const float sinh_delta = delta * (1.0F + square * (1.0F / 6.0F));
        const float cosh_delta = 1.0F + square * 0.5F;
        const float old_sinh = pedal->nonlinear_value[index];
        const float old_cosh = pedal->nonlinear_cosh[index];
        pedal->nonlinear_value[index] =
            old_sinh * cosh_delta + old_cosh * sinh_delta;
        pedal->nonlinear_cosh[index] =
            old_cosh * cosh_delta + old_sinh * sinh_delta;
    } else {
        const float multiplier =
            1.0F + delta * (1.0F + delta * (0.5F + delta * (1.0F / 6.0F)));
        pedal->nonlinear_value[index] *= multiplier;
    }
}

static void working_coefficients_init(void)
{
    for (uint32_t row = 0U; row < 6U; ++row) {
        working_active_bias[row] = full_pedal_active_bias[row];
        working_active_input[row] = full_pedal_active_input[row];
        working_output_active[row] = full_pedal_output_active[row];
        for (uint32_t column = 0U; column < 13U; ++column) {
            working_active_state[row][column] =
                full_pedal_active_state[row][column];
        }
        for (uint32_t column = 0U; column < 6U; ++column) {
            working_active_influence[row][column] =
                full_pedal_active_influence[row][column];
        }
    }
    for (uint32_t row = 0U; row < 13U; ++row) {
        working_state_bias[row] = full_pedal_state_bias[row];
        working_state_input[row] = full_pedal_state_input[row];
        working_state_pole[row] = full_pedal_state_pole[row];
        working_output_state[row] = full_pedal_output_state[row];
        for (uint32_t column = 0U; column < 6U; ++column) {
            working_state_active[row][column] =
                full_pedal_state_active[row][column];
        }
    }
    working_output_bias = full_pedal_output_bias;
    working_output_input = full_pedal_output_input;
}

#define full_pedal_active_bias working_active_bias
#define full_pedal_active_input working_active_input
#define full_pedal_active_state working_active_state
#define full_pedal_active_influence working_active_influence
#define full_pedal_state_bias working_state_bias
#define full_pedal_state_input working_state_input
#define full_pedal_state_pole working_state_pole
#define full_pedal_state_active working_state_active
#define full_pedal_output_state working_output_state
#define full_pedal_output_active working_output_active
#define full_pedal_output_bias working_output_bias
#define full_pedal_output_input working_output_input
__attribute__((noinline, noclone)) static void affine_6x13(
    const float bias[6], const float input_coefficient[6],
    const float matrix[6][13], const float state[13], float input,
    float result[6])
{
    float value0 = bias[0] + input_coefficient[0] * input;
    float value1 = bias[1] + input_coefficient[1] * input;
    float value2 = bias[2] + input_coefficient[2] * input;
    float value3 = bias[3] + input_coefficient[3] * input;
    float value4 = bias[4] + input_coefficient[4] * input;
    float value5 = bias[5] + input_coefficient[5] * input;
    for (uint32_t column = 0U; column < 13U; ++column) {
        const float sample = state[column];
        value0 += matrix[0][column] * sample;
        value1 += matrix[1][column] * sample;
        value2 += matrix[2][column] * sample;
        value3 += matrix[3][column] * sample;
        value4 += matrix[4][column] * sample;
        value5 += matrix[5][column] * sample;
    }
    result[0] = value0;
    result[1] = value1;
    result[2] = value2;
    result[3] = value3;
    result[4] = value4;
    result[5] = value5;
}

static void linear_predictor(
    const pedal_state_t *pedal, float input,
    float linear_q[FULL_PEDAL_ACTIVE_COUNT])
{
    affine_6x13(
        full_pedal_active_bias, full_pedal_active_input,
        full_pedal_active_state, pedal->state, input, linear_q);
}

__attribute__((noinline, noclone)) static void project_active_6x6(
    const float linear_q[6], const float current[6], float projected[6])
{
    float value0 = linear_q[0], value1 = linear_q[1], value2 = linear_q[2];
    float value3 = linear_q[3], value4 = linear_q[4], value5 = linear_q[5];
    for (uint32_t column = 0U; column < 6U; ++column) {
        const float sample = current[column];
        value0 -= full_pedal_active_influence[0][column] * sample;
        value1 -= full_pedal_active_influence[1][column] * sample;
        value2 -= full_pedal_active_influence[2][column] * sample;
        value3 -= full_pedal_active_influence[3][column] * sample;
        value4 -= full_pedal_active_influence[4][column] * sample;
        value5 -= full_pedal_active_influence[5][column] * sample;
    }
    projected[0] = value0; projected[1] = value1; projected[2] = value2;
    projected[3] = value3; projected[4] = value4; projected[5] = value5;
}

__attribute__((always_inline)) static inline void solve_2x2(
    float matrix[4][4], float right[4], float result[4])
{
    const float inverse = 1.0F /
        (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]);
    result[0] = (right[0] * matrix[1][1] - matrix[0][1] * right[1]) * inverse;
    result[1] = (matrix[0][0] * right[1] - right[0] * matrix[1][0]) * inverse;
}

__attribute__((always_inline)) static inline void solve_3x3(
    float matrix[4][4], float right[4], float result[4])
{
    const float a = matrix[0][0], b = matrix[0][1], c = matrix[0][2];
    const float d = matrix[1][0], e = matrix[1][1], f = matrix[1][2];
    const float g = matrix[2][0], h = matrix[2][1], i = matrix[2][2];
    const float c00 = e * i - f * h;
    const float c01 = f * g - d * i;
    const float c02 = d * h - e * g;
    const float inverse = 1.0F / (a * c00 + b * c01 + c * c02);
    result[0] = inverse * (c00 * right[0] + (c * h - b * i) * right[1] +
                           (b * f - c * e) * right[2]);
    result[1] = inverse * (c01 * right[0] + (a * i - c * g) * right[1] +
                           (c * d - a * f) * right[2]);
    result[2] = inverse * (c02 * right[0] + (b * g - a * h) * right[1] +
                           (a * e - b * d) * right[2]);
}

static void solve_4x4(float matrix[4][4], float right[4], float result[4])
{
    float augmented[4][5];
    for (uint32_t row = 0U; row < 4U; ++row) {
        for (uint32_t column = 0U; column < 4U; ++column) {
            augmented[row][column] = matrix[row][column];
        }
        augmented[row][4] = right[row];
    }
    for (uint32_t pivot = 0U; pivot < 4U; ++pivot) {
        uint32_t best = pivot;
        float best_value = __builtin_fabsf(augmented[pivot][pivot]);
        for (uint32_t row = pivot + 1U; row < 4U; ++row) {
            const float candidate = __builtin_fabsf(augmented[row][pivot]);
            if (candidate > best_value) {
                best = row;
                best_value = candidate;
            }
        }
        if (best != pivot) {
            for (uint32_t column = pivot; column < 5U; ++column) {
                const float temporary = augmented[pivot][column];
                augmented[pivot][column] = augmented[best][column];
                augmented[best][column] = temporary;
            }
        }
        const float inverse = 1.0F / augmented[pivot][pivot];
        for (uint32_t row = pivot + 1U; row < 4U; ++row) {
            const float factor = augmented[row][pivot] * inverse;
            for (uint32_t column = pivot + 1U; column < 5U; ++column) {
                augmented[row][column] -= factor * augmented[pivot][column];
            }
        }
    }
    for (int32_t row = 3; row >= 0; --row) {
        float value = augmented[row][4];
        for (uint32_t column = (uint32_t)row + 1U; column < 4U; ++column) {
            value -= augmented[row][column] * result[column];
        }
        result[row] = value / augmented[row][row];
    }
}

#if !defined(FULL_PEDAL_CHEAP_ONLY)
static void solve_4x4_fixed(
    float matrix[4][4], float right[4], float result[4])
{
    const float inverse0 = 1.0F / matrix[0][0];
    const float factor10 = matrix[1][0] * inverse0;
    const float factor20 = matrix[2][0] * inverse0;
    const float factor30 = matrix[3][0] * inverse0;
    for (uint32_t column = 1U; column < 4U; ++column) {
        matrix[1][column] -= factor10 * matrix[0][column];
        matrix[2][column] -= factor20 * matrix[0][column];
        matrix[3][column] -= factor30 * matrix[0][column];
    }
    right[1] -= factor10 * right[0];
    right[2] -= factor20 * right[0];
    right[3] -= factor30 * right[0];

    const float inverse1 = 1.0F / matrix[1][1];
    const float factor21 = matrix[2][1] * inverse1;
    const float factor31 = matrix[3][1] * inverse1;
    matrix[2][2] -= factor21 * matrix[1][2];
    matrix[2][3] -= factor21 * matrix[1][3];
    matrix[3][2] -= factor31 * matrix[1][2];
    matrix[3][3] -= factor31 * matrix[1][3];
    right[2] -= factor21 * right[1];
    right[3] -= factor31 * right[1];

    const float inverse2 = 1.0F / matrix[2][2];
    const float factor32 = matrix[3][2] * inverse2;
    matrix[3][3] -= factor32 * matrix[2][3];
    right[3] -= factor32 * right[2];

    const float inverse3 = 1.0F / matrix[3][3];
    result[3] = right[3] * inverse3;
    result[2] = (right[2] - matrix[2][3] * result[3]) * inverse2;
    result[1] = (right[1] - matrix[1][2] * result[2] -
                 matrix[1][3] * result[3]) * inverse1;
    result[0] = (right[0] - matrix[0][1] * result[1] -
                 matrix[0][2] * result[2] - matrix[0][3] * result[3]) * inverse0;
}
#endif

__attribute__((always_inline)) static inline float active_residual(
    uint32_t row, const pedal_state_t *pedal, const float linear_q[6],
    const float current[6])
{
    float value = pedal->q[row] - linear_q[row];
    value += full_pedal_active_influence[row][0] * current[0];
    value += full_pedal_active_influence[row][1] * current[1];
    value += full_pedal_active_influence[row][2] * current[2];
    value += full_pedal_active_influence[row][3] * current[3];
    value += full_pedal_active_influence[row][4] * current[4];
    value += full_pedal_active_influence[row][5] * current[5];
    return value;
}

__attribute__((always_inline)) static inline void apply_correction(
    pedal_state_t *pedal, uint32_t index, float correction,
    float current[6], float first[6])
{
    const float old_q = pedal->q[index];
    pedal->q[index] -= correction;
    nonlinear_cache_update(pedal, index, old_q);
    nonlinear_from_cache(index, pedal, &current[index], &first[index]);
}

#if !defined(FULL_PEDAL_ROBUST_ONLY)
static void correct_first_three(
    pedal_state_t *pedal, const float linear_q[6],
    float current[6], float first[6])
{
    const float a = 1.0F + full_pedal_active_influence[0][0] * first[0];
    const float b = full_pedal_active_influence[0][1] * first[1];
    const float c = full_pedal_active_influence[0][2] * first[2];
    const float d = full_pedal_active_influence[1][0] * first[0];
    const float e = 1.0F + full_pedal_active_influence[1][1] * first[1];
    const float f = full_pedal_active_influence[1][2] * first[2];
    const float g = full_pedal_active_influence[2][0] * first[0];
    const float h = full_pedal_active_influence[2][1] * first[1];
    const float i = 1.0F + full_pedal_active_influence[2][2] * first[2];
    const float right0 = active_residual(0U, pedal, linear_q, current);
    const float right1 = active_residual(1U, pedal, linear_q, current);
    const float right2 = active_residual(2U, pedal, linear_q, current);
    const float c00 = e * i - f * h;
    const float c01 = f * g - d * i;
    const float c02 = d * h - e * g;
    const float inverse = 1.0F / (a * c00 + b * c01 + c * c02);
    const float correction0 = inverse *
        (c00 * right0 + (c * h - b * i) * right1 +
         (b * f - c * e) * right2);
    const float correction1 = inverse *
        (c01 * right0 + (a * i - c * g) * right1 +
         (c * d - a * f) * right2);
    const float correction2 = inverse *
        (c02 * right0 + (b * g - a * h) * right1 +
         (a * e - b * d) * right2);
    apply_correction(pedal, 0U, correction0, current, first);
    apply_correction(pedal, 1U, correction1, current, first);
    apply_correction(pedal, 2U, correction2, current, first);
}
#endif

#if !defined(FULL_PEDAL_CHEAP_ONLY)
static void correct_first_four(
    pedal_state_t *pedal, const float linear_q[6],
    float current[6], float first[6])
{
    float matrix[4][4];
    float right[4];
    float correction[4];
    for (uint32_t row = 0U; row < 4U; ++row) {
        right[row] = active_residual(row, pedal, linear_q, current);
        matrix[row][0] = (row == 0U ? 1.0F : 0.0F) +
                         full_pedal_active_influence[row][0] * first[0];
        matrix[row][1] = (row == 1U ? 1.0F : 0.0F) +
                         full_pedal_active_influence[row][1] * first[1];
        matrix[row][2] = (row == 2U ? 1.0F : 0.0F) +
                         full_pedal_active_influence[row][2] * first[2];
        matrix[row][3] = (row == 3U ? 1.0F : 0.0F) +
                         full_pedal_active_influence[row][3] * first[3];
    }
    solve_4x4_fixed(matrix, right, correction);
    apply_correction(pedal, 0U, correction[0], current, first);
    apply_correction(pedal, 1U, correction[1], current, first);
    apply_correction(pedal, 2U, correction[2], current, first);
    apply_correction(pedal, 3U, correction[3], current, first);
}
#endif

#if !defined(FULL_PEDAL_ROBUST_ONLY)
static void correct_q2(
    pedal_state_t *pedal, const float linear_q[6],
    float current[6], float first[6])
{
    const float correction = active_residual(3U, pedal, linear_q, current) /
        (1.0F + full_pedal_active_influence[3][3] * first[3]);
    apply_correction(pedal, 3U, correction, current, first);
}
#endif

static void correct_q1(
    pedal_state_t *pedal, const float linear_q[6],
    float current[6], float first[6])
{
    const float a = 1.0F + full_pedal_active_influence[4][4] * first[4];
    const float b = full_pedal_active_influence[4][5] * first[5];
    const float c = full_pedal_active_influence[5][4] * first[4];
    const float d = 1.0F + full_pedal_active_influence[5][5] * first[5];
    const float right0 = active_residual(4U, pedal, linear_q, current);
    const float right1 = active_residual(5U, pedal, linear_q, current);
    const float inverse = 1.0F / (a * d - b * c);
    const float correction0 = (right0 * d - b * right1) * inverse;
    const float correction1 = (a * right1 - right0 * c) * inverse;
    apply_correction(pedal, 4U, correction0, current, first);
    apply_correction(pedal, 5U, correction1, current, first);
}

static void update_state_and_output(
    pedal_state_t *pedal, float input,
    const float current[FULL_PEDAL_ACTIVE_COUNT],
    const float linear_q[FULL_PEDAL_ACTIVE_COUNT])
{
    float next[FULL_PEDAL_STATE_COUNT];
    for (uint32_t row = 0U; row < FULL_PEDAL_STATE_COUNT; ++row) {
        float even = full_pedal_state_bias[row] +
                     full_pedal_state_input[row] * input;
        float odd = full_pedal_state_pole[row] * pedal->state[row];
        even -= full_pedal_state_active[row][0] * current[0];
        odd -= full_pedal_state_active[row][1] * current[1];
        even -= full_pedal_state_active[row][2] * current[2];
        odd -= full_pedal_state_active[row][3] * current[3];
        even -= full_pedal_state_active[row][4] * current[4];
        odd -= full_pedal_state_active[row][5] * current[5];
        next[row] = even + odd;
    }
    float output_even = full_pedal_output_bias + full_pedal_output_input * input;
    float output_odd = 0.0F;
    for (uint32_t column = 0U; column < FULL_PEDAL_STATE_COUNT; column += 2U) {
        output_even += full_pedal_output_state[column] * pedal->state[column];
        if (column + 1U < FULL_PEDAL_STATE_COUNT) {
            output_odd += full_pedal_output_state[column + 1U] *
                          pedal->state[column + 1U];
        }
    }
    output_even -= full_pedal_output_active[0] * current[0];
    output_odd -= full_pedal_output_active[1] * current[1];
    output_even -= full_pedal_output_active[2] * current[2];
    output_odd -= full_pedal_output_active[3] * current[3];
    output_even -= full_pedal_output_active[4] * current[4];
    output_odd -= full_pedal_output_active[5] * current[5];
    for (uint32_t column = 0U; column < FULL_PEDAL_STATE_COUNT; ++column) {
        pedal->state[column] = next[column];
    }
    pedal->output = output_even + output_odd;
    float projected[6];
    project_active_6x6(linear_q, current, projected);
    for (uint32_t index = 0U; index < 6U; ++index) {
        const float old_q = pedal->q[index];
        pedal->q[index] = projected[index];
        nonlinear_cache_update(pedal, index, old_q);
    }
    ++pedal->sample_index;
    if ((pedal->sample_index & 63U) == 0U) {
        nonlinear_refresh_all(pedal);
    }
}

static void pedal_step(pedal_state_t *pedal, float input, uint32_t robust)
{
    float linear_q[FULL_PEDAL_ACTIVE_COUNT];
    float current[FULL_PEDAL_ACTIVE_COUNT];
    float first[FULL_PEDAL_ACTIVE_COUNT];
    linear_predictor(pedal, input, linear_q);
    nonlinear_all_cached(pedal, current, first);
#if defined(FULL_PEDAL_CHEAP_ONLY)
    (void)robust;
    correct_first_three(pedal, linear_q, current, first);
    correct_q2(pedal, linear_q, current, first);
#elif defined(FULL_PEDAL_ROBUST_ONLY)
    (void)robust;
    correct_first_four(pedal, linear_q, current, first);
#else
    if (robust != 0U) {
        correct_first_four(pedal, linear_q, current, first);
    } else {
        correct_first_three(pedal, linear_q, current, first);
        correct_q2(pedal, linear_q, current, first);
    }
#endif
    correct_q1(pedal, linear_q, current, first);
    correct_q1(pedal, linear_q, current, first);
    update_state_and_output(pedal, input, current, linear_q);
}

static void reset_pedal(pedal_state_t *pedal)
{
    for (uint32_t index = 0U; index < FULL_PEDAL_STATE_COUNT; ++index) {
        pedal->state[index] = full_pedal_dc_state[index];
    }
    for (uint32_t index = 0U; index < FULL_PEDAL_ACTIVE_COUNT; ++index) {
        pedal->q[index] = full_pedal_dc_active[index];
    }
    pedal->output = 0.0F;
    pedal->sample_index = 0U;
    pedal->fallback_count = 0U;
    nonlinear_refresh_all(pedal);
}

__attribute__((noinline)) static uint32_t profile_counter(void)
{
    __asm volatile("" ::: "memory");
    const uint32_t value = DWT->CYCCNT;
    __asm volatile("" ::: "memory");
    return value;
}

static stream_metrics_t measure_stream(uint32_t robust, uint32_t overhead)
{
    stream_metrics_t metrics = {0};
    uint64_t total = 0U;
    pedal_state_t pedal;
    for (uint32_t pass = 0U; pass < MEASUREMENT_PASSES; ++pass) {
        reset_pedal(&pedal);
        for (uint32_t sample = 0U; sample < FULL_PEDAL_STREAM_COUNT; ++sample) {
            const uint32_t begin = profile_counter();
            pedal_step(&pedal, full_pedal_stream_input[sample], robust);
            const uint32_t end = profile_counter();
            const uint32_t cycles = end - begin > overhead ? end - begin - overhead : 0U;
            total += cycles;
            if (cycles > metrics.maximum_cycles) {
                metrics.maximum_cycles = cycles;
            }
            const float_bits_t output = {.value = pedal.output};
            if ((output.bits & 0x7F800000U) == 0x7F800000U) {
                ++metrics.nonfinite_count;
            }
        }
        metrics.fallback_count += pedal.fallback_count;
    }
    metrics.average_cycles = (uint32_t)(
        total / (MEASUREMENT_PASSES * FULL_PEDAL_STREAM_COUNT));
    metrics.final_output_bits = ((float_bits_t){.value = pedal.output}).bits;
    s_sink = pedal.output;
    return metrics;
}

static uint32_t measure_component(uint32_t component, uint32_t overhead)
{
    pedal_state_t pedal;
    float linear_q[6], current[6], first[6];
    float matrix[4][4] = {{2.0F, 0.1F, 0.2F, 0.3F},
                          {0.2F, 2.1F, 0.1F, 0.2F},
                          {0.1F, 0.3F, 1.9F, 0.1F},
                          {0.3F, 0.1F, 0.2F, 2.2F}};
    float right[4] = {0.1F, -0.2F, 0.3F, -0.1F};
    float result[4] = {0.0F};
    reset_pedal(&pedal);
    linear_predictor(&pedal, full_pedal_stream_input[400], linear_q);
    nonlinear_all_cached(&pedal, current, first);
    const uint32_t begin = profile_counter();
    if (component == 0U) nonlinear_all_cached(&pedal, current, first);
    else if (component == 1U) solve_2x2(matrix, right, result);
    else if (component == 2U) solve_3x3(matrix, right, result);
    else if (component == 3U) solve_4x4(matrix, right, result);
    else if (component == 4U) linear_predictor(&pedal, 0.05F, linear_q);
    else if (component == 5U) update_state_and_output(&pedal, 0.05F, current, linear_q);
    else if (component == 6U) nonlinear_refresh_all(&pedal);
    else if (component == 7U) {
#if defined(FULL_PEDAL_ROBUST_ONLY)
        correct_first_four(&pedal, linear_q, current, first);
#else
        correct_first_three(&pedal, linear_q, current, first);
        correct_q2(&pedal, linear_q, current, first);
#endif
    }
    else if (component == 8U) correct_q1(&pedal, linear_q, current, first);
    else {
        correct_q1(&pedal, linear_q, current, first);
        correct_q1(&pedal, linear_q, current, first);
    }
    const uint32_t end = profile_counter();
    s_sink = component < 4U ? result[0] + current[0] : pedal.output + linear_q[0];
    return end - begin > overhead ? end - begin - overhead : 0U;
}

static void clock_init_170mhz(void)
{
    RCC->APB1ENR1 |= RCC_APB1ENR1_PWREN;
    (void)RCC->APB1ENR1;
    PWR->CR5 &= ~PWR_CR5_R1MODE;
    PWR->CR1 = (PWR->CR1 & ~PWR_CR1_VOS) | PWR_CR1_VOS_0;
    while ((PWR->SR2 & PWR_SR2_VOSF) != 0U) {}
    FLASH->ACR = FLASH_ACR_PRFTEN | FLASH_ACR_ICEN | FLASH_ACR_DCEN |
                 FLASH_ACR_LATENCY_4WS;
    while ((FLASH->ACR & FLASH_ACR_LATENCY) != FLASH_ACR_LATENCY_4WS) {}
    RCC->CR |= RCC_CR_HSION;
    while ((RCC->CR & RCC_CR_HSIRDY) == 0U) {}
    RCC->CR &= ~RCC_CR_PLLON;
    while ((RCC->CR & RCC_CR_PLLRDY) != 0U) {}
    RCC->PLLCFGR = RCC_PLLCFGR_PLLSRC_HSI |
                   (3UL << RCC_PLLCFGR_PLLM_Pos) |
                   (85UL << RCC_PLLCFGR_PLLN_Pos) | RCC_PLLCFGR_PLLREN;
    RCC->CR |= RCC_CR_PLLON;
    while ((RCC->CR & RCC_CR_PLLRDY) == 0U) {}
    RCC->CFGR = (RCC->CFGR & ~(RCC_CFGR_SW | RCC_CFGR_HPRE |
                               RCC_CFGR_PPRE1 | RCC_CFGR_PPRE2)) | RCC_CFGR_SW_PLL;
    while ((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_PLL) {}
    SystemCoreClock = 170000000U;
}

static void uart_init(void)
{
    RCC->AHB2ENR |= RCC_AHB2ENR_GPIOAEN;
    RCC->APB1ENR1 |= RCC_APB1ENR1_USART2EN;
    (void)RCC->AHB2ENR;
    GPIOA->MODER = (GPIOA->MODER & ~(3UL << 4U)) | (2UL << 4U);
    GPIOA->AFR[0] = (GPIOA->AFR[0] & ~(0xFUL << 8U)) | (7UL << 8U);
    GPIOA->OSPEEDR |= 3UL << 4U;
    USART2->BRR = (SystemCoreClock + 57600U) / 115200U;
    USART2->CR1 = USART_CR1_TE | USART_CR1_UE;
    while ((USART2->ISR & USART_ISR_TEACK) == 0U) {}
}

static void uart_character(char value)
{
    while ((USART2->ISR & USART_ISR_TXE) == 0U) {}
    USART2->TDR = (uint8_t)value;
}

static void uart_text(const char *text)
{
    while (*text != '\0') uart_character(*text++);
}

static void uart_unsigned(uint32_t value)
{
    char digits[10];
    uint32_t count = 0U;
    do {
        digits[count++] = (char)('0' + value % 10U);
        value /= 10U;
    } while (value != 0U);
    while (count != 0U) uart_character(digits[--count]);
}

static void report(const char *name, uint32_t value)
{
    uart_text(name);
    uart_character('=');
    uart_unsigned(value);
    uart_text("\r\n");
}

int main(void)
{
    working_coefficients_init();
    clock_init_170mhz();
    uart_init();
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0U;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    RCC->AHB1ENR |= RCC_AHB1ENR_CORDICEN;
    (void)RCC->AHB1ENR;
    RCC->AHB1RSTR |= RCC_AHB1RSTR_CORDICRST;
    RCC->AHB1RSTR &= ~RCC_AHB1RSTR_CORDICRST;
    CORDIC->CSR = CORDIC_CSR_FUNC_2 | CORDIC_CSR_FUNC_0 |
                  CORDIC_CSR_PRECISION_2 | CORDIC_CSR_PRECISION_1 |
                  CORDIC_CSR_SCALE_0 | CORDIC_CSR_NRES;
    const uint32_t begin = profile_counter();
    const uint32_t end = profile_counter();
    const uint32_t overhead = end - begin;
    const uint32_t components[10] = {
        measure_component(0U, overhead), measure_component(1U, overhead),
        measure_component(2U, overhead), measure_component(3U, overhead),
        measure_component(4U, overhead), measure_component(5U, overhead),
        measure_component(6U, overhead), measure_component(7U, overhead),
        measure_component(8U, overhead), measure_component(9U, overhead)};
    const stream_metrics_t cheap = measure_stream(0U, overhead);
    const stream_metrics_t robust = measure_stream(1U, overhead);
    for (;;) {
        uart_text("FULL_PEDAL_BENCHMARK_V1\r\n");
        report("core_hz", SystemCoreClock);
        report("sample_rate_hz", 384000U);
        report("budget_cycles", SystemCoreClock / 384000U);
        report("profile_overhead_cycles", overhead);
        report("nonlinear_six_cycles", components[0]);
        report("nonlinear_refresh_six_cycles", components[6]);
        report("solve_2x2_cycles", components[1]);
        report("solve_3x3_cycles", components[2]);
        report("solve_4x4_cycles", components[3]);
        report("linear_predictor_cycles", components[4]);
        report("state_output_update_cycles", components[5]);
        report("frontend_correction_cycles", components[7]);
        report("q1_correction_once_cycles", components[8]);
        report("q1_correction_twice_cycles", components[9]);
        report("cheap_average_cycles", cheap.average_cycles);
        report("cheap_maximum_cycles", cheap.maximum_cycles);
        report("cheap_nonfinite_count", cheap.nonfinite_count);
        report("cheap_final_output_bits", cheap.final_output_bits);
        report("cheap_fallback_count", cheap.fallback_count);
        report("robust_average_cycles", robust.average_cycles);
        report("robust_maximum_cycles", robust.maximum_cycles);
        report("robust_nonfinite_count", robust.nonfinite_count);
        report("robust_final_output_bits", robust.final_output_bits);
        report("robust_fallback_count", robust.fallback_count);
        uart_text("END\r\n");
        for (volatile uint32_t delay = 0U; delay < 20000000U; ++delay) {}
    }
}

#endif
