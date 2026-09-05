#ifdef PORT_MULTIRATE_BENCHMARK

#include <stdint.h>

#include "stm32g4xx.h"
#include "port_multirate_fixture.h"

#define MEASUREMENT_PASSES 8U
#define FORWARD_SCALE 25.8649e-3F
#define DIODE_SCALE (1.9F * 25.8649e-3F)
#define FORWARD_IS 10.025e-15F
#define REVERSE_IS 12.0e-15F
#define DIODE_IS_TWICE 4.0e-9F
#define CACHE_DELTA_LIMIT 0.25F
#define ADAPTIVE_RESIDUAL_LIMIT 0.001F

typedef union { float value; uint32_t bits; } float_bits_t;

typedef struct {
    float fast_state[PORT_FAST_STATE_COUNT];
    float fast_q[PORT_FAST_ACTIVE_COUNT];
    float slow_state[PORT_SLOW_STATE_COUNT];
    float slow_q[PORT_SLOW_ACTIVE_COUNT];
    float port_offset;
    float port_voltage;
    float output;
#ifdef PORT_MULTIRATE_CACHED
    float fast_nonlinear_value[PORT_FAST_ACTIVE_COUNT];
    float fast_nonlinear_cosh[PORT_FAST_ACTIVE_COUNT];
    float slow_nonlinear_value[PORT_SLOW_ACTIVE_COUNT];
    uint32_t fast_step_index;
    uint32_t slow_step_index;
    uint32_t fallback_count;
#endif
#ifdef PORT_MULTIRATE_ADAPTIVE
    float fast_saved_lu[4][4];
    uint32_t fast_lu_valid;
    uint32_t predicted_step_count;
    uint32_t full_step_count;
    uint32_t pre_rejected_step_count;
#endif
} port_state_t;

typedef struct {
    uint32_t average_cycles;
    uint32_t maximum_cycles;
    uint32_t nonfinite_count;
    uint32_t final_output_bits;
    uint32_t fallback_count;
    uint32_t predicted_step_count;
    uint32_t full_step_count;
    uint32_t pre_rejected_step_count;
} stream_metrics_t;

static volatile float s_sink;
#ifdef PORT_MULTIRATE_PROFILE
static uint32_t s_fast_phase_cycles[7];
#define FAST_MARK(index, previous) do { \
    __asm volatile("" ::: "memory"); \
    const uint32_t now = DWT->CYCCNT; \
    __asm volatile("" ::: "memory"); \
    s_fast_phase_cycles[index] = now - (previous); \
    (previous) = now; \
} while (0)
#endif

static float scale_power_of_two(float value, int32_t delta)
{
    float_bits_t representation = {.value = value};
    const int32_t exponent =
        (int32_t)((representation.bits >> 23U) & 0xFFU) + delta;
    if (exponent <= 0) return 0.0F;
    if (exponent >= 255) {
        representation.bits = 0x7F800000U;
        return representation.value;
    }
    representation.bits = (representation.bits & 0x807FFFFFU) |
                          ((uint32_t)exponent << 23U);
    return representation.value;
}

static void cordic_sinh_cosh(float argument, float *sinh_value, float *cosh_value)
{
    const float inverse_ln2 = 1.4426950408889634F;
    const float ln2 = 0.6931471805599453F;
    const float q31_per_unit = 1073741824.0F;
    const float unit_per_q31 = 9.313225746154785e-10F;
    if (argument > 80.0F) argument = 80.0F;
    else if (argument < -80.0F) argument = -80.0F;
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

#ifndef PORT_MULTIRATE_CACHED
static void exponential_terms(float q, float saturation, float *current, float *first)
{
    float sinh_value, cosh_value;
    cordic_sinh_cosh(q / FORWARD_SCALE, &sinh_value, &cosh_value);
    const float exponential = sinh_value + cosh_value;
    *current = saturation * (exponential - 1.0F);
    *first = saturation * exponential / FORWARD_SCALE;
}

static void diode_terms(float q, float *current, float *first)
{
    float sinh_value, cosh_value;
    cordic_sinh_cosh(q / DIODE_SCALE, &sinh_value, &cosh_value);
    *current = DIODE_IS_TWICE * sinh_value;
    *first = DIODE_IS_TWICE * cosh_value / DIODE_SCALE;
}

static void fast_terms(
    const float q[PORT_FAST_ACTIVE_COUNT],
    float current[PORT_FAST_ACTIVE_COUNT], float first[PORT_FAST_ACTIVE_COUNT])
{
    exponential_terms(q[0], FORWARD_IS, &current[0], &first[0]);
    exponential_terms(q[1], FORWARD_IS, &current[1], &first[1]);
    diode_terms(q[2], &current[2], &first[2]);
    diode_terms(q[3], &current[3], &first[3]);
}

static void slow_terms(
    const float q[PORT_SLOW_ACTIVE_COUNT],
    float current[PORT_SLOW_ACTIVE_COUNT], float first[PORT_SLOW_ACTIVE_COUNT])
{
    exponential_terms(q[0], FORWARD_IS, &current[0], &first[0]);
    exponential_terms(q[1], REVERSE_IS, &current[1], &first[1]);
}
#endif

#ifdef PORT_MULTIRATE_CACHED
static void fast_refresh_one(port_state_t *state, uint32_t index)
{
    if (index >= 2U) {
        cordic_sinh_cosh(
            state->fast_q[index] / DIODE_SCALE,
            &state->fast_nonlinear_value[index],
            &state->fast_nonlinear_cosh[index]);
    } else {
        float ignored;
        cordic_sinh_cosh(
            state->fast_q[index] / FORWARD_SCALE, &ignored,
            &state->fast_nonlinear_value[index]);
        state->fast_nonlinear_value[index] += ignored;
    }
}

static void slow_refresh_one(port_state_t *state, uint32_t index)
{
    float ignored;
    cordic_sinh_cosh(
        state->slow_q[index] / FORWARD_SCALE, &ignored,
        &state->slow_nonlinear_value[index]);
    state->slow_nonlinear_value[index] += ignored;
}

static void fast_refresh_all(port_state_t *state)
{
    fast_refresh_one(state, 0U);
#ifndef PORT_MULTIRATE_Q3_CUBIC
    fast_refresh_one(state, 1U);
#endif
    fast_refresh_one(state, 2U);
    fast_refresh_one(state, 3U);
}

static void slow_refresh_all(port_state_t *state)
{
    for (uint32_t index = 0U; index < 2U; ++index) slow_refresh_one(state, index);
}

__attribute__((always_inline)) static inline void fast_terms_cached(
    const port_state_t *state, float current[4], float first[4])
{
    current[0] = FORWARD_IS * (state->fast_nonlinear_value[0] - 1.0F);
#ifdef PORT_MULTIRATE_Q3_CUBIC
    const float delta = state->fast_q[1] - port_fast_dc_active[1];
    const float z = delta / FORWARD_SCALE;
    const float square = z * z;
    current[1] = port_fast_dc_current[1] + port_fast_dc_first[1] * delta *
        (1.0F + 0.5F * z + square * (1.0F / 6.0F));
#else
    current[1] = FORWARD_IS * (state->fast_nonlinear_value[1] - 1.0F);
#endif
    first[0] = FORWARD_IS * state->fast_nonlinear_value[0] / FORWARD_SCALE;
#ifdef PORT_MULTIRATE_Q3_CUBIC
    first[1] = port_fast_dc_first[1] * (1.0F + z + 0.5F * square);
#else
    first[1] = FORWARD_IS * state->fast_nonlinear_value[1] / FORWARD_SCALE;
#endif
    for (uint32_t index = 2U; index < 4U; ++index) {
        current[index] = DIODE_IS_TWICE * state->fast_nonlinear_value[index];
        first[index] = DIODE_IS_TWICE * state->fast_nonlinear_cosh[index] /
                       DIODE_SCALE;
    }
}

__attribute__((always_inline)) static inline void slow_terms_cached(
    const port_state_t *state, float current[2], float first[2])
{
    current[0] = FORWARD_IS * (state->slow_nonlinear_value[0] - 1.0F);
    current[1] = REVERSE_IS * (state->slow_nonlinear_value[1] - 1.0F);
    first[0] = FORWARD_IS * state->slow_nonlinear_value[0] / FORWARD_SCALE;
    first[1] = REVERSE_IS * state->slow_nonlinear_value[1] / FORWARD_SCALE;
}

__attribute__((always_inline)) static inline void fast_cache_update_exponential(
    port_state_t *state, uint32_t index, float old_q)
{
#ifdef PORT_MULTIRATE_Q3_CUBIC
    if (index == 1U) return;
#endif
    const float delta = (state->fast_q[index] - old_q) / FORWARD_SCALE;
    if (__builtin_fabsf(delta) > CACHE_DELTA_LIMIT) {
        fast_refresh_one(state, index);
        ++state->fallback_count;
        return;
    }
    state->fast_nonlinear_value[index] *=
        1.0F + delta * (1.0F + delta * (0.5F + delta * (1.0F / 6.0F)));
}

__attribute__((always_inline)) static inline void fast_cache_update_diode(
    port_state_t *state, uint32_t index, float old_q)
{
    const float delta = (state->fast_q[index] - old_q) / DIODE_SCALE;
    if (__builtin_fabsf(delta) > CACHE_DELTA_LIMIT) {
        fast_refresh_one(state, index);
        ++state->fallback_count;
        return;
    }
    const float square = delta * delta;
    const float sinh_delta = delta * (1.0F + square * (1.0F / 6.0F));
    const float cosh_delta = 1.0F + square * 0.5F;
    const float old_sinh = state->fast_nonlinear_value[index];
    const float old_cosh = state->fast_nonlinear_cosh[index];
    state->fast_nonlinear_value[index] =
        old_sinh * cosh_delta + old_cosh * sinh_delta;
    state->fast_nonlinear_cosh[index] =
        old_cosh * cosh_delta + old_sinh * sinh_delta;
}

__attribute__((always_inline)) static inline void fast_cache_update_all(
    port_state_t *state, const float old_q[4])
{
    fast_cache_update_exponential(state, 0U, old_q[0]);
    fast_cache_update_exponential(state, 1U, old_q[1]);
    fast_cache_update_diode(state, 2U, old_q[2]);
    fast_cache_update_diode(state, 3U, old_q[3]);
}

static void slow_cache_update(port_state_t *state, uint32_t index, float old_q)
{
    const float delta = (state->slow_q[index] - old_q) / FORWARD_SCALE;
    if (__builtin_fabsf(delta) > CACHE_DELTA_LIMIT) {
        slow_refresh_one(state, index);
        ++state->fallback_count;
        return;
    }
    state->slow_nonlinear_value[index] *=
        1.0F + delta * (1.0F + delta * (0.5F + delta * (1.0F / 6.0F)));
}
#endif

static void solve_2x2(float matrix[2][2], const float right[2], float result[2])
{
    const float inverse = 1.0F /
        (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]);
    result[0] = (right[0] * matrix[1][1] - matrix[0][1] * right[1]) * inverse;
    result[1] = (matrix[0][0] * right[1] - right[0] * matrix[1][0]) * inverse;
}

#ifndef PORT_MULTIRATE_3X3_PLUS_1
static void solve_4x4_fixed(
    float matrix[4][4], float right[4], float result[4])
{
    for (uint32_t pivot = 0U; pivot < 3U; ++pivot) {
        const float inverse = 1.0F / matrix[pivot][pivot];
        for (uint32_t row = pivot + 1U; row < 4U; ++row) {
            const float factor = matrix[row][pivot] * inverse;
            matrix[row][pivot] = factor;
            for (uint32_t column = pivot + 1U; column < 4U; ++column) {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
            right[row] -= factor * right[pivot];
        }
    }
    for (int32_t row = 3; row >= 0; --row) {
        float value = right[row];
        for (uint32_t column = (uint32_t)row + 1U; column < 4U; ++column) {
            value -= matrix[row][column] * result[column];
        }
        result[row] = value / matrix[row][row];
    }
}

#ifdef PORT_MULTIRATE_ADAPTIVE
__attribute__((always_inline)) static inline void solve_saved_4x4(
    const float matrix[4][4], const float right[4], float result[4])
{
    float work[4] = {right[0], right[1], right[2], right[3]};
    for (uint32_t row = 1U; row < 4U; ++row) {
        for (uint32_t column = 0U; column < row; ++column) {
            work[row] -= matrix[row][column] * work[column];
        }
    }
    for (int32_t row = 3; row >= 0; --row) {
        float value = work[row];
        for (uint32_t column = (uint32_t)row + 1U; column < 4U; ++column) {
            value -= matrix[row][column] * result[column];
        }
        result[row] = value / matrix[row][row];
    }
}

static uint32_t fast_candidate_terms(
    const port_state_t *state, const float candidate[4],
    float nonlinear_value[4], float nonlinear_cosh[4], float current[4])
{
    for (uint32_t index = 0U; index < 2U; ++index) {
        const float delta = (candidate[index] - state->fast_q[index]) / FORWARD_SCALE;
        if (__builtin_fabsf(delta) > CACHE_DELTA_LIMIT) return 0U;
        nonlinear_value[index] = state->fast_nonlinear_value[index] *
            (1.0F + delta * (1.0F + delta * (0.5F + delta * (1.0F / 6.0F))));
        nonlinear_cosh[index] = 0.0F;
        current[index] = FORWARD_IS * (nonlinear_value[index] - 1.0F);
    }
    for (uint32_t index = 2U; index < 4U; ++index) {
        const float delta = (candidate[index] - state->fast_q[index]) / DIODE_SCALE;
        if (__builtin_fabsf(delta) > CACHE_DELTA_LIMIT) return 0U;
        const float square = delta * delta;
        const float sinh_delta = delta * (1.0F + square * (1.0F / 6.0F));
        const float cosh_delta = 1.0F + square * 0.5F;
        nonlinear_value[index] = state->fast_nonlinear_value[index] * cosh_delta
            + state->fast_nonlinear_cosh[index] * sinh_delta;
        nonlinear_cosh[index] = state->fast_nonlinear_cosh[index] * cosh_delta
            + state->fast_nonlinear_value[index] * sinh_delta;
        current[index] = DIODE_IS_TWICE * nonlinear_value[index];
    }
    return 1U;
}
#endif
#endif

#ifdef PORT_MULTIRATE_3X3_PLUS_1
__attribute__((always_inline)) static inline void solve_3x3_fixed(
    float matrix[4][4], float right[4], float result[4])
{
    const float a = matrix[0][0], b = matrix[0][1], c = matrix[0][2];
    const float d = matrix[1][0], e = matrix[1][1], f = matrix[1][2];
    const float g = matrix[2][0], h = matrix[2][1], i = matrix[2][2];
    const float c00 = e*i - f*h;
    const float c01 = f*g - d*i;
    const float c02 = d*h - e*g;
    const float inverse = 1.0F / (a*c00 + b*c01 + c*c02);
    result[0] = inverse * (c00*right[0] + (c*h - b*i)*right[1] +
                           (b*f - c*e)*right[2]);
    result[1] = inverse * (c01*right[0] + (a*i - c*g)*right[1] +
                           (c*d - a*f)*right[2]);
    result[2] = inverse * (c02*right[0] + (b*g - a*h)*right[1] +
                           (a*e - b*d)*right[2]);
}
#endif

static float fast_step(port_state_t *state, float input)
{
#ifdef PORT_MULTIRATE_PROFILE
    uint32_t phase_begin = DWT->CYCCNT;
#endif
    float linear_q[4];
#ifdef PORT_MULTIRATE_GENERATED_LINEAR
    port_fast_predict_generated(
        state->fast_state, input, state->port_offset, linear_q);
#else
    for (uint32_t row = 0U; row < 4U; ++row) {
        float value = port_fast_active_bias[row]
            + port_fast_active_input[row] * input
            + port_fast_active_port[row] * state->port_offset;
        for (uint32_t column = 0U; column < 9U; ++column) {
            value += port_fast_active_state[row][column] * state->fast_state[column];
        }
        linear_q[row] = value;
    }
#endif
#ifdef PORT_MULTIRATE_PROFILE
    FAST_MARK(0U, phase_begin);
#endif
    float current[4], first[4], matrix[4][4], right[4], correction[4];
#ifdef PORT_MULTIRATE_CACHED
    fast_terms_cached(state, current, first);
#else
    fast_terms(state->fast_q, current, first);
#endif
#ifdef PORT_MULTIRATE_PROFILE
    FAST_MARK(1U, phase_begin);
#endif
    for (uint32_t row = 0U; row < 4U; ++row) {
        right[row] = state->fast_q[row] - linear_q[row];
        for (uint32_t column = 0U; column < 4U; ++column) {
            right[row] += port_fast_active_influence[row][column] * current[column];
            matrix[row][column] = (row == column ? 1.0F : 0.0F)
                + port_fast_active_influence[row][column] * first[column];
        }
    }
#ifdef PORT_MULTIRATE_PROFILE
    FAST_MARK(2U, phase_begin);
#endif
#ifdef PORT_MULTIRATE_3X3_PLUS_1
    solve_3x3_fixed(matrix, right, correction);
#elif defined(PORT_MULTIRATE_ADAPTIVE)
    uint32_t prediction_accepted = 0U;
    float candidate_value[4], candidate_cosh[4], candidate_current[4];
    if (state->fast_lu_valid != 0U) {
        solve_saved_4x4(state->fast_saved_lu, right, correction);
        float second[4] = {
            first[0] / FORWARD_SCALE,
            first[1] / FORWARD_SCALE,
            current[2] / (DIODE_SCALE * DIODE_SCALE),
            current[3] / (DIODE_SCALE * DIODE_SCALE)
        };
        uint32_t quadratic_allows = 1U;
        static const float scale[4] = {
            FORWARD_SCALE, FORWARD_SCALE, DIODE_SCALE, DIODE_SCALE
        };
        for (uint32_t row = 0U; row < 4U; ++row) {
            float estimate = 0.0F;
            for (uint32_t column = 0U; column < 4U; ++column) {
                estimate += port_fast_active_influence[row][column] *
                    second[column] * correction[column] * correction[column];
            }
            if (0.5F * __builtin_fabsf(estimate) >
                    ADAPTIVE_RESIDUAL_LIMIT * scale[row]) {
                quadratic_allows = 0U;
            }
        }
        const float candidate[4] = {
            state->fast_q[0] - correction[0], state->fast_q[1] - correction[1],
            state->fast_q[2] - correction[2], state->fast_q[3] - correction[3]
        };
        if (quadratic_allows != 0U && fast_candidate_terms(
                state, candidate, candidate_value, candidate_cosh,
                candidate_current) != 0U) {
            prediction_accepted = 1U;
            for (uint32_t row = 0U; row < 4U; ++row) {
                float residual = candidate[row] - linear_q[row];
                for (uint32_t column = 0U; column < 4U; ++column) {
                    residual += port_fast_active_influence[row][column] *
                                candidate_current[column];
                }
                if (__builtin_fabsf(residual) >
                        ADAPTIVE_RESIDUAL_LIMIT * scale[row]) {
                    prediction_accepted = 0U;
                }
            }
        } else if (quadratic_allows == 0U) {
            ++state->pre_rejected_step_count;
        }
    }
    if (prediction_accepted != 0U) {
        ++state->predicted_step_count;
    } else {
        solve_4x4_fixed(matrix, right, correction);
        for (uint32_t row = 0U; row < 4U; ++row) {
            for (uint32_t column = 0U; column < 4U; ++column) {
                state->fast_saved_lu[row][column] = matrix[row][column];
            }
        }
        state->fast_lu_valid = 1U;
        ++state->full_step_count;
    }
#else
    solve_4x4_fixed(matrix, right, correction);
#endif
#ifdef PORT_MULTIRATE_PROFILE
    FAST_MARK(3U, phase_begin);
#endif
#ifdef PORT_MULTIRATE_CACHED
    const float old_q[4] = {
        state->fast_q[0], state->fast_q[1], state->fast_q[2], state->fast_q[3]
    };
#endif
#ifdef PORT_MULTIRATE_3X3_PLUS_1
    state->fast_q[0] -= correction[0];
    state->fast_q[1] -= correction[1];
    state->fast_q[2] -= correction[2];
#ifdef PORT_MULTIRATE_CACHED
    fast_cache_update_exponential(state, 0U, old_q[0]);
    fast_cache_update_exponential(state, 1U, old_q[1]);
    fast_cache_update_diode(state, 2U, old_q[2]);
    fast_terms_cached(state, current, first);
#else
    fast_terms(state->fast_q, current, first);
#endif
    float scalar_residual = state->fast_q[3] - linear_q[3];
    scalar_residual += port_fast_active_influence[3][0] * current[0];
    scalar_residual += port_fast_active_influence[3][1] * current[1];
    scalar_residual += port_fast_active_influence[3][2] * current[2];
    scalar_residual += port_fast_active_influence[3][3] * current[3];
    state->fast_q[3] -= scalar_residual /
        (1.0F + port_fast_active_influence[3][3] * first[3]);
#ifdef PORT_MULTIRATE_CACHED
    fast_cache_update_diode(state, 3U, old_q[3]);
#endif
#else
    state->fast_q[0] -= correction[0];
    state->fast_q[1] -= correction[1];
    state->fast_q[2] -= correction[2];
    state->fast_q[3] -= correction[3];
#if defined(PORT_MULTIRATE_ADAPTIVE) && defined(PORT_MULTIRATE_CACHED)
    if (prediction_accepted != 0U) {
        for (uint32_t index = 0U; index < 4U; ++index) {
            state->fast_nonlinear_value[index] = candidate_value[index];
            state->fast_nonlinear_cosh[index] = candidate_cosh[index];
        }
    } else {
        fast_cache_update_all(state, old_q);
    }
#elif defined(PORT_MULTIRATE_CACHED)
    fast_cache_update_all(state, old_q);
#endif
#endif
#ifdef PORT_MULTIRATE_CACHED
    fast_terms_cached(state, current, first);
#else
    fast_terms(state->fast_q, current, first);
#endif
#ifdef PORT_MULTIRATE_PROFILE
    FAST_MARK(4U, phase_begin);
#endif
    float next[9];
#ifdef PORT_MULTIRATE_GENERATED_LINEAR
    port_fast_state_generated(
        state->fast_state, input, state->port_offset, current, next);
#else
    for (uint32_t row = 0U; row < 9U; ++row) {
        float value = port_fast_state_bias[row]
            + port_fast_state_input[row] * input
            + port_fast_state_port[row] * state->port_offset;
        for (uint32_t column = 0U; column < 9U; ++column) {
            value += port_fast_state_transition[row][column] * state->fast_state[column];
        }
        for (uint32_t column = 0U; column < 4U; ++column) {
            value -= port_fast_state_active[row][column] * current[column];
        }
        next[row] = value;
    }
#endif
#ifdef PORT_MULTIRATE_PROFILE
    FAST_MARK(5U, phase_begin);
#endif
#ifdef PORT_MULTIRATE_GENERATED_LINEAR
    const float projected_old_q[4] = {
        state->fast_q[0], state->fast_q[1], state->fast_q[2], state->fast_q[3]
    };
    const float port_voltage = port_fast_project_generated(
        state->fast_state, input, state->port_offset, current, linear_q,
        state->fast_q);
#ifdef PORT_MULTIRATE_CACHED
    fast_cache_update_all(state, projected_old_q);
#endif
#else
    float port_voltage = port_fast_port_bias
        + port_fast_port_input * input
        + port_fast_port_port * state->port_offset;
    for (uint32_t column = 0U; column < 9U; ++column) {
        port_voltage += port_fast_port_state[column] * state->fast_state[column];
    }
    for (uint32_t column = 0U; column < 4U; ++column) {
        port_voltage -= port_fast_port_active[column] * current[column];
        const float old_q = state->fast_q[column];
        state->fast_q[column] = linear_q[column];
        for (uint32_t active = 0U; active < 4U; ++active) {
            state->fast_q[column] -=
                port_fast_active_influence[column][active] * current[active];
        }
#ifdef PORT_MULTIRATE_CACHED
        if (column < 2U) fast_cache_update_exponential(state, column, old_q);
        else fast_cache_update_diode(state, column, old_q);
#else
        (void)old_q;
#endif
    }
#endif
    for (uint32_t row = 0U; row < 9U; ++row) state->fast_state[row] = next[row];
    state->port_voltage = port_voltage;
#ifdef PORT_MULTIRATE_CACHED
    ++state->fast_step_index;
    if ((state->fast_step_index & 63U) == 0U) fast_refresh_all(state);
#endif
#ifdef PORT_MULTIRATE_PROFILE
    FAST_MARK(6U, phase_begin);
#endif
    return port_voltage;
}

static void slow_step(port_state_t *state, float port_voltage)
{
    float linear_q[2];
    for (uint32_t row = 0U; row < 2U; ++row) {
        float value = port_slow_active_bias[row]
            + port_slow_active_port[row] * port_voltage;
        for (uint32_t column = 0U; column < 4U; ++column) {
            value += port_slow_active_state[row][column] * state->slow_state[column];
        }
        linear_q[row] = value;
    }
    float current[2], first[2];
    for (uint32_t pass = 0U; pass < 2U; ++pass) {
        float matrix[2][2], right[2], correction[2];
#ifdef PORT_MULTIRATE_CACHED
        slow_terms_cached(state, current, first);
#else
        slow_terms(state->slow_q, current, first);
#endif
        for (uint32_t row = 0U; row < 2U; ++row) {
            right[row] = state->slow_q[row] - linear_q[row];
            for (uint32_t column = 0U; column < 2U; ++column) {
                right[row] += port_slow_active_influence[row][column] * current[column];
                matrix[row][column] = (row == column ? 1.0F : 0.0F)
                    + port_slow_active_influence[row][column] * first[column];
            }
        }
        solve_2x2(matrix, right, correction);
        for (uint32_t index = 0U; index < 2U; ++index) {
#ifdef PORT_MULTIRATE_CACHED
            const float old_q = state->slow_q[index];
#endif
            state->slow_q[index] -= correction[index];
#ifdef PORT_MULTIRATE_CACHED
            slow_cache_update(state, index, old_q);
#endif
        }
    }
#ifdef PORT_MULTIRATE_CACHED
    slow_terms_cached(state, current, first);
#else
    slow_terms(state->slow_q, current, first);
#endif
    float next[4];
    for (uint32_t row = 0U; row < 4U; ++row) {
        float value = port_slow_state_bias[row]
            + port_slow_state_port[row] * port_voltage;
        for (uint32_t column = 0U; column < 4U; ++column) {
            value += port_slow_state_transition[row][column] * state->slow_state[column];
        }
        for (uint32_t column = 0U; column < 2U; ++column) {
            value -= port_slow_state_active[row][column] * current[column];
        }
        next[row] = value;
    }
    float output = port_slow_output_bias + port_slow_output_port * port_voltage;
    float port_current = port_slow_current_bias + port_slow_current_port * port_voltage;
    for (uint32_t column = 0U; column < 4U; ++column) {
        output += port_slow_output_state[column] * state->slow_state[column];
        port_current += port_slow_current_state[column] * state->slow_state[column];
    }
    for (uint32_t column = 0U; column < 2U; ++column) {
        output -= port_slow_output_active[column] * current[column];
        port_current += port_slow_current_active[column] * current[column];
        const float old_q = state->slow_q[column];
        state->slow_q[column] = linear_q[column];
        for (uint32_t active = 0U; active < 2U; ++active) {
            state->slow_q[column] -=
                port_slow_active_influence[column][active] * current[active];
        }
#ifdef PORT_MULTIRATE_CACHED
        slow_cache_update(state, column, old_q);
#else
        (void)old_q;
#endif
    }
    for (uint32_t row = 0U; row < 4U; ++row) state->slow_state[row] = next[row];
    state->port_offset = port_current - port_fixed_conductance * port_voltage;
    state->output = output;
#ifdef PORT_MULTIRATE_CACHED
    ++state->slow_step_index;
    if ((state->slow_step_index & 63U) == 0U) slow_refresh_all(state);
#endif
}

static void base_step(port_state_t *state, float previous_input, float input)
{
    const float delta = input - previous_input;
    (void)fast_step(state, previous_input + 0.25F * delta);
    slow_step(state, fast_step(state, previous_input + 0.50F * delta));
    (void)fast_step(state, previous_input + 0.75F * delta);
    slow_step(state, fast_step(state, input));
}

static void reset_state(port_state_t *state)
{
    for (uint32_t i = 0U; i < 9U; ++i) state->fast_state[i] = port_fast_dc_state[i];
    for (uint32_t i = 0U; i < 4U; ++i) state->fast_q[i] = port_fast_dc_active[i];
    for (uint32_t i = 0U; i < 4U; ++i) state->slow_state[i] = port_slow_dc_state[i];
    for (uint32_t i = 0U; i < 2U; ++i) state->slow_q[i] = port_slow_dc_active[i];
    state->port_offset = port_dc_offset;
    state->port_voltage = 0.0F;
    state->output = 0.0F;
#ifdef PORT_MULTIRATE_CACHED
    state->fast_step_index = 0U;
    state->slow_step_index = 0U;
    state->fallback_count = 0U;
    fast_refresh_all(state);
    slow_refresh_all(state);
#endif
#ifdef PORT_MULTIRATE_ADAPTIVE
    state->fast_lu_valid = 0U;
    state->predicted_step_count = 0U;
    state->full_step_count = 0U;
    state->pre_rejected_step_count = 0U;
#endif
}

__attribute__((noinline)) static uint32_t counter(void)
{
    __asm volatile("" ::: "memory");
    const uint32_t value = DWT->CYCCNT;
    __asm volatile("" ::: "memory");
    return value;
}

static stream_metrics_t measure_stream(uint32_t overhead)
{
    stream_metrics_t metrics = {0};
    uint64_t total = 0U;
    port_state_t state;
    for (uint32_t pass = 0U; pass < MEASUREMENT_PASSES; ++pass) {
        reset_state(&state);
        float previous = port_stream_input[0];
        for (uint32_t sample = 1U; sample < PORT_STREAM_COUNT; ++sample) {
            const uint32_t begin = counter();
            base_step(&state, previous, port_stream_input[sample]);
            const uint32_t end = counter();
            const uint32_t cycles = end - begin > overhead ? end - begin - overhead : 0U;
            total += cycles;
            if (cycles > metrics.maximum_cycles) metrics.maximum_cycles = cycles;
            if ((((float_bits_t){.value = state.output}).bits & 0x7F800000U) == 0x7F800000U) {
                ++metrics.nonfinite_count;
            }
            previous = port_stream_input[sample];
        }
        metrics.final_output_bits = ((float_bits_t){.value = state.output}).bits;
#ifdef PORT_MULTIRATE_CACHED
        metrics.fallback_count += state.fallback_count;
#endif
#ifdef PORT_MULTIRATE_ADAPTIVE
        metrics.predicted_step_count += state.predicted_step_count;
        metrics.full_step_count += state.full_step_count;
        metrics.pre_rejected_step_count += state.pre_rejected_step_count;
#endif
    }
    metrics.average_cycles = (uint32_t)(
        total / (MEASUREMENT_PASSES * (PORT_STREAM_COUNT - 1U)));
    s_sink = state.output;
    return metrics;
}

static uint32_t measure_component(uint32_t component, uint32_t overhead)
{
    port_state_t state;
    reset_state(&state);
    if (component != 0U) {
        (void)fast_step(&state, port_stream_input[1]);
    }
    const uint32_t begin = counter();
    if (component == 0U) {
        (void)fast_step(&state, port_stream_input[1]);
    } else {
        slow_step(&state, state.port_voltage);
    }
    const uint32_t end = counter();
    s_sink = state.output + state.port_voltage;
    return end - begin > overhead ? end - begin - overhead : 0U;
}

static void clock_init_170mhz(void)
{
    RCC->APB1ENR1 |= RCC_APB1ENR1_PWREN; (void)RCC->APB1ENR1;
    PWR->CR5 &= ~PWR_CR5_R1MODE;
    PWR->CR1 = (PWR->CR1 & ~PWR_CR1_VOS) | PWR_CR1_VOS_0;
    while ((PWR->SR2 & PWR_SR2_VOSF) != 0U) {}
    FLASH->ACR = FLASH_ACR_PRFTEN | FLASH_ACR_ICEN | FLASH_ACR_DCEN | FLASH_ACR_LATENCY_4WS;
    RCC->CR |= RCC_CR_HSION; while ((RCC->CR & RCC_CR_HSIRDY) == 0U) {}
    RCC->CR &= ~RCC_CR_PLLON; while ((RCC->CR & RCC_CR_PLLRDY) != 0U) {}
    RCC->PLLCFGR = RCC_PLLCFGR_PLLSRC_HSI | (3UL << RCC_PLLCFGR_PLLM_Pos)
        | (85UL << RCC_PLLCFGR_PLLN_Pos) | RCC_PLLCFGR_PLLREN;
    RCC->CR |= RCC_CR_PLLON; while ((RCC->CR & RCC_CR_PLLRDY) == 0U) {}
    RCC->CFGR = (RCC->CFGR & ~(RCC_CFGR_SW | RCC_CFGR_HPRE | RCC_CFGR_PPRE1 | RCC_CFGR_PPRE2))
        | RCC_CFGR_SW_PLL;
    while ((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_PLL) {}
    SystemCoreClock = 170000000U;
}

static void uart_init(void)
{
    RCC->AHB2ENR |= RCC_AHB2ENR_GPIOAEN; RCC->APB1ENR1 |= RCC_APB1ENR1_USART2EN;
    GPIOA->MODER = (GPIOA->MODER & ~(3UL << 4U)) | (2UL << 4U);
    GPIOA->AFR[0] = (GPIOA->AFR[0] & ~(0xFUL << 8U)) | (7UL << 8U);
    USART2->BRR = (SystemCoreClock + 57600U) / 115200U;
    USART2->CR1 = USART_CR1_TE | USART_CR1_UE;
    while ((USART2->ISR & USART_ISR_TEACK) == 0U) {}
}

static void uart_character(char value)
{ while ((USART2->ISR & USART_ISR_TXE) == 0U) {} USART2->TDR = (uint8_t)value; }
static void uart_text(const char *text) { while (*text != '\0') uart_character(*text++); }
static void uart_unsigned(uint32_t value)
{
    char digits[10]; uint32_t count = 0U;
    do { digits[count++] = (char)('0' + value % 10U); value /= 10U; } while (value != 0U);
    while (count != 0U) uart_character(digits[--count]);
}
static void report(const char *name, uint32_t value)
{ uart_text(name); uart_character('='); uart_unsigned(value); uart_text("\r\n"); }

int main(void)
{
    clock_init_170mhz(); uart_init();
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0U; DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    RCC->AHB1ENR |= RCC_AHB1ENR_CORDICEN; (void)RCC->AHB1ENR;
    RCC->AHB1RSTR |= RCC_AHB1RSTR_CORDICRST; RCC->AHB1RSTR &= ~RCC_AHB1RSTR_CORDICRST;
    CORDIC->CSR = CORDIC_CSR_FUNC_2 | CORDIC_CSR_FUNC_0 |
                  CORDIC_CSR_PRECISION_2 | CORDIC_CSR_PRECISION_1 |
                  CORDIC_CSR_SCALE_0 | CORDIC_CSR_NRES;
    const uint32_t begin = counter(); const uint32_t end = counter();
    const uint32_t overhead = end - begin;
    const uint32_t fast_cycles = measure_component(0U, overhead);
    const uint32_t slow_cycles = measure_component(1U, overhead);
    const stream_metrics_t metrics = measure_stream(overhead);
    for (;;) {
        uart_text("PORT_MULTIRATE_BENCHMARK_V1\r\n");
        report("core_hz", SystemCoreClock);
        report("base_rate_hz", 48000U);
        report("budget_cycles", SystemCoreClock / 48000U);
        report("fast_step_cycles", fast_cycles);
        report("slow_step_cycles", slow_cycles);
        report("average_cycles", metrics.average_cycles);
        report("maximum_cycles", metrics.maximum_cycles);
        report("nonfinite_count", metrics.nonfinite_count);
        report("final_output_bits", metrics.final_output_bits);
        report("fallback_count", metrics.fallback_count);
        report("predicted_step_count", metrics.predicted_step_count);
        report("full_step_count", metrics.full_step_count);
        report("pre_rejected_step_count", metrics.pre_rejected_step_count);
#ifdef PORT_MULTIRATE_PROFILE
        report("fast_predict_cycles", s_fast_phase_cycles[0]);
        report("fast_terms_cycles", s_fast_phase_cycles[1]);
        report("fast_form_cycles", s_fast_phase_cycles[2]);
        report("fast_solve_cycles", s_fast_phase_cycles[3]);
        report("fast_correct_cycles", s_fast_phase_cycles[4]);
        report("fast_state_cycles", s_fast_phase_cycles[5]);
        report("fast_project_cycles", s_fast_phase_cycles[6]);
#endif
        uart_text("END\r\n");
        for (volatile uint32_t delay = 0U; delay < 20000000U; ++delay) {}
    }
}

#endif
