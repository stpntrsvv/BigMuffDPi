#include "audio_io.h"
#include "audio_engine.h"
#ifdef COMPOSED_PEDAL_RUNTIME
#include "composed_pedal.h"
#endif
#include "stm32g4xx.h"
#include "stm32g4xx_ll_adc.h"
#include "stm32g4xx_ll_dac.h"
#include "stm32g4xx_ll_opamp.h"
#include <stddef.h>
#include <stdint.h>

/* PA1=OPAMP1_VINP0 -> ADC1_IN13 internally, PA4=DAC1_OUT1. */
#define RATE_HZ 48000U
#define BUFFER_SAMPLES 64U
#define HALF_SAMPLES (BUFFER_SAMPLES / 2U)
#define MID_CODE 2048.0F
#define VOLTS_PER_CODE (3.3F / 4095.0F)
#define CODES_PER_VOLT (4095.0F / 3.3F)
#define BLOCK_BUDGET_CYCLES ((SystemCoreClock / RATE_HZ) * HALF_SAMPLES)

typedef union { float f; uint32_t u; } float_bits_t;

static uint16_t adc_buffer[BUFFER_SAMPLES];
static uint16_t dac_buffer[BUFFER_SAMPLES];
static float input_block[HALF_SAMPLES];
static float output_block[HALF_SAMPLES];
static bool initialized;
static volatile uint32_t block_count;
static volatile uint32_t maximum_block_cycles;
static volatile uint32_t deadline_overruns;
static volatile uint32_t dma_late_flags;
static volatile uint32_t dma_transfer_errors;
static volatile uint32_t dac_clip_count;
static volatile uint32_t nonfinite_count;
static volatile uint64_t total_block_cycles;
static volatile uint64_t adc_code_sum;
static volatile uint32_t adc_code_count;
static volatile uint16_t adc_code_min = 4095U;
static volatile uint16_t adc_code_max;
static uint32_t reported_blocks;

static void uart_init(void)
{
    RCC->APB1ENR1 |= RCC_APB1ENR1_USART2EN;
    GPIOA->MODER = (GPIOA->MODER & ~(3UL << 4U)) | (2UL << 4U);
    GPIOA->AFR[0] = (GPIOA->AFR[0] & ~(15UL << 8U)) | (7UL << 8U);
    USART2->BRR = (SystemCoreClock + 57600U) / 115200U;
    USART2->CR1 = USART_CR1_TE | USART_CR1_UE;
    while ((USART2->ISR & USART_ISR_TEACK) == 0U) {}
}

static void uart_char(char value)
{
    while ((USART2->ISR & USART_ISR_TXE) == 0U) {}
    USART2->TDR = (uint8_t)value;
}

static void uart_text(const char *text)
{
    while (*text != '\0') uart_char(*text++);
}

static void uart_u32(uint32_t value)
{
    char digits[10];
    uint32_t count = 0U;
    do { digits[count++] = (char)('0' + value % 10U); value /= 10U; } while (value != 0U);
    while (count != 0U) uart_char(digits[--count]);
}

static void uart_report(const char *name, uint32_t value)
{
    uart_text(name); uart_char('='); uart_u32(value); uart_text("\r\n");
}

static uint16_t to_dac(float sample)
{
    if ((((float_bits_t){.f = sample}.u >> 23U) & 255U) == 255U) {
        ++nonfinite_count;
        sample = 0.0F;
    }
    float code = MID_CODE + sample * CODES_PER_VOLT;
    if (code < 0.0F) { code = 0.0F; ++dac_clip_count; }
    if (code > 4095.0F) { code = 4095.0F; ++dac_clip_count; }
    return (uint16_t)(code + 0.5F);
}

static void process_half(size_t offset)
{
    const uint32_t started = DWT->CYCCNT;
    for (size_t i = 0U; i < HALF_SAMPLES; ++i) {
        const uint16_t adc_code = adc_buffer[offset + i];
        adc_code_sum += adc_code;
        ++adc_code_count;
        if (adc_code < adc_code_min) adc_code_min = adc_code;
        if (adc_code > adc_code_max) adc_code_max = adc_code;
#ifdef COMPOSED_RUNTIME_ZERO_INPUT
        input_block[i] = 0.0F;
#else
        input_block[i] = ((float)adc_code - MID_CODE) * VOLTS_PER_CODE;
#endif
    }
    audio_engine_process(input_block, output_block, HALF_SAMPLES);
    for (size_t i = 0U; i < HALF_SAMPLES; ++i)
        dac_buffer[offset + i] = to_dac(output_block[i]);
    const uint32_t elapsed = DWT->CYCCNT - started;
    total_block_cycles += elapsed;
    ++block_count;
    if (elapsed > maximum_block_cycles) maximum_block_cycles = elapsed;
    if (elapsed > BLOCK_BUDGET_CYCLES) ++deadline_overruns;
}

void DMA1_Channel1_IRQHandler(void)
{
    const uint32_t status = DMA1->ISR;
    if ((status & (DMA_ISR_HTIF1 | DMA_ISR_TCIF1)) == (DMA_ISR_HTIF1 | DMA_ISR_TCIF1))
        ++dma_late_flags;
    if ((status & DMA_ISR_HTIF1) != 0U) {
        DMA1->IFCR = DMA_IFCR_CHTIF1;
        process_half(0U);
    }
    if ((status & DMA_ISR_TCIF1) != 0U) {
        DMA1->IFCR = DMA_IFCR_CTCIF1;
        process_half(HALF_SAMPLES);
    }
    if ((status & DMA_ISR_TEIF1) != 0U) { DMA1->IFCR = DMA_IFCR_CTEIF1; ++dma_transfer_errors; }
    __DSB();
}

bool audio_io_init(void)
{
    RCC->AHB2ENR |= RCC_AHB2ENR_GPIOAEN | RCC_AHB2ENR_ADC12EN | RCC_AHB2ENR_DAC1EN;
    RCC->AHB1ENR |= RCC_AHB1ENR_DMA1EN | RCC_AHB1ENR_DMAMUX1EN;
    RCC->APB1ENR1 |= RCC_APB1ENR1_TIM6EN;
    (void)RCC->APB1ENR1;
    GPIOA->MODER |= (3UL << 2U) | (3UL << 8U);
    GPIOA->PUPDR &= ~((3UL << 2U) | (3UL << 8U));
    uart_init();
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0U;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    for (size_t i = 0U; i < BUFFER_SAMPLES; ++i) {
        adc_buffer[i] = (uint16_t)MID_CODE;
        dac_buffer[i] = (uint16_t)MID_CODE;
    }

    ADC12_COMMON->CCR = (ADC12_COMMON->CCR & ~ADC_CCR_CKMODE) | ADC_CCR_CKMODE_0;
    OPAMP1->CSR = LL_OPAMP_MODE_FOLLOWER | LL_OPAMP_INPUT_NONINVERT_IO0 |
                  LL_OPAMP_INTERNAL_OUPUT_ENABLED | OPAMP_CSR_HIGHSPEEDEN |
                  OPAMP_CSR_OPAMPxEN;
    for (volatile uint32_t i = 0U; i < SystemCoreClock / 100000U; ++i) __NOP();
    ADC1->CR &= ~ADC_CR_DEEPPWD;
    ADC1->CR |= ADC_CR_ADVREGEN;
    for (volatile uint32_t i = 0U; i < SystemCoreClock / 50000U; ++i) __NOP();
    ADC1->CR |= ADC_CR_ADCAL;
    while ((ADC1->CR & ADC_CR_ADCAL) != 0U) {}
    ADC1->SMPR2 = (ADC1->SMPR2 & ~ADC_SMPR2_SMP13) | (4UL << ADC_SMPR2_SMP13_Pos);
    ADC1->SQR1 = 13UL << ADC_SQR1_SQ1_Pos;
    ADC1->CFGR = ADC_CFGR_DMAEN | ADC_CFGR_DMACFG |
                 (LL_ADC_REG_TRIG_EXT_TIM6_TRGO & (ADC_CFGR_EXTSEL | ADC_CFGR_EXTEN));
    ADC1->ISR = ADC_ISR_ADRDY;
    ADC1->CR |= ADC_CR_ADEN;
    while ((ADC1->ISR & ADC_ISR_ADRDY) == 0U) {}

    DMA1_Channel1->CPAR = (uint32_t)(uintptr_t)&ADC1->DR;
    DMA1_Channel1->CMAR = (uint32_t)(uintptr_t)adc_buffer;
    DMA1_Channel1->CNDTR = BUFFER_SAMPLES;
    DMA1_Channel1->CCR = DMA_CCR_MINC | DMA_CCR_CIRC | DMA_CCR_HTIE | DMA_CCR_TCIE |
                         DMA_CCR_TEIE | DMA_CCR_PL_1 | DMA_CCR_MSIZE_0 | DMA_CCR_PSIZE_0;
    DMAMUX1_Channel0->CCR = 5U;
    DMA1_Channel2->CPAR = (uint32_t)(uintptr_t)&DAC1->DHR12R1;
    DMA1_Channel2->CMAR = (uint32_t)(uintptr_t)dac_buffer;
    DMA1_Channel2->CNDTR = BUFFER_SAMPLES;
    DMA1_Channel2->CCR = DMA_CCR_MINC | DMA_CCR_CIRC | DMA_CCR_DIR | DMA_CCR_PL_1 |
                         DMA_CCR_MSIZE_0 | DMA_CCR_PSIZE_0;
    DMAMUX1_Channel1->CCR = 6U;

    DAC1->MCR &= ~DAC_MCR_MODE1;
    DAC1->DHR12R1 = (uint32_t)MID_CODE;
    DAC1->CR = LL_DAC_TRIG_EXT_TIM6_TRGO | DAC_CR_TEN1 | DAC_CR_DMAEN1 | DAC_CR_EN1;
    TIM6->PSC = 0U;
    TIM6->ARR = SystemCoreClock / RATE_HZ - 1U;
    TIM6->CR2 = TIM_CR2_MMS_1;
    TIM6->EGR = TIM_EGR_UG;
    NVIC_SetPriority(DMA1_Channel1_IRQn, 1U);
    NVIC_EnableIRQ(DMA1_Channel1_IRQn);
    initialized = true;
    return true;
}

bool audio_io_start(void)
{
    if (!initialized) return false;
    DMA1->IFCR = DMA_IFCR_CGIF1 | DMA_IFCR_CGIF2;
    DMA1_Channel2->CCR |= DMA_CCR_EN;
    DMA1_Channel1->CCR |= DMA_CCR_EN;
    ADC1->CR |= ADC_CR_ADSTART;
    TIM6->CR1 |= TIM_CR1_CEN;
    return true;
}

void audio_io_poll(void)
{
    if ((uint32_t)(block_count - reported_blocks) < 3000U) return;
    __disable_irq();
    const uint32_t blocks = block_count;
    const uint32_t maximum = maximum_block_cycles;
    const uint32_t overruns = deadline_overruns;
    const uint32_t late = dma_late_flags;
    const uint32_t errors = dma_transfer_errors;
    const uint32_t clips = dac_clip_count;
    const uint32_t nonfinite = nonfinite_count;
    const uint64_t total = total_block_cycles;
    const uint64_t adc_sum = adc_code_sum;
    const uint32_t adc_count = adc_code_count;
    const uint32_t adc_min = adc_code_min;
    const uint32_t adc_max = adc_code_max;
    __enable_irq();
    reported_blocks = blocks;
    uart_text("COMPOSED_AUDIO_RUNTIME_V1\r\n");
    uart_report("blocks", blocks);
    uart_report("block_samples", HALF_SAMPLES);
    uart_report("block_budget_cycles", BLOCK_BUDGET_CYCLES);
    uart_report("average_block_cycles", blocks != 0U ? (uint32_t)(total / blocks) : 0U);
    uart_report("maximum_block_cycles", maximum);
    uart_report("deadline_overruns", overruns);
    uart_report("dma_late_flags", late);
    uart_report("dma_transfer_errors", errors);
    uart_report("dac_clip_count", clips);
    uart_report("nonfinite_count", nonfinite);
    uart_report("adc_code_min", adc_min);
    uart_report("adc_code_max", adc_max);
    uart_report("adc_code_average", adc_count != 0U ? (uint32_t)(adc_sum / adc_count) : 0U);
#ifdef COMPOSED_PEDAL_RUNTIME
    uart_report("dsp_fault_stage", composed_pedal_fault_stage());
    uart_report("dsp_hold_count", composed_pedal_hold_count());
#endif
    uart_text("END\r\n");
}
