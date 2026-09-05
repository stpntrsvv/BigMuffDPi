#ifdef FMAC_BENCHMARK

#include <stdint.h>
#include "stm32g4xx.h"

#define TAP_COUNT 13U
#define PASSES 256U
#define DMA_COUNT 128U

static const int16_t s_coeff[TAP_COUNT] = {
    1638, -983, 655, 327, -164, 1311, -524, 262, 819, -328, 197, 98, -49
};
static volatile int16_t s_input[TAP_COUNT] = {
    4096, -2048, 1024, 512, -256, 3072, -1536, 768, 384, -192, 96, -48, 24
};
static volatile int32_t s_sink;
static int16_t s_dma_input[DMA_COUNT];
static int16_t s_dma_output[DMA_COUNT];
static uint32_t s_dma_status;
static uint32_t s_fmac_status;
static uint32_t s_fmac_control;
static uint32_t s_dma_remaining;

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
    RCC->CFGR = (RCC->CFGR & ~(RCC_CFGR_SW | RCC_CFGR_HPRE | RCC_CFGR_PPRE1 | RCC_CFGR_PPRE2)) | RCC_CFGR_SW_PLL;
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
static void putc_uart(char v) { while ((USART2->ISR & USART_ISR_TXE) == 0U) {} USART2->TDR = (uint8_t)v; }
static void puts_uart(const char *s) { while (*s != '\0') putc_uart(*s++); }
static void putu(uint32_t v) { char d[10]; uint32_t n=0; do { d[n++]=(char)('0'+v%10U); v/=10U; } while(v); while(n) putc_uart(d[--n]); }
static void report(const char *n, uint32_t v) { puts_uart(n); putc_uart('='); putu(v); puts_uart("\r\n"); }
static uint32_t counter(void) { __asm volatile("" ::: "memory"); uint32_t v=DWT->CYCCNT; __asm volatile("" ::: "memory"); return v; }

static void fmac_reset(void)
{
    FMAC->CR = FMAC_CR_RESET;
    FMAC->CR = 0U;
    FMAC->X1BUFCFG = (0U << FMAC_X1BUFCFG_X1_BASE_Pos) | (16U << FMAC_X1BUFCFG_X1_BUF_SIZE_Pos);
    FMAC->X2BUFCFG = (32U << FMAC_X2BUFCFG_X2_BASE_Pos) | (TAP_COUNT << FMAC_X2BUFCFG_X2_BUF_SIZE_Pos);
    FMAC->YBUFCFG = (64U << FMAC_YBUFCFG_Y_BASE_Pos) | (16U << FMAC_YBUFCFG_Y_BUF_SIZE_Pos);
}

static void fmac_load_coeff(void)
{
    FMAC->PARAM = (TAP_COUNT << FMAC_PARAM_P_Pos) | FMAC_PARAM_FUNC_1 | FMAC_PARAM_START;
    for (uint32_t i=0; i<TAP_COUNT; ++i) FMAC->WDATA = (uint16_t)s_coeff[i];
    while ((FMAC->PARAM & FMAC_PARAM_START) != 0U) {}
}

static int16_t fmac_dot(void)
{
    FMAC->PARAM = (TAP_COUNT << FMAC_PARAM_P_Pos) | FMAC_PARAM_FUNC_3 | FMAC_PARAM_START;
    int16_t last = 0;
    for (uint32_t i=0; i<TAP_COUNT; ++i) {
        while ((FMAC->SR & FMAC_SR_X1FULL) != 0U) {}
        FMAC->WDATA = (uint16_t)s_input[i];
        while ((FMAC->SR & FMAC_SR_YEMPTY) == 0U) last = (int16_t)FMAC->RDATA;
    }
    while ((FMAC->SR & FMAC_SR_YEMPTY) != 0U) {}
    last = (int16_t)FMAC->RDATA;
    FMAC->PARAM &= ~FMAC_PARAM_START;
    return last;
}

static int32_t cpu_dot(void)
{
    int32_t sum=0;
    for (uint32_t i=0; i<TAP_COUNT; ++i) sum += (int32_t)s_coeff[i] * s_input[i];
    return sum;
}

static void fmac_stream_start(void)
{
    fmac_reset(); fmac_load_coeff();
    FMAC->PARAM = ((TAP_COUNT - 1U) << FMAC_PARAM_P_Pos) | FMAC_PARAM_FUNC_0 | FMAC_PARAM_START;
    for (uint32_t i=0; i<TAP_COUNT-1U; ++i) FMAC->WDATA = (uint16_t)s_input[i];
    while ((FMAC->PARAM & FMAC_PARAM_START) != 0U) {}
    FMAC->PARAM = (TAP_COUNT << FMAC_PARAM_P_Pos) | FMAC_PARAM_FUNC_3 | FMAC_PARAM_START;
}

static int16_t fmac_stream_one(int16_t input)
{
    while ((FMAC->SR & FMAC_SR_X1FULL) != 0U) {}
    FMAC->WDATA = (uint16_t)input;
    while ((FMAC->SR & FMAC_SR_YEMPTY) != 0U) {}
    return (int16_t)FMAC->RDATA;
}

static void fmac_dma_measure(
    uint32_t *setup_cycles, uint32_t *total_cycles, uint32_t *free_iterations)
{
    for (uint32_t i=0; i<DMA_COUNT; ++i) s_dma_input[i] = (int16_t)(i-64);
    fmac_reset(); fmac_load_coeff();
    FMAC->PARAM = ((TAP_COUNT - 1U) << FMAC_PARAM_P_Pos) | FMAC_PARAM_FUNC_0 | FMAC_PARAM_START;
    for (uint32_t i=0; i<TAP_COUNT-1U; ++i) FMAC->WDATA = (uint16_t)s_input[i];
    while ((FMAC->PARAM & FMAC_PARAM_START) != 0U) {}
    RCC->AHB1ENR |= RCC_AHB1ENR_DMA1EN | RCC_AHB1ENR_DMAMUX1EN; (void)RCC->AHB1ENR;
    DMA1_Channel1->CCR = 0U; DMA1_Channel2->CCR = 0U;
    DMAMUX1_Channel0->CCR = 111U;
    DMAMUX1_Channel1->CCR = 110U;
    DMA1->IFCR = DMA_IFCR_CGIF1 | DMA_IFCR_CGIF2;
    DMA1_Channel1->CPAR = (uint32_t)&FMAC->WDATA;
    DMA1_Channel1->CMAR = (uint32_t)s_dma_input;
    DMA1_Channel1->CNDTR = DMA_COUNT;
    DMA1_Channel2->CPAR = (uint32_t)&FMAC->RDATA;
    DMA1_Channel2->CMAR = (uint32_t)s_dma_output;
    DMA1_Channel2->CNDTR = DMA_COUNT;
    const uint32_t begin = counter();
    FMAC->CR = FMAC_CR_DMAWEN | FMAC_CR_DMAREN;
    DMA1_Channel2->CCR = DMA_CCR_MINC | DMA_CCR_MSIZE_0 | DMA_CCR_PSIZE_1 | DMA_CCR_EN;
    DMA1_Channel1->CCR = DMA_CCR_DIR | DMA_CCR_MINC | DMA_CCR_MSIZE_0 | DMA_CCR_PSIZE_1 | DMA_CCR_EN;
    FMAC->PARAM = (TAP_COUNT << FMAC_PARAM_P_Pos) | FMAC_PARAM_FUNC_3 | FMAC_PARAM_START;
    const uint32_t configured = counter();
    uint32_t iterations = 0U;
    while (((DMA1->ISR & DMA_ISR_TCIF2) == 0U) &&
           ((DWT->CYCCNT - begin) < 1000000U)) {
        ++iterations;
        __asm volatile("nop");
    }
    const uint32_t end = counter();
    *setup_cycles = configured - begin;
    *total_cycles = end - begin;
    *free_iterations = iterations;
    s_dma_status = DMA1->ISR;
    s_fmac_status = FMAC->SR;
    s_fmac_control = FMAC->CR;
    s_dma_remaining = (DMA1_Channel1->CNDTR << 16U) | DMA1_Channel2->CNDTR;
    s_sink = s_dma_output[DMA_COUNT-1U];
}

int main(void)
{
    clock_init_170mhz(); uart_init();
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk; DWT->CYCCNT=0; DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    RCC->AHB1ENR |= RCC_AHB1ENR_FMACEN; (void)RCC->AHB1ENR;
    fmac_reset(); fmac_load_coeff();
    uint32_t b=counter(), e=counter(), overhead=e-b;
    uint32_t cpu=0, reuse=0, reload=0, stream=0;
    for (uint32_t p=0; p<PASSES; ++p) { b=counter(); s_sink=cpu_dot(); e=counter(); cpu += e-b-overhead; }
    for (uint32_t p=0; p<PASSES; ++p) { b=counter(); s_sink=fmac_dot(); e=counter(); reuse += e-b-overhead; }
    for (uint32_t p=0; p<PASSES; ++p) { b=counter(); fmac_load_coeff(); s_sink=fmac_dot(); e=counter(); reload += e-b-overhead; }
    fmac_stream_start();
    for (uint32_t p=0; p<PASSES; ++p) { b=counter(); s_sink=fmac_stream_one((int16_t)(p-128)); e=counter(); stream += e-b-overhead; }
    uint32_t dma_setup, dma_total, dma_free;
    fmac_dma_measure(&dma_setup, &dma_total, &dma_free);
    for (;;) {
        puts_uart("FMAC_BENCHMARK_V1\r\n"); report("core_hz",SystemCoreClock); report("tap_count",TAP_COUNT);
        report("cpu_dot_cycles",cpu/PASSES); report("fmac_reuse_cycles",reuse/PASSES); report("fmac_reload_cycles",reload/PASSES);
        report("fmac_stream_cycles",stream/PASSES);
        report("dma_count",DMA_COUNT); report("fmac_dma_setup_cycles",dma_setup);
        report("fmac_dma_total_cycles",dma_total); report("fmac_dma_free_iterations",dma_free);
        report("dma_status",s_dma_status); report("fmac_status",s_fmac_status);
        report("fmac_control",s_fmac_control); report("dma_remaining",s_dma_remaining);
        report("sink",(uint32_t)s_sink); puts_uart("END\r\n"); for(volatile uint32_t i=0;i<20000000U;++i){}
    }
}
#endif
