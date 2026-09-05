#include "board.h"

#include "stm32g4xx.h"

#define STATUS_LED_PIN 5U

static volatile uint32_t s_milliseconds;

static void clock_init_170mhz(void)
{
    RCC->APB1ENR1 |= RCC_APB1ENR1_PWREN;
    PWR->CR5 &= ~PWR_CR5_R1MODE;
    PWR->CR1 = (PWR->CR1 & ~PWR_CR1_VOS) | PWR_CR1_VOS_0;
    while ((PWR->SR2 & PWR_SR2_VOSF) != 0U) {}
    FLASH->ACR = FLASH_ACR_PRFTEN | FLASH_ACR_ICEN | FLASH_ACR_DCEN | FLASH_ACR_LATENCY_4WS;
    RCC->CR |= RCC_CR_HSION;
    while ((RCC->CR & RCC_CR_HSIRDY) == 0U) {}
    RCC->CR &= ~RCC_CR_PLLON;
    while ((RCC->CR & RCC_CR_PLLRDY) != 0U) {}
    RCC->PLLCFGR = RCC_PLLCFGR_PLLSRC_HSI | (3UL << RCC_PLLCFGR_PLLM_Pos) |
                   (85UL << RCC_PLLCFGR_PLLN_Pos) | RCC_PLLCFGR_PLLREN;
    RCC->CR |= RCC_CR_PLLON;
    while ((RCC->CR & RCC_CR_PLLRDY) == 0U) {}
    RCC->CFGR = (RCC->CFGR & ~(RCC_CFGR_SW | RCC_CFGR_HPRE | RCC_CFGR_PPRE1 |
                               RCC_CFGR_PPRE2)) | RCC_CFGR_SW_PLL;
    while ((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_PLL) {}
    SystemCoreClock = 170000000U;
}

void SysTick_Handler(void)
{
    ++s_milliseconds;
}

void board_init(void)
{
    clock_init_170mhz();

    RCC->AHB2ENR |= RCC_AHB2ENR_GPIOAEN;
    (void)RCC->AHB2ENR;

    GPIOA->MODER = (GPIOA->MODER & ~(3UL << (STATUS_LED_PIN * 2U))) |
                   (1UL << (STATUS_LED_PIN * 2U));
    GPIOA->OTYPER &= ~(1UL << STATUS_LED_PIN);
    GPIOA->OSPEEDR &= ~(3UL << (STATUS_LED_PIN * 2U));
    GPIOA->PUPDR &= ~(3UL << (STATUS_LED_PIN * 2U));

    board_status_led_set(false);
    (void)SysTick_Config(SystemCoreClock / 1000U);
}

uint32_t board_millis(void)
{
    return s_milliseconds;
}

void board_status_led_set(bool enabled)
{
    GPIOA->BSRR = enabled ? (1UL << STATUS_LED_PIN)
                          : (1UL << (STATUS_LED_PIN + 16U));
}

void board_status_led_toggle(void)
{
    GPIOA->ODR ^= (1UL << STATUS_LED_PIN);
}
