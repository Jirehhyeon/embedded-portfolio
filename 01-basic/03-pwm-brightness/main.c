/*
 * PWM LED Brightness Control
 * 
 * Hardware: ATmega328P
 * PWM Outputs: PD6 (OC0A), PD5 (OC0B), PB1 (OC1A)
 * ADC Input: PC0 (ADC0) - Potentiometer
 * Buttons: PD2, PD3
 * 
 * This program demonstrates:
 * - PWM generation using Timer0 and Timer1
 * - RGB LED color mixing
 * - Smooth fading effects
 * - ADC-based brightness control
 * - Multiple lighting effects
 * 
 * Author: [Your Name]
 * Date: 2024-01-20
 */

#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>
#include <stdbool.h>
#include <math.h>

// PWM 출력 핀 정의
#define RED_PIN     PD6  // OC0A
#define GREEN_PIN   PD5  // OC0B  
#define BLUE_PIN    PB1  // OC1A

// 버튼 핀 정의
#define BUTTON1_PIN PD2
#define BUTTON2_PIN PD3

// ADC 채널
#define POT_CHANNEL 0

// 효과 모드
typedef enum {
    MODE_MANUAL,      // 가변저항으로 제어
    MODE_FADE,        // 페이드 인/아웃
    MODE_RAINBOW,     // 무지개 효과
    MODE_BREATHING,   // 호흡 효과
    MODE_STROBE,      // 스트로브 효과
    MODE_FIRE,        // 불꽃 효과
    MODE_COUNT
} effect_mode_t;

// RGB 색상 구조체
typedef struct {
    uint8_t r;
    uint8_t g;
    uint8_t b;
} rgb_color_t;

// 전역 변수
volatile effect_mode_t current_mode = MODE_MANUAL;
volatile bool button1_pressed = false;
volatile bool button2_pressed = false;
volatile uint16_t adc_value = 0;

// 미리 정의된 색상
const rgb_color_t COLOR_RED     = {255, 0, 0};
const rgb_color_t COLOR_GREEN   = {0, 255, 0};
const rgb_color_t COLOR_BLUE    = {0, 0, 255};
const rgb_color_t COLOR_YELLOW  = {255, 255, 0};
const rgb_color_t COLOR_CYAN    = {0, 255, 255};
const rgb_color_t COLOR_MAGENTA = {255, 0, 255};
const rgb_color_t COLOR_WHITE   = {255, 255, 255};
const rgb_color_t COLOR_ORANGE  = {255, 127, 0};
const rgb_color_t COLOR_PURPLE  = {128, 0, 128};

// 감마 보정 테이블 (γ = 2.2)
const uint8_t gamma_table[256] PROGMEM = {
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,
    1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,
    2,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  5,  5,  5,
    5,  6,  6,  6,  6,  7,  7,  7,  7,  8,  8,  8,  9,  9,  9, 10,
   10, 10, 11, 11, 11, 12, 12, 13, 13, 13, 14, 14, 15, 15, 16, 16,
   17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 24, 24, 25,
   25, 26, 27, 27, 28, 29, 29, 30, 31, 32, 32, 33, 34, 35, 35, 36,
   37, 38, 39, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 50,
   51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 68,
   69, 70, 72, 73, 74, 75, 77, 78, 79, 81, 82, 83, 85, 86, 87, 89,
   90, 92, 93, 95, 96, 98, 99,101,102,104,105,107,109,110,112,114,
  115,117,119,120,122,124,126,127,129,131,133,135,137,138,140,142,
  144,146,148,150,152,154,156,158,160,162,164,167,169,171,173,175,
  177,180,182,184,186,189,191,193,196,198,200,203,205,208,210,213,
  215,218,220,223,225,228,231,233,236,239,241,244,247,249,252,255
};

// 함수 프로토타입
void system_init(void);
void pwm_init(void);
void adc_init(void);
void gpio_init(void);
void timer2_init(void);
void set_rgb(uint8_t r, uint8_t g, uint8_t b);
void set_rgb_gamma(uint8_t r, uint8_t g, uint8_t b);
uint8_t apply_gamma(uint8_t value);
void fade_to_color(rgb_color_t from, rgb_color_t to, uint16_t duration_ms);
void effect_manual(void);
void effect_fade(void);
void effect_rainbow(void);
void effect_breathing(void);
void effect_strobe(void);
void effect_fire(void);
rgb_color_t hsv_to_rgb(uint16_t h, uint8_t s, uint8_t v);

// Timer2 오버플로우 인터럽트 (주기적 작업용)
ISR(TIMER2_OVF_vect) {
    static uint8_t counter = 0;
    
    // 버튼 읽기 (디바운싱 포함)
    static uint8_t btn1_state = 0;
    static uint8_t btn2_state = 0;
    
    btn1_state = (btn1_state << 1) | !(PIND & (1 << BUTTON1_PIN));
    btn2_state = (btn2_state << 1) | !(PIND & (1 << BUTTON2_PIN));
    
    if(btn1_state == 0xFF) button1_pressed = true;
    if(btn2_state == 0xFF) button2_pressed = true;
    
    // ADC 읽기 (10ms마다)
    if(++counter >= 10) {
        counter = 0;
        ADCSRA |= (1 << ADSC);  // ADC 변환 시작
    }
}

// ADC 변환 완료 인터럽트
ISR(ADC_vect) {
    adc_value = ADC;
}

int main(void) {
    system_init();
    
    while(1) {
        // 버튼1: 모드 변경
        if(button1_pressed) {
            button1_pressed = false;
            current_mode = (current_mode + 1) % MODE_COUNT;
            
            // 모드 변경 시 LED 깜빡임
            set_rgb(255, 255, 255);
            _delay_ms(100);
            set_rgb(0, 0, 0);
            _delay_ms(100);
        }
        
        // 현재 모드에 따른 효과 실행
        switch(current_mode) {
            case MODE_MANUAL:
                effect_manual();
                break;
                
            case MODE_FADE:
                effect_fade();
                break;
                
            case MODE_RAINBOW:
                effect_rainbow();
                break;
                
            case MODE_BREATHING:
                effect_breathing();
                break;
                
            case MODE_STROBE:
                effect_strobe();
                break;
                
            case MODE_FIRE:
                effect_fire();
                break;
                
            default:
                break;
        }
    }
    
    return 0;
}

void system_init(void) {
    pwm_init();
    adc_init();
    gpio_init();
    timer2_init();
    sei();  // 전역 인터럽트 활성화
}

void pwm_init(void) {
    // Timer0 설정 (8-bit Fast PWM)
    TCCR0A = (1 << WGM01) | (1 << WGM00);  // Fast PWM mode
    TCCR0A |= (1 << COM0A1) | (1 << COM0B1); // Non-inverting mode
    TCCR0B = (1 << CS01);  // Prescaler = 8 (7.8kHz PWM)
    
    // Timer1 설정 (8-bit Fast PWM)
    TCCR1A = (1 << WGM10) | (1 << COM1A1);  // 8-bit Fast PWM, non-inverting
    TCCR1B = (1 << WGM12) | (1 << CS11);    // Prescaler = 8
    
    // PWM 핀을 출력으로 설정
    DDRD |= (1 << RED_PIN) | (1 << GREEN_PIN);
    DDRB |= (1 << BLUE_PIN);
}

void adc_init(void) {
    // ADC 활성화, 인터럽트 활성화, 프리스케일러 = 128
    ADCSRA = (1 << ADEN) | (1 << ADIE) | (1 << ADPS2) | (1 << ADPS1) | (1 << ADPS0);
    
    // AVCC 기준전압, ADC0 선택
    ADMUX = (1 << REFS0);
}

void gpio_init(void) {
    // 버튼 핀을 입력으로 설정 (풀업 활성화)
    DDRD &= ~((1 << BUTTON1_PIN) | (1 << BUTTON2_PIN));
    PORTD |= (1 << BUTTON1_PIN) | (1 << BUTTON2_PIN);
}

void timer2_init(void) {
    // Timer2 일반 모드, 프리스케일러 = 64
    TCCR2B = (1 << CS22);  // ~1kHz 오버플로우
    TIMSK2 = (1 << TOIE2); // 오버플로우 인터럽트 활성화
}

void set_rgb(uint8_t r, uint8_t g, uint8_t b) {
    OCR0A = r;    // Red
    OCR0B = g;    // Green
    OCR1A = b;    // Blue
}

void set_rgb_gamma(uint8_t r, uint8_t g, uint8_t b) {
    OCR0A = apply_gamma(r);
    OCR0B = apply_gamma(g);
    OCR1A = apply_gamma(b);
}

uint8_t apply_gamma(uint8_t value) {
    return pgm_read_byte(&gamma_table[value]);
}

void fade_to_color(rgb_color_t from, rgb_color_t to, uint16_t duration_ms) {
    uint16_t steps = duration_ms / 10;
    if(steps == 0) steps = 1;
    
    for(uint16_t i = 0; i <= steps; i++) {
        uint8_t r = from.r + ((int32_t)(to.r - from.r) * i) / steps;
        uint8_t g = from.g + ((int32_t)(to.g - from.g) * i) / steps;
        uint8_t b = from.b + ((int32_t)(to.b - from.b) * i) / steps;
        
        set_rgb_gamma(r, g, b);
        _delay_ms(10);
        
        // 모드 변경 확인
        if(button1_pressed) return;
    }
}

void effect_manual(void) {
    // 가변저항으로 밝기 제어
    uint8_t brightness = adc_value >> 2;  // 10-bit to 8-bit
    
    // 버튼2로 색상 선택
    static uint8_t color_index = 0;
    if(button2_pressed) {
        button2_pressed = false;
        color_index = (color_index + 1) % 7;
    }
    
    rgb_color_t color;
    switch(color_index) {
        case 0: color = COLOR_WHITE; break;
        case 1: color = COLOR_RED; break;
        case 2: color = COLOR_GREEN; break;
        case 3: color = COLOR_BLUE; break;
        case 4: color = COLOR_YELLOW; break;
        case 5: color = COLOR_CYAN; break;
        case 6: color = COLOR_MAGENTA; break;
        default: color = COLOR_WHITE; break;
    }
    
    // 밝기 적용
    set_rgb_gamma(
        (color.r * brightness) / 255,
        (color.g * brightness) / 255,
        (color.b * brightness) / 255
    );
    
    _delay_ms(10);
}

void effect_fade(void) {
    static rgb_color_t colors[] = {
        {255, 0, 0}, {255, 127, 0}, {255, 255, 0},
        {0, 255, 0}, {0, 255, 255}, {0, 0, 255},
        {255, 0, 255}
    };
    
    static uint8_t index = 0;
    uint8_t next = (index + 1) % 7;
    
    // 가변저항으로 속도 조절
    uint16_t speed = 500 + (adc_value * 2);  // 0.5 ~ 2.5초
    
    fade_to_color(colors[index], colors[next], speed);
    index = next;
}

void effect_rainbow(void) {
    static uint16_t hue = 0;
    
    // 가변저항으로 속도 조절
    uint8_t speed = (adc_value >> 6) + 1;  // 1-16
    
    rgb_color_t color = hsv_to_rgb(hue, 255, 255);
    set_rgb_gamma(color.r, color.g, color.b);
    
    hue = (hue + speed) % 360;
    _delay_ms(20);
}

void effect_breathing(void) {
    static uint16_t phase = 0;
    static uint8_t color_index = 0;
    
    // 버튼2로 색상 변경
    if(button2_pressed) {
        button2_pressed = false;
        color_index = (color_index + 1) % 5;
    }
    
    rgb_color_t color;
    switch(color_index) {
        case 0: color = COLOR_WHITE; break;
        case 1: color = COLOR_RED; break;
        case 2: color = COLOR_GREEN; break;
        case 3: color = COLOR_BLUE; break;
        case 4: color = COLOR_PURPLE; break;
        default: color = COLOR_WHITE; break;
    }
    
    // 사인파로 부드러운 호흡 효과
    float brightness = (sin(phase * 0.01745) + 1.0) / 2.0;  // 0.0 ~ 1.0
    
    set_rgb_gamma(
        color.r * brightness,
        color.g * brightness,
        color.b * brightness
    );
    
    // 가변저항으로 속도 조절
    uint8_t speed = (adc_value >> 7) + 1;  // 1-8
    phase = (phase + speed) % 360;
    _delay_ms(20);
}

void effect_strobe(void) {
    // 가변저항으로 속도 조절
    uint16_t delay = (adc_value >> 2) + 50;  // 50-300ms
    
    set_rgb(255, 255, 255);
    _delay_ms(50);
    set_rgb(0, 0, 0);
    _delay_ms(delay);
}

void effect_fire(void) {
    // 불꽃 효과 시뮬레이션
    static uint8_t heat = 128;
    
    // 랜덤하게 열 변화
    int8_t change = (rand() % 50) - 25;
    int16_t new_heat = heat + change;
    
    if(new_heat < 50) new_heat = 50;
    if(new_heat > 255) new_heat = 255;
    heat = new_heat;
    
    // 열을 색상으로 변환
    uint8_t r = heat;
    uint8_t g = heat > 200 ? (heat - 200) : 0;
    uint8_t b = 0;
    
    set_rgb_gamma(r, g, b);
    _delay_ms(50);
}

// HSV to RGB 변환 (H: 0-359, S: 0-255, V: 0-255)
rgb_color_t hsv_to_rgb(uint16_t h, uint8_t s, uint8_t v) {
    rgb_color_t rgb;
    
    if(s == 0) {
        rgb.r = rgb.g = rgb.b = v;
        return rgb;
    }
    
    uint8_t region = h / 60;
    uint8_t remainder = (h % 60) * 255 / 60;
    
    uint8_t p = (v * (255 - s)) / 255;
    uint8_t q = (v * (255 - (s * remainder) / 255)) / 255;
    uint8_t t = (v * (255 - (s * (255 - remainder)) / 255)) / 255;
    
    switch(region) {
        case 0: rgb.r = v; rgb.g = t; rgb.b = p; break;
        case 1: rgb.r = q; rgb.g = v; rgb.b = p; break;
        case 2: rgb.r = p; rgb.g = v; rgb.b = t; break;
        case 3: rgb.r = p; rgb.g = q; rgb.b = v; break;
        case 4: rgb.r = t; rgb.g = p; rgb.b = v; break;
        default: rgb.r = v; rgb.g = p; rgb.b = q; break;
    }
    
    return rgb;
}