# 🌈 PWM Brightness Control - PWM을 이용한 LED 밝기 제어

## 📋 프로젝트 개요

PWM(Pulse Width Modulation)을 이용하여 LED 밝기를 제어하고, 다양한 효과를 구현하는 프로젝트입니다.

### 학습 목표
- PWM 원리 이해 및 구현
- 타이머/카운터를 이용한 PWM 생성
- 듀티 사이클(Duty Cycle) 제어
- 부드러운 페이드 효과 구현

### 핵심 개념
- PWM (Pulse Width Modulation)
- 듀티 사이클 (Duty Cycle)
- 타이머/카운터 모드 (Fast PWM, Phase Correct PWM)
- 주파수와 해상도의 관계

## 🛠️ 하드웨어 구성

### 필요 부품
- ATmega328P (Arduino Uno/Nano)
- LED x 3 (RGB LED 가능)
- 저항 220Ω x 3
- 가변저항 10kΩ x 1
- 버튼 x 2
- 점퍼 와이어

### 회로도
```
    ATmega328P
    ┌─────────┐
    │         │
    │    PD6  ├──[220Ω]──┤▶├── GND  (Red LED)
    │  (OC0A) │
    │         │
    │    PD5  ├──[220Ω]──┤▶├── GND  (Green LED)
    │  (OC0B) │
    │         │
    │    PB1  ├──[220Ω]──┤▶├── GND  (Blue LED)
    │  (OC1A) │
    │         │        VCC
    │    PC0  ├─────────┤
    │   (ADC0)│         │
    │         │        ┌┴┐
    │         │        │█│ 10kΩ POT
    │         │        │█│
    │         │        │█│◄─── Wiper
    │         │        └┬┘
    │         │         │
    │         │        GND
    └─────────┘
```

## 💻 소스 코드

### Version 1: 기본 PWM 제어
```c
#include <avr/io.h>
#include <util/delay.h>

void pwm_init(void) {
    // Timer0 Fast PWM 모드 설정
    TCCR0A = (1 << WGM01) | (1 << WGM00);  // Fast PWM
    TCCR0A |= (1 << COM0A1) | (1 << COM0B1); // Non-inverting mode
    TCCR0B = (1 << CS01);  // Prescaler 8
    
    // Timer1 Fast PWM 8-bit 모드
    TCCR1A = (1 << WGM10) | (1 << COM1A1);
    TCCR1B = (1 << WGM12) | (1 << CS11);
    
    // PWM 핀을 출력으로 설정
    DDRD |= (1 << PD6) | (1 << PD5);  // OC0A, OC0B
    DDRB |= (1 << PB1);                // OC1A
}

int main(void) {
    pwm_init();
    
    while(1) {
        // 밝기 증가 (Fade In)
        for(uint8_t i = 0; i < 255; i++) {
            OCR0A = i;  // Red
            OCR0B = i;  // Green
            OCR1A = i;  // Blue
            _delay_ms(10);
        }
        
        // 밝기 감소 (Fade Out)
        for(uint8_t i = 255; i > 0; i--) {
            OCR0A = i;
            OCR0B = i;
            OCR1A = i;
            _delay_ms(10);
        }
    }
    
    return 0;
}
```

### Version 2: ADC를 이용한 밝기 제어
```c
#include <avr/io.h>
#include <util/delay.h>

void adc_init(void) {
    // ADC 활성화, 프리스케일러 128
    ADCSRA = (1 << ADEN) | (1 << ADPS2) | (1 << ADPS1) | (1 << ADPS0);
    
    // AVCC를 기준전압으로 사용
    ADMUX = (1 << REFS0);
}

uint16_t adc_read(uint8_t channel) {
    // 채널 선택
    ADMUX = (ADMUX & 0xF0) | (channel & 0x0F);
    
    // 변환 시작
    ADCSRA |= (1 << ADSC);
    
    // 변환 완료 대기
    while(ADCSRA & (1 << ADSC));
    
    return ADC;
}

int main(void) {
    pwm_init();
    adc_init();
    
    while(1) {
        // 가변저항 값 읽기 (0-1023)
        uint16_t pot_value = adc_read(0);
        
        // 8비트로 변환 (0-255)
        uint8_t brightness = pot_value >> 2;
        
        // 모든 LED에 적용
        OCR0A = brightness;
        OCR0B = brightness;
        OCR1A = brightness;
        
        _delay_ms(10);
    }
    
    return 0;
}
```

### Version 3: RGB LED 색상 효과
```c
#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>
#include <math.h>

#define RED_PIN    PD6  // OC0A
#define GREEN_PIN  PD5  // OC0B
#define BLUE_PIN   PB1  // OC1A

typedef struct {
    uint8_t r;
    uint8_t g;
    uint8_t b;
} rgb_color_t;

// 색상 정의
const rgb_color_t colors[] = {
    {255, 0, 0},    // Red
    {255, 127, 0},  // Orange
    {255, 255, 0},  // Yellow
    {0, 255, 0},    // Green
    {0, 0, 255},    // Blue
    {75, 0, 130},   // Indigo
    {148, 0, 211}   // Violet
};

void set_rgb(uint8_t r, uint8_t g, uint8_t b) {
    OCR0A = r;
    OCR0B = g;
    OCR1A = b;
}

void fade_to_color(rgb_color_t from, rgb_color_t to, uint16_t duration_ms) {
    uint16_t steps = duration_ms / 10;
    
    for(uint16_t i = 0; i <= steps; i++) {
        uint8_t r = from.r + ((int16_t)(to.r - from.r) * i) / steps;
        uint8_t g = from.g + ((int16_t)(to.g - from.g) * i) / steps;
        uint8_t b = from.b + ((int16_t)(to.b - from.b) * i) / steps;
        
        set_rgb(r, g, b);
        _delay_ms(10);
    }
}

// 호흡 효과 (Breathing effect)
void breathing_effect(rgb_color_t color) {
    for(uint16_t i = 0; i < 360; i += 5) {
        float brightness = (sin(i * M_PI / 180.0) + 1.0) / 2.0;
        
        set_rgb(color.r * brightness,
                color.g * brightness,
                color.b * brightness);
        
        _delay_ms(50);
    }
}

// 무지개 효과
void rainbow_effect(void) {
    uint8_t num_colors = sizeof(colors) / sizeof(colors[0]);
    
    for(uint8_t i = 0; i < num_colors; i++) {
        uint8_t next = (i + 1) % num_colors;
        fade_to_color(colors[i], colors[next], 1000);
    }
}

int main(void) {
    pwm_init();
    
    rgb_color_t current_color = {0, 0, 0};
    uint8_t mode = 0;
    
    while(1) {
        switch(mode) {
            case 0:  // 무지개 효과
                rainbow_effect();
                break;
                
            case 1:  // 빨간색 호흡
                breathing_effect((rgb_color_t){255, 0, 0});
                break;
                
            case 2:  // 파란색 호흡
                breathing_effect((rgb_color_t){0, 0, 255});
                break;
                
            case 3:  // 흰색 페이드
                fade_to_color(current_color, (rgb_color_t){255, 255, 255}, 2000);
                current_color = (rgb_color_t){255, 255, 255};
                _delay_ms(1000);
                fade_to_color(current_color, (rgb_color_t){0, 0, 0}, 2000);
                current_color = (rgb_color_t){0, 0, 0};
                break;
        }
        
        mode = (mode + 1) % 4;
    }
    
    return 0;
}
```

## 📊 기술 분석

### PWM 동작 원리
```
듀티 사이클 25%:  ┌─┐   ┌─┐   ┌─┐
                └─┴───┴─┴───┴─┴───
                
듀티 사이클 50%:  ┌──┐  ┌──┐  ┌──┐
                └──┴──┴──┴──┴──┴──
                
듀티 사이클 75%:  ┌───┐ ┌───┐ ┌───┐
                └─┴───┴─┴───┴─┴───
```

### PWM 주파수 계산
```
주파수 = F_CPU / (프리스케일러 × TOP값)

Fast PWM (8-bit):
주파수 = 16MHz / (8 × 256) = 7.8125 kHz

Phase Correct PWM (8-bit):
주파수 = 16MHz / (8 × 510) = 3.922 kHz
```

### 타이머 모드 비교
| 모드 | 특징 | 용도 |
|-----|------|------|
| Fast PWM | 높은 주파수, 고정 주기 | LED 제어, 모터 속도 |
| Phase Correct | 대칭 파형, 낮은 주파수 | 모터 제어, 오디오 |
| CTC | 정확한 주파수 생성 | 타이밍, 통신 |

## 🔍 측정 결과

### LED 밝기 vs 듀티 사이클
- 0%: 완전히 꺼짐
- 25%: 희미하게 보임
- 50%: 중간 밝기
- 75%: 밝음
- 100%: 최대 밝기

### 인간의 눈 특성
- 선형적이지 않은 밝기 인식
- 감마 보정 필요 (γ = 2.2)
- 실제 밝기 = (설정값/255)^2.2 × 255

## 🚀 심화 학습

### 도전 과제
1. 감마 보정 적용하여 선형적 밝기 변화 구현
2. 음악에 반응하는 LED 효과
3. 온도에 따른 색상 변화 (색온도)
4. 다중 LED를 이용한 애니메이션

### 응용 프로젝트
- RGB LED 무드등
- 디지털 캔들 효과
- 경찰차/구급차 사이렌 라이트
- 게임 상태 표시 LED

## 📝 학습 노트

### 배운 점
- PWM의 원리와 다양한 활용법
- 타이머 설정의 중요성
- 부드러운 전환 효과 구현 방법
- RGB 색상 혼합 원리

### 어려웠던 점
- 타이머 레지스터 설정이 복잡함
- 여러 타이머 동시 사용 시 충돌
- 정확한 타이밍 계산

### 개선 사항
- DMA를 이용한 자동 페이드
- 더 많은 PWM 채널 활용
- 외부 PWM 드라이버 칩 사용 고려

---
*Last Updated: 2024-01-20*