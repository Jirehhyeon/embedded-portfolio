# 🔘 Button Input - 디지털 입력 및 디바운싱

## 📋 프로젝트 개요

버튼 입력을 통한 디지털 신호 처리와 하드웨어 디바운싱 기법을 학습하는 프로젝트입니다.

### 학습 목표
- 디지털 입력 (Pull-up 저항 이해)
- 버튼 디바운싱 기법 구현
- 상태 머신을 이용한 버튼 이벤트 처리
- 인터럽트 기반 입력 처리

### 핵심 개념
- Input Pull-up 저항
- 디바운싱 (Debouncing)
- 엣지 검출 (Rising/Falling Edge)
- 상태 머신 (State Machine)

## 🛠️ 하드웨어 구성

### 필요 부품
- ATmega328P (Arduino Uno/Nano)
- 택트 스위치 (Push Button) x 2
- LED x 1
- 저항 220Ω x 1
- 저항 10kΩ x 2 (외부 풀업용, 옵션)
- 점퍼 와이어

### 회로도
```
    ATmega328P                    VCC
    ┌─────────┐                    │
    │         │                    ┌┴┐
    │    PD2  ├────────────┬───────┤ │ 10kΩ (옵션)
    │ (Pin 2) │            │       └┬┘
    │         │            │        │
    │         │          ──┴──      │
    │         │          ────  SW1  │
    │         │            │        │
    │         │            └────────┴──── GND
    │         │
    │    PB5  ├──[220Ω]──┤▶├── GND
    │(Pin 13) │          LED
    └─────────┘
```

## 💻 소스 코드

### Version 1: 기본 버튼 읽기 (폴링 방식)
```c
#include <avr/io.h>
#include <util/delay.h>

#define BUTTON_PIN  PD2
#define BUTTON_PORT PORTD
#define BUTTON_DDR  DDRD
#define BUTTON_PINS PIND

#define LED_PIN     PB5
#define LED_PORT    PORTB
#define LED_DDR     DDRB

int main(void) {
    // LED를 출력으로 설정
    LED_DDR |= (1 << LED_PIN);
    
    // 버튼을 입력으로 설정 (내부 풀업 활성화)
    BUTTON_DDR &= ~(1 << BUTTON_PIN);
    BUTTON_PORT |= (1 << BUTTON_PIN);
    
    while(1) {
        // 버튼이 눌렸는지 확인 (액티브 로우)
        if(!(BUTTON_PINS & (1 << BUTTON_PIN))) {
            LED_PORT |= (1 << LED_PIN);  // LED ON
        } else {
            LED_PORT &= ~(1 << LED_PIN); // LED OFF
        }
    }
    
    return 0;
}
```

### Version 2: 디바운싱 구현
```c
#include <avr/io.h>
#include <util/delay.h>
#include <stdbool.h>

#define BUTTON_PIN  PD2
#define BUTTON_PORT PORTD
#define BUTTON_DDR  DDRD
#define BUTTON_PINS PIND

#define LED_PIN     PB5
#define LED_PORT    PORTB
#define LED_DDR     DDRB

#define DEBOUNCE_TIME 50  // 50ms

typedef enum {
    BUTTON_IDLE,
    BUTTON_DEBOUNCE,
    BUTTON_PRESSED,
    BUTTON_RELEASE_DEBOUNCE
} button_state_t;

bool read_button(void) {
    static button_state_t state = BUTTON_IDLE;
    static uint8_t debounce_counter = 0;
    static bool button_pressed = false;
    
    bool current_state = !(BUTTON_PINS & (1 << BUTTON_PIN));
    
    switch(state) {
        case BUTTON_IDLE:
            if(current_state) {
                state = BUTTON_DEBOUNCE;
                debounce_counter = 0;
            }
            break;
            
        case BUTTON_DEBOUNCE:
            if(current_state) {
                debounce_counter++;
                if(debounce_counter >= DEBOUNCE_TIME) {
                    state = BUTTON_PRESSED;
                    button_pressed = true;
                }
            } else {
                state = BUTTON_IDLE;
            }
            break;
            
        case BUTTON_PRESSED:
            if(!current_state) {
                state = BUTTON_RELEASE_DEBOUNCE;
                debounce_counter = 0;
            }
            break;
            
        case BUTTON_RELEASE_DEBOUNCE:
            if(!current_state) {
                debounce_counter++;
                if(debounce_counter >= DEBOUNCE_TIME) {
                    state = BUTTON_IDLE;
                    button_pressed = false;
                }
            } else {
                state = BUTTON_PRESSED;
            }
            break;
    }
    
    return button_pressed;
}

int main(void) {
    // 초기화
    LED_DDR |= (1 << LED_PIN);
    BUTTON_DDR &= ~(1 << BUTTON_PIN);
    BUTTON_PORT |= (1 << BUTTON_PIN);
    
    bool led_state = false;
    bool last_button_state = false;
    
    while(1) {
        bool button_state = read_button();
        
        // 버튼이 눌렸을 때 (라이징 엣지)
        if(button_state && !last_button_state) {
            led_state = !led_state;  // LED 토글
            
            if(led_state) {
                LED_PORT |= (1 << LED_PIN);
            } else {
                LED_PORT &= ~(1 << LED_PIN);
            }
        }
        
        last_button_state = button_state;
        _delay_ms(1);
    }
    
    return 0;
}
```

### Version 3: 인터럽트 기반 처리
```c
#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>
#include <stdbool.h>

#define BUTTON1_PIN  PD2  // INT0
#define BUTTON2_PIN  PD3  // INT1
#define LED_PIN      PB5

volatile bool button1_flag = false;
volatile bool button2_flag = false;
volatile uint8_t press_count = 0;

// INT0 인터럽트 서비스 루틴
ISR(INT0_vect) {
    button1_flag = true;
}

// INT1 인터럽트 서비스 루틴
ISR(INT1_vect) {
    button2_flag = true;
    press_count++;
}

void setup_interrupts(void) {
    // 외부 인터럽트 INT0, INT1을 하강 엣지에서 트리거
    EICRA = (1 << ISC01) | (1 << ISC11);
    
    // INT0, INT1 인터럽트 활성화
    EIMSK = (1 << INT0) | (1 << INT1);
    
    // 전역 인터럽트 활성화
    sei();
}

int main(void) {
    // GPIO 설정
    DDRB |= (1 << LED_PIN);
    DDRD &= ~((1 << BUTTON1_PIN) | (1 << BUTTON2_PIN));
    PORTD |= (1 << BUTTON1_PIN) | (1 << BUTTON2_PIN);  // 풀업
    
    setup_interrupts();
    
    while(1) {
        // 버튼1 처리
        if(button1_flag) {
            button1_flag = false;
            
            // LED 토글
            PORTB ^= (1 << LED_PIN);
            
            // 디바운싱을 위한 짧은 지연
            _delay_ms(200);
        }
        
        // 버튼2 처리 (3번 누르면 빠른 점멸)
        if(press_count >= 3) {
            press_count = 0;
            
            // 빠른 점멸 5회
            for(uint8_t i = 0; i < 10; i++) {
                PORTB ^= (1 << LED_PIN);
                _delay_ms(100);
            }
        }
    }
    
    return 0;
}
```

## 📊 기술 분석

### 풀업 저항의 이해
```
VCC ──[10kΩ]──┬──── MCU 입력핀
              │
            ──┴── 버튼
              │
             GND
```
- 버튼을 누르지 않을 때: HIGH (VCC)
- 버튼을 누를 때: LOW (GND)

### 디바운싱 원리
버튼의 기계적 특성상 접점이 닫힐 때 바운싱 현상 발생:
```
실제 신호:    ─┐ ┌─┐ ┌─┐ ┌───────
              └─┘ └─┘ └─┘
              
디바운싱 후:  ─┐         ┌───────
              └─────────┘
```

### 인터럽트 vs 폴링
| 방식 | 장점 | 단점 |
|-----|------|------|
| 폴링 | 구현 간단, 디버깅 쉬움 | CPU 자원 낭비 |
| 인터럽트 | CPU 효율적, 빠른 응답 | 구현 복잡, 디버깅 어려움 |

## 🔍 실험 결과

### 디바운싱 시간 측정
- 일반 택트 스위치: 5-20ms
- 오래된 스위치: 20-50ms
- 권장 디바운스 시간: 50ms

### 성능 비교
- 폴링 방식: ~1ms 응답 시간
- 인터럽트 방식: <10μs 응답 시간

## 🚀 심화 학습

### 도전 과제
1. 더블 클릭, 롱 프레스 감지 구현
2. 여러 버튼의 조합 감지
3. 버튼으로 메뉴 시스템 구현
4. 로터리 엔코더 읽기

### 응용 프로젝트
- 버튼 기반 게임 (반응속도 테스트)
- 모스 부호 입력기
- 디지털 주사위
- 간단한 계산기

## 📝 학습 노트

### 배운 점
- 내부 풀업 저항의 편리함
- 디바운싱의 중요성 체감
- 상태 머신 설계 방법
- 인터럽트의 강력함과 주의점

### 어려웠던 점
- 디바운싱 없이는 오작동 발생
- 인터럽트 내에서 지연 함수 사용 불가
- 여러 버튼 동시 처리 복잡도

### 개선 사항
- 타이머 인터럽트를 이용한 정확한 디바운싱
- 버튼 라이브러리 제작
- 이벤트 기반 아키텍처 적용

---
*Last Updated: 2024-01-20*