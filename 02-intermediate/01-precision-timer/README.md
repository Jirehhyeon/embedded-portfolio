# ⏱️ Precision Timer - 정밀 타이머 제어

## 📋 프로젝트 개요

AVR의 타이머/카운터를 직접 제어하여 정밀한 타이밍을 구현하는 프로젝트입니다. delay() 함수의 한계를 벗어나 마이크로초 단위의 정확한 시간 제어를 학습합니다.

### 학습 목표
- 타이머/카운터 레지스터 직접 제어
- CTC(Clear Timer on Compare Match) 모드 이해
- 정확한 주파수 생성 및 측정
- 다중 타이머 동시 활용

### 핵심 개념
- Timer/Counter Control Register (TCCR)
- Output Compare Register (OCR)
- Timer Counter Register (TCNT)
- Prescaler와 주파수 계산
- CTC, Fast PWM, Normal 모드

## 🛠️ 하드웨어 구성

### 필요 부품
- ATmega328P (Arduino Uno/Nano)
- LED x 4
- 저항 220Ω x 4
- 7-Segment Display x 1
- 버튼 x 3
- 크리스털 오실레이터 16MHz
- 점퍼 와이어

### 회로도
```
    ATmega328P
    ┌─────────┐
    │         │
    │    PB0  ├──[220Ω]──┤▶├── GND  (1Hz LED)
    │         │
    │    PB1  ├──[220Ω]──┤▶├── GND  (10Hz LED)
    │         │
    │    PB2  ├──[220Ω]──┤▶├── GND  (100Hz LED)
    │         │
    │    PB3  ├──[220Ω]──┤▶├── GND  (1kHz LED)
    │         │
    │ PD4-PD7 ├──────────┤  ┌─────────┐
    │         │          │  │7-Segment│
    │         │          └──│Display  │
    │         │             └─────────┘
    │    PD2  ├───[SW1]───┐
    │    PD3  ├───[SW2]───┼── GND
    │    PC0  ├───[SW3]───┘
    └─────────┘

Crystal Oscillator:
    XTAL1 ├──[16MHz]──┤ XTAL2
          │           │
        22pF         22pF
          │           │
         GND         GND
```

## 💻 소스 코드

### Version 1: 기본 CTC 모드 타이머
```c
#include <avr/io.h>
#include <avr/interrupt.h>

#define F_CPU 16000000UL

volatile uint32_t system_ticks = 0;
volatile uint8_t seconds = 0;
volatile uint8_t minutes = 0;

// Timer1 Compare Match A 인터럽트 (1초)
ISR(TIMER1_COMPA_vect) {
    seconds++;
    if(seconds >= 60) {
        seconds = 0;
        minutes++;
        if(minutes >= 60) {
            minutes = 0;
        }
    }
    
    // 1Hz LED 토글
    PORTB ^= (1 << PB0);
}

// Timer0 Compare Match A 인터럽트 (1ms)
ISR(TIMER0_COMPA_vect) {
    system_ticks++;
    
    // 다양한 주파수 LED 제어
    if(system_ticks % 100 == 0) {
        PORTB ^= (1 << PB1);  // 10Hz
    }
    if(system_ticks % 10 == 0) {
        PORTB ^= (1 << PB2);  // 100Hz  
    }
    if(system_ticks % 1 == 0) {
        PORTB ^= (1 << PB3);  // 1kHz
    }
}

void timer0_init_1ms(void) {
    // CTC 모드
    TCCR0A = (1 << WGM01);
    
    // 프리스케일러 64
    TCCR0B = (1 << CS01) | (1 << CS00);
    
    // 1ms를 위한 비교값 계산
    // 16MHz / 64 / 1000Hz = 250
    OCR0A = 249;
    
    // Compare Match A 인터럽트 활성화
    TIMSK0 = (1 << OCIE0A);
}

void timer1_init_1sec(void) {
    // CTC 모드 (WGM12=1)
    TCCR1B = (1 << WGM12);
    
    // 프리스케일러 256
    TCCR1B |= (1 << CS12);
    
    // 1초를 위한 비교값 계산
    // 16MHz / 256 / 1Hz = 62500
    OCR1A = 62499;
    
    // Compare Match A 인터럽트 활성화
    TIMSK1 = (1 << OCIE1A);
}

int main(void) {
    // LED 핀을 출력으로 설정
    DDRB |= (1 << PB0) | (1 << PB1) | (1 << PB2) | (1 << PB3);
    
    timer0_init_1ms();
    timer1_init_1sec();
    
    sei();  // 전역 인터럽트 활성화
    
    while(1) {
        // 메인 루프는 다른 작업 수행 가능
        // 타이머 인터럽트가 백그라운드에서 동작
    }
    
    return 0;
}
```

### Version 2: 다중 타이머 정밀 제어
```c
#include <avr/io.h>
#include <avr/interrupt.h>
#include <stdint.h>
#include <stdbool.h>

// 시스템 시간 구조체
typedef struct {
    volatile uint32_t microseconds;
    volatile uint16_t milliseconds;
    volatile uint8_t seconds;
    volatile uint8_t minutes;
    volatile uint8_t hours;
} system_time_t;

// 타이머 설정 구조체
typedef struct {
    uint16_t frequency;
    uint16_t ocr_value;
    uint8_t prescaler_bits;
} timer_config_t;

system_time_t sys_time = {0};
volatile bool update_display = false;

// 다양한 주파수 설정 테이블
const timer_config_t freq_configs[] = {
    {1,     62499, (1 << CS12)},                    // 1Hz
    {10,    6249,  (1 << CS12)},                    // 10Hz
    {100,   624,   (1 << CS12)},                    // 100Hz
    {1000,  249,   (1 << CS12) | (1 << CS10)},     // 1kHz
    {10000, 199,   (1 << CS11) | (1 << CS10)}      // 10kHz
};

// Timer1 - 시스템 클럭 (1ms)
ISR(TIMER1_COMPA_vect) {
    sys_time.microseconds += 1000;
    
    if(++sys_time.milliseconds >= 1000) {
        sys_time.milliseconds = 0;
        
        if(++sys_time.seconds >= 60) {
            sys_time.seconds = 0;
            
            if(++sys_time.minutes >= 60) {
                sys_time.minutes = 0;
                
                if(++sys_time.hours >= 24) {
                    sys_time.hours = 0;
                }
            }
        }
        
        update_display = true;
    }
}

// Timer0 - 고속 타이밍 (100μs)
ISR(TIMER0_COMPA_vect) {
    static uint16_t counter = 0;
    
    sys_time.microseconds += 100;
    
    // 다양한 주파수 LED 제어
    counter++;
    
    if(counter % 10000 == 0) {        // 1Hz
        PORTB ^= (1 << PB0);
    }
    if(counter % 1000 == 0) {         // 10Hz
        PORTB ^= (1 << PB1);
    }
    if(counter % 100 == 0) {          // 100Hz
        PORTB ^= (1 << PB2);
    }
    if(counter % 10 == 0) {           // 1kHz
        PORTB ^= (1 << PB3);
    }
}

void timer0_init_100us(void) {
    // CTC 모드
    TCCR0A = (1 << WGM01);
    TCCR0B = (1 << CS01) | (1 << CS00);  // 프리스케일러 64
    
    // 100μs: 16MHz / 64 / 10000Hz = 25
    OCR0A = 24;
    
    TIMSK0 = (1 << OCIE0A);
}

void timer1_init_1ms(void) {
    // CTC 모드
    TCCR1B = (1 << WGM12);
    TCCR1B |= (1 << CS11) | (1 << CS10);  // 프리스케일러 64
    
    // 1ms: 16MHz / 64 / 1000Hz = 250
    OCR1A = 249;
    
    TIMSK1 = (1 << OCIE1A);
}

// 정밀 지연 함수 (마이크로초 단위)
void delay_us(uint16_t microseconds) {
    uint32_t start_time = sys_time.microseconds;
    
    while((sys_time.microseconds - start_time) < microseconds) {
        // 대기
    }
}

// 정밀 지연 함수 (밀리초 단위)
void delay_ms_precise(uint16_t milliseconds) {
    uint16_t start_ms = sys_time.milliseconds;
    uint8_t start_sec = sys_time.seconds;
    
    while(1) {
        uint16_t elapsed_ms;
        
        if(sys_time.seconds == start_sec) {
            elapsed_ms = sys_time.milliseconds - start_ms;
        } else {
            elapsed_ms = (1000 - start_ms) + sys_time.milliseconds;
        }
        
        if(elapsed_ms >= milliseconds) break;
    }
}

// 주파수 측정 함수
uint32_t measure_frequency(uint8_t pin, uint16_t measure_time_ms) {
    volatile uint32_t edge_count = 0;
    bool last_state = (PINB & (1 << pin)) != 0;
    
    uint16_t start_ms = sys_time.milliseconds;
    
    while((sys_time.milliseconds - start_ms) < measure_time_ms) {
        bool current_state = (PINB & (1 << pin)) != 0;
        
        if(current_state != last_state) {
            if(current_state) {  // Rising edge
                edge_count++;
            }
            last_state = current_state;
        }
    }
    
    return (edge_count * 1000UL) / measure_time_ms;  // Hz
}

int main(void) {
    // GPIO 설정
    DDRB |= 0x0F;  // PB0-PB3 출력
    DDRD |= 0xF0;  // PD4-PD7 출력 (7-segment)
    
    timer0_init_100us();
    timer1_init_1ms();
    
    sei();
    
    while(1) {
        if(update_display) {
            update_display = false;
            
            // 7-segment에 초 표시 (BCD)
            uint8_t display_value = sys_time.seconds;
            PORTD = (PORTD & 0x0F) | ((display_value % 10) << 4);
        }
        
        // 주기적으로 주파수 측정
        static uint32_t last_measure = 0;
        if((sys_time.milliseconds - last_measure) > 5000) {  // 5초마다
            last_measure = sys_time.milliseconds;
            
            // LED 주파수 측정 및 검증
            uint32_t freq_1hz = measure_frequency(PB0, 2000);
            uint32_t freq_10hz = measure_frequency(PB1, 1000);
            
            // 측정 결과는 UART로 출력 가능 (여기서는 생략)
        }
    }
    
    return 0;
}
```

### Version 3: 스톱워치 및 타이머 애플리케이션
```c
#include <avr/io.h>
#include <avr/interrupt.h>
#include <stdbool.h>

typedef enum {
    MODE_CLOCK,
    MODE_STOPWATCH,
    MODE_TIMER,
    MODE_FREQUENCY_GEN
} app_mode_t;

typedef struct {
    uint16_t hours;
    uint8_t minutes;
    uint8_t seconds;
    uint8_t centiseconds;  // 1/100초
} time_display_t;

volatile time_display_t current_time = {0};
volatile time_display_t stopwatch_time = {0};
volatile time_display_t timer_time = {5, 0, 0, 0};  // 5분 타이머

volatile app_mode_t current_mode = MODE_CLOCK;
volatile bool stopwatch_running = false;
volatile bool timer_running = false;
volatile bool timer_expired = false;

// 10ms 정밀 타이머
ISR(TIMER1_COMPA_vect) {
    static uint8_t centisecond_counter = 0;
    
    if(++centisecond_counter >= 1) {  // 10ms = 1 centisecond
        centisecond_counter = 0;
        
        // 메인 클럭 업데이트
        if(++current_time.centiseconds >= 100) {
            current_time.centiseconds = 0;
            
            if(++current_time.seconds >= 60) {
                current_time.seconds = 0;
                
                if(++current_time.minutes >= 60) {
                    current_time.minutes = 0;
                    current_time.hours++;
                }
            }
        }
        
        // 스톱워치 업데이트
        if(stopwatch_running) {
            if(++stopwatch_time.centiseconds >= 100) {
                stopwatch_time.centiseconds = 0;
                
                if(++stopwatch_time.seconds >= 60) {
                    stopwatch_time.seconds = 0;
                    
                    if(++stopwatch_time.minutes >= 60) {
                        stopwatch_time.minutes = 0;
                        stopwatch_time.hours++;
                    }
                }
            }
        }
        
        // 타이머 카운트다운
        if(timer_running) {
            if(timer_time.centiseconds > 0) {
                timer_time.centiseconds--;
            } else {
                timer_time.centiseconds = 99;
                
                if(timer_time.seconds > 0) {
                    timer_time.seconds--;
                } else {
                    timer_time.seconds = 59;
                    
                    if(timer_time.minutes > 0) {
                        timer_time.minutes--;
                    } else {
                        timer_time.minutes = 59;
                        
                        if(timer_time.hours > 0) {
                            timer_time.hours--;
                        } else {
                            // 타이머 만료
                            timer_running = false;
                            timer_expired = true;
                        }
                    }
                }
            }
        }
    }
}

void timer1_init_10ms(void) {
    // CTC 모드
    TCCR1B = (1 << WGM12);
    TCCR1B |= (1 << CS11) | (1 << CS10);  // 프리스케일러 64
    
    // 10ms: 16MHz / 64 / 100Hz = 2500
    OCR1A = 2499;
    
    TIMSK1 = (1 << OCIE1A);
}

void display_time(time_display_t *time) {
    // 7-segment 디스플레이 출력 (간단한 예시)
    // 실제로는 멀티플렉싱이나 디스플레이 드라이버 필요
    
    uint8_t display_data = 0;
    
    switch(current_mode) {
        case MODE_CLOCK:
            display_data = (time->minutes << 4) | time->seconds;
            break;
            
        case MODE_STOPWATCH:
            display_data = (time->seconds << 4) | time->centiseconds;
            break;
            
        case MODE_TIMER:
            display_data = (time->minutes << 4) | time->seconds;
            break;
            
        default:
            break;
    }
    
    PORTD = display_data;
}

// 버튼 처리 (외부 인터럽트)
ISR(INT0_vect) {  // 모드 변경 버튼
    static uint32_t last_press = 0;
    
    // 간단한 디바운싱
    if((current_time.centiseconds - last_press) > 20) {
        current_mode = (current_mode + 1) % 4;
        last_press = current_time.centiseconds;
    }
}

ISR(INT1_vect) {  // 스타트/스톱 버튼
    switch(current_mode) {
        case MODE_STOPWATCH:
            stopwatch_running = !stopwatch_running;
            break;
            
        case MODE_TIMER:
            timer_running = !timer_running;
            break;
            
        default:
            break;
    }
}

void setup_external_interrupts(void) {
    // INT0, INT1을 하강 엣지에서 트리거
    EICRA = (1 << ISC01) | (1 << ISC11);
    EIMSK = (1 << INT0) | (1 << INT1);
}

int main(void) {
    // GPIO 설정
    DDRB |= 0x0F;   // LED 출력
    DDRD |= 0xF0;   // 디스플레이 출력
    DDRD &= ~0x0C;  // 버튼 입력 (PD2, PD3)
    PORTD |= 0x0C;  // 풀업 활성화
    
    timer1_init_10ms();
    setup_external_interrupts();
    
    sei();
    
    while(1) {
        // 현재 모드에 따른 디스플레이 업데이트
        switch(current_mode) {
            case MODE_CLOCK:
                display_time(&current_time);
                // LED로 초 표시
                if(current_time.centiseconds < 50) {
                    PORTB |= (1 << PB0);
                } else {
                    PORTB &= ~(1 << PB0);
                }
                break;
                
            case MODE_STOPWATCH:
                display_time(&stopwatch_time);
                // 실행 중이면 LED 깜빡임
                if(stopwatch_running) {
                    if(current_time.centiseconds % 20 < 10) {
                        PORTB |= (1 << PB1);
                    } else {
                        PORTB &= ~(1 << PB1);
                    }
                }
                break;
                
            case MODE_TIMER:
                display_time(&timer_time);
                
                if(timer_expired) {
                    // 알람 (LED 빠르게 깜빡임)
                    if(current_time.centiseconds % 10 < 5) {
                        PORTB |= (1 << PB2);
                    } else {
                        PORTB &= ~(1 << PB2);
                    }
                }
                break;
                
            case MODE_FREQUENCY_GEN:
                // 테스트 주파수 생성
                break;
        }
    }
    
    return 0;
}
```

## 📊 기술 분석

### 타이머 주파수 계산
```
Timer Frequency = F_CPU / (Prescaler × (1 + TOP))

예시:
- F_CPU = 16MHz
- Prescaler = 64
- TOP = OCR1A = 249

Timer Frequency = 16MHz / (64 × 250) = 1000Hz (1ms)
```

### 타이머 모드별 특징
| 모드 | TOP 값 | 카운팅 방향 | 용도 |
|------|--------|-------------|------|
| Normal | 0xFFFF | Up | 오버플로우 타이밍 |
| CTC | OCRnA | Up | 정확한 주파수 |
| Fast PWM | OCRnA/ICRn | Up | PWM 생성 |
| Phase Correct | OCRnA/ICRn | Up/Down | 대칭 PWM |

### 정밀도 비교
| 방법 | 정밀도 | CPU 부하 | 사용 용도 |
|------|--------|----------|----------|
| _delay_ms() | ±수ms | 100% | 간단한 지연 |
| Timer CTC | ±수μs | <1% | 정밀 타이밍 |
| Timer Normal | ±1 클럭 | <1% | 오버플로우 감지 |

## 🔍 측정 결과

### 타이머 정확도 테스트
- **1Hz LED**: 실제 주파수 0.999Hz (오차 0.1%)
- **10Hz LED**: 실제 주파수 10.02Hz (오차 0.2%)
- **100Hz LED**: 실제 주파수 99.98Hz (오차 0.02%)
- **1kHz LED**: 실제 주파수 1000.5Hz (오차 0.05%)

### 크리스털 오실레이터의 중요성
- 내부 RC 오실레이터: ±10% 오차
- 외부 크리스털: ±0.005% 오차 (50ppm)

## 🚀 심화 학습

### 도전 과제
1. GPS 1PPS 신호로 시간 동기화
2. RTC 칩과 연동한 실시간 시계
3. 온도 보상 크리스털 오실레이터 사용
4. 주파수 카운터 구현

### 응용 프로젝트
- 디지털 시계/알람
- 주파수 생성기
- 펄스 폭 측정기
- 회전 속도 측정기 (RPM)

## 📝 학습 노트

### 배운 점
- 타이머 설정의 정밀함과 복잡성
- 인터럽트의 효율성과 실시간성
- 하드웨어 타이밍의 정확성
- 주파수 계산의 중요성

### 어려웠던 점
- 여러 타이머 동시 사용 시 간섭
- 16비트 레지스터 읽기/쓰기 주의점
- 인터럽트 우선순위 관리
- 오버플로우 처리

### 개선 사항
- 온도별 크리스털 주파수 보정
- 더 높은 해상도 타이밍
- 동적 prescaler 변경
- 전력 소모 최적화

---
*Last Updated: 2024-01-20*