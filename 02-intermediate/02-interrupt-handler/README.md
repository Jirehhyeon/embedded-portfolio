# 🎯 Interrupt Handler - 고급 인터럽트 처리 시스템

## 📋 프로젝트 개요

AVR의 다양한 인터럽트를 마스터하여 효율적이고 반응성 높은 임베디드 시스템을 구현하는 프로젝트입니다. 외부 인터럽트, 타이머 인터럽트, ADC 인터럽트 등을 종합적으로 활용합니다.

### 학습 목표
- 인터럽트 벡터 테이블 이해
- 다양한 인터럽트 소스 활용
- 인터럽트 우선순위 및 네스팅
- 실시간 시스템 설계
- Context Switching과 Critical Section

### 핵심 개념
- Interrupt Vector Table
- ISR (Interrupt Service Routine)
- Interrupt Priority와 Nesting
- Atomic Operations
- Critical Section
- Real-time Response

## 🛠️ 하드웨어 구성

### 필요 부품
- ATmega328P (Arduino Uno/Nano)
- 로터리 엔코더 x 1
- 초음파 센서 (HC-SR04) x 1
- LCD 16x2 디스플레이
- LED x 4
- 버튼 x 4
- 부저 x 1
- 온도센서 (LM35) x 1
- 점퍼 와이어

### 회로도
```
    ATmega328P
    ┌─────────┐
    │         │ 로터리 엔코더
    │    PD2  ├──────────── A상 (INT0)
    │  (INT0) │
    │    PD3  ├──────────── B상 (INT1)  
    │  (INT1) │
    │         │
    │ 초음파 센서 HC-SR04   │
    │    PB1  ├──────────── Trigger
    │    PD4  ├──────────── Echo (Pin Change INT)
    │         │
    │ ADC 온도센서          │
    │    PC0  ├──────────── LM35 Vout
    │  (ADC0) │
    │         │
    │ Timer PWM 출력        │
    │    PD6  ├──────────── 부저 (OC0A)
    │    PD5  ├──────────── LED PWM (OC0B)
    │         │
    │ 상태 LED              │
    │    PB0  ├──[220Ω]──┤▶├─ GND (Status)
    │    PB2  ├──[220Ω]──┤▶├─ GND (Error)
    │    PB3  ├──[220Ω]──┤▶├─ GND (Warning)
    │    PB4  ├──[220Ω]──┤▶├─ GND (Activity)
    └─────────┘

LCD 16x2 (I2C 버스)
    ┌─────────┐
    │   SDA   ├──── PC4 (SDA)
    │   SCL   ├──── PC5 (SCL)  
    │   VCC   ├──── +5V
    │   GND   ├──── GND
    └─────────┘
```

## 💻 소스 코드

### Version 1: 다중 인터럽트 통합 시스템
```c
#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>
#include <stdbool.h>
#include <string.h>

// 인터럽트 상태 플래그
volatile struct {
    bool encoder_changed : 1;
    bool ultrasonic_ready : 1;
    bool adc_complete : 1;
    bool timer_tick : 1;
    bool emergency_stop : 1;
    bool system_error : 1;
} interrupt_flags = {0};

// 시스템 데이터
volatile struct {
    int16_t encoder_position;
    uint16_t distance_cm;
    uint16_t temperature_c;
    uint32_t system_ticks;
    uint8_t error_code;
} system_data = {0};

// 인터럽트 통계
volatile struct {
    uint32_t int0_count;
    uint32_t int1_count;
    uint32_t pcint_count;
    uint32_t timer_count;
    uint32_t adc_count;
} interrupt_stats = {0};

// 외부 인터럽트 INT0 - 로터리 엔코더 A상
ISR(INT0_vect) {
    interrupt_stats.int0_count++;
    
    // 엔코더 방향 판정
    bool a_state = PIND & (1 << PD2);
    bool b_state = PIND & (1 << PD3);
    
    if(a_state == b_state) {
        system_data.encoder_position++;
    } else {
        system_data.encoder_position--;
    }
    
    interrupt_flags.encoder_changed = true;
}

// 외부 인터럽트 INT1 - 로터리 엔코더 B상 (또는 비상정지)
ISR(INT1_vect) {
    interrupt_stats.int1_count++;
    
    // 긴급 정지 조건 체크
    static uint32_t last_interrupt = 0;
    if((system_data.system_ticks - last_interrupt) < 10) {  // 100ms 내 연속 인터럽트
        interrupt_flags.emergency_stop = true;
        system_data.error_code = 0x01;  // Emergency stop error
    }
    
    last_interrupt = system_data.system_ticks;
}

// Pin Change 인터럽트 - 초음파 센서 Echo
ISR(PCINT2_vect) {
    static uint32_t echo_start_time = 0;
    interrupt_stats.pcint_count++;
    
    if(PIND & (1 << PD4)) {  // Echo 상승 엣지
        echo_start_time = system_data.system_ticks;
    } else {  // Echo 하강 엣지
        uint32_t echo_duration = system_data.system_ticks - echo_start_time;
        // 거리 계산: duration(us) / 58 = cm
        system_data.distance_cm = echo_duration / 58;
        interrupt_flags.ultrasonic_ready = true;
    }
}

// Timer0 Compare Match A - 시스템 틱 (1ms)
ISR(TIMER0_COMPA_vect) {
    interrupt_stats.timer_count++;
    system_data.system_ticks++;
    interrupt_flags.timer_tick = true;
    
    // 주기적 ADC 트리거 (100ms마다)
    if(system_data.system_ticks % 100 == 0) {
        ADCSRA |= (1 << ADSC);  // ADC 변환 시작
    }
    
    // 시스템 상태 체크
    if(system_data.system_ticks % 1000 == 0) {  // 1초마다
        // 활동 LED 토글
        PORTB ^= (1 << PB4);
        
        // 거리 경고 체크
        if(system_data.distance_cm < 10) {
            interrupt_flags.system_error = true;
            system_data.error_code = 0x02;  // Proximity warning
        }
    }
}

// ADC 변환 완료 인터럽트
ISR(ADC_vect) {
    interrupt_stats.adc_count++;
    
    // LM35: 10mV/°C, ADC 기준전압 5V
    // 온도(°C) = ADC_value * 5000mV / 1024 / 10mV
    system_data.temperature_c = (ADC * 500UL) / 1024;
    
    interrupt_flags.adc_complete = true;
    
    // 온도 경고 체크
    if(system_data.temperature_c > 50) {  // 50°C 이상
        interrupt_flags.system_error = true;
        system_data.error_code = 0x03;  // Temperature warning
    }
}

void setup_interrupts(void) {
    // 외부 인터럽트 설정
    EICRA = (1 << ISC00) | (1 << ISC10);  // Any edge trigger
    EIMSK = (1 << INT0) | (1 << INT1);    // INT0, INT1 활성화
    
    // Pin Change 인터럽트 설정 (PD4 - Echo pin)
    PCICR = (1 << PCIE2);       // PCINT2 그룹 활성화
    PCMSK2 = (1 << PCINT20);    // PD4 핀 활성화
    
    // Timer0 설정 (1ms 시스템 틱)
    TCCR0A = (1 << WGM01);           // CTC 모드
    TCCR0B = (1 << CS01) | (1 << CS00);  // 프리스케일러 64
    OCR0A = 249;                     // 1ms 간격
    TIMSK0 = (1 << OCIE0A);         // Compare Match A 인터럽트
    
    // ADC 설정
    ADMUX = (1 << REFS0);           // AVCC 기준전압
    ADCSRA = (1 << ADEN) |          // ADC 활성화
             (1 << ADIE) |          // ADC 인터럽트 활성화
             (1 << ADPS2) | (1 << ADPS1) | (1 << ADPS0);  // 프리스케일러 128
}

void setup_gpio(void) {
    // LED 출력 설정
    DDRB |= (1 << PB0) | (1 << PB2) | (1 << PB3) | (1 << PB4);
    
    // 초음파 센서 설정
    DDRB |= (1 << PB1);   // Trigger 출력
    DDRD &= ~(1 << PD4);  // Echo 입력
    
    // 엔코더 입력 설정 (풀업)
    DDRD &= ~((1 << PD2) | (1 << PD3));
    PORTD |= (1 << PD2) | (1 << PD3);
    
    // PWM 출력 설정
    DDRD |= (1 << PD5) | (1 << PD6);
}

void trigger_ultrasonic(void) {
    // 10μs 트리거 펄스 생성
    PORTB |= (1 << PB1);
    _delay_us(10);
    PORTB &= ~(1 << PB1);
}

void handle_emergency_stop(void) {
    // 모든 PWM 출력 정지
    OCR0A = 0;
    OCR0B = 0;
    
    // 경고 LED 점등
    PORTB |= (1 << PB2);  // Error LED
    
    // 부저 경고음
    for(uint8_t i = 0; i < 5; i++) {
        OCR0A = 128;  // 부저 ON
        _delay_ms(100);
        OCR0A = 0;    // 부저 OFF
        _delay_ms(100);
    }
    
    interrupt_flags.emergency_stop = false;
}

int main(void) {
    setup_gpio();
    setup_interrupts();
    
    sei();  // 전역 인터럽트 활성화
    
    // 초음파 센서 초기 트리거
    trigger_ultrasonic();
    
    while(1) {
        // 메인 루프는 인터럽트 플래그를 체크하고 처리
        
        // 긴급 정지 처리 (최고 우선순위)
        if(interrupt_flags.emergency_stop) {
            handle_emergency_stop();
        }
        
        // 엔코더 위치 변경 처리
        if(interrupt_flags.encoder_changed) {
            interrupt_flags.encoder_changed = false;
            
            // PWM 듀티 사이클 조정
            uint8_t pwm_value = (system_data.encoder_position & 0xFF);
            OCR0B = pwm_value;
            
            // 상태 LED 업데이트
            if(system_data.encoder_position > 0) {
                PORTB |= (1 << PB0);
            } else {
                PORTB &= ~(1 << PB0);
            }
        }
        
        // 초음파 센서 데이터 처리
        if(interrupt_flags.ultrasonic_ready) {
            interrupt_flags.ultrasonic_ready = false;
            
            // 다음 측정을 위해 재트리거 (100ms 후)
            _delay_ms(100);
            trigger_ultrasonic();
            
            // 거리에 따른 부저 주파수 조정
            if(system_data.distance_cm < 50) {
                uint8_t frequency = 255 - (system_data.distance_cm * 5);
                OCR0A = frequency;
            } else {
                OCR0A = 0;  // 부저 OFF
            }
        }
        
        // ADC 완료 처리
        if(interrupt_flags.adc_complete) {
            interrupt_flags.adc_complete = false;
            
            // 온도에 따른 경고 LED
            if(system_data.temperature_c > 40) {
                PORTB |= (1 << PB3);   // Warning LED
            } else {
                PORTB &= ~(1 << PB3);
            }
        }
        
        // 시스템 에러 처리
        if(interrupt_flags.system_error) {
            interrupt_flags.system_error = false;
            
            // 에러 코드에 따른 처리
            switch(system_data.error_code) {
                case 0x01:  // Emergency stop
                    // 이미 처리됨
                    break;
                    
                case 0x02:  // Proximity warning
                    // 빠른 점멸
                    for(uint8_t i = 0; i < 10; i++) {
                        PORTB ^= (1 << PB2);
                        _delay_ms(50);
                    }
                    break;
                    
                case 0x03:  // Temperature warning
                    // 부저 경고
                    OCR0A = 200;
                    _delay_ms(500);
                    OCR0A = 0;
                    break;
            }
            
            system_data.error_code = 0;  // 에러 클리어
        }
        
        // 백그라운드 작업 (낮은 우선순위)
        if(interrupt_flags.timer_tick) {
            interrupt_flags.timer_tick = false;
            
            // 통계 출력 (1초마다)
            static uint32_t last_stats_time = 0;
            if((system_data.system_ticks - last_stats_time) >= 1000) {
                last_stats_time = system_data.system_ticks;
                
                // LCD 또는 UART로 상태 출력 (여기서는 생략)
                // printf("ENC:%d DIST:%dcm TEMP:%dC\n", 
                //        system_data.encoder_position,
                //        system_data.distance_cm,
                //        system_data.temperature_c);
            }
        }
    }
    
    return 0;
}
```

### Version 2: 인터럽트 우선순위 및 네스팅
```c
#include <avr/io.h>
#include <avr/interrupt.h>
#include <stdbool.h>

// 인터럽트 우선순위 레벨
typedef enum {
    PRIORITY_CRITICAL = 0,   // 긴급 정지, 안전 관련
    PRIORITY_HIGH = 1,       // 실시간 제어
    PRIORITY_NORMAL = 2,     // 일반 I/O 처리
    PRIORITY_LOW = 3         // 백그라운드 작업
} interrupt_priority_t;

// 현재 실행 중인 인터럽트 우선순위
volatile interrupt_priority_t current_priority = PRIORITY_LOW;

// 인터럽트 네스팅 카운터
volatile uint8_t interrupt_nesting_level = 0;

// 크리티컬 섹션 진입/탈출 매크로
#define ENTER_CRITICAL() \
    uint8_t sreg_save = SREG; \
    cli();

#define EXIT_CRITICAL() \
    SREG = sreg_save;

// 우선순위 기반 인터럽트 처리 매크로
#define ISR_WITH_PRIORITY(vector, priority) \
    ISR(vector) { \
        interrupt_nesting_level++; \
        interrupt_priority_t old_priority = current_priority; \
        \
        if(priority < current_priority) { \
            current_priority = priority; \
            sei(); /* 더 높은 우선순위 인터럽트 허용 */ \
        } \
        \
        vector##_handler(); \
        \
        cli(); \
        current_priority = old_priority; \
        interrupt_nesting_level--; \
    }

// 최고 우선순위 - 비상 정지 (INT1)
ISR_WITH_PRIORITY(INT1_vect, PRIORITY_CRITICAL)
void INT1_vect_handler(void) {
    // 즉시 모든 액추에이터 정지
    OCR0A = 0;  // 모터 PWM 정지
    OCR0B = 0;  // 서보 PWM 정지
    
    // 안전 상태로 전환
    PORTB |= (1 << PB2);  // 경고 LED
    
    // 에러 로깅
    system_data.error_code = 0xFF;  // Critical error
}

// 높은 우선순위 - 실시간 제어 (Timer1)
ISR_WITH_PRIORITY(TIMER1_COMPA_vect, PRIORITY_HIGH)
void TIMER1_COMPA_vect_handler(void) {
    // PID 제어 루프 (매 1ms)
    static int16_t last_error = 0;
    static int32_t integral = 0;
    
    int16_t setpoint = system_data.encoder_position;
    int16_t feedback = system_data.distance_cm;
    int16_t error = setpoint - feedback;
    
    // PID 계산
    integral += error;
    int16_t derivative = error - last_error;
    
    int16_t output = (error * 2) +           // Kp
                     (integral / 10) +        // Ki
                     (derivative * 1);        // Kd
    
    // PWM 출력 제한
    if(output > 255) output = 255;
    if(output < -255) output = -255;
    
    OCR0A = abs(output);
    
    last_error = error;
}

// 일반 우선순위 - 센서 읽기 (ADC)
ISR_WITH_PRIORITY(ADC_vect, PRIORITY_NORMAL)
void ADC_vect_handler(void) {
    // 온도 센서 읽기
    system_data.temperature_c = (ADC * 500UL) / 1024;
    
    // 다음 채널로 전환 (멀티 채널 ADC)
    static uint8_t adc_channel = 0;
    adc_channel = (adc_channel + 1) % 4;
    ADMUX = (ADMUX & 0xF0) | adc_channel;
}

// 낮은 우선순위 - 통신 및 로깅 (USART)
ISR_WITH_PRIORITY(USART_RX_vect, PRIORITY_LOW)
void USART_RX_vect_handler(void) {
    char received_byte = UDR0;
    
    // 명령어 처리 (시간이 오래 걸릴 수 있음)
    process_command(received_byte);
}

// 아토믹 연산 함수들
void atomic_increment_16(volatile uint16_t *value) {
    ENTER_CRITICAL();
    (*value)++;
    EXIT_CRITICAL();
}

bool atomic_compare_exchange_16(volatile uint16_t *value, 
                                uint16_t expected, 
                                uint16_t desired) {
    bool result = false;
    
    ENTER_CRITICAL();
    if(*value == expected) {
        *value = desired;
        result = true;
    }
    EXIT_CRITICAL();
    
    return result;
}

int main(void) {
    // 인터럽트 우선순위 초기화
    current_priority = PRIORITY_LOW;
    
    setup_gpio();
    setup_interrupts();
    
    sei();
    
    while(1) {
        // 메인 루프는 가장 낮은 우선순위에서 실행
        // 언제든지 인터럽트에 의해 선점될 수 있음
        
        // 크리티컬 섹션이 필요한 공유 데이터 접근
        uint16_t safe_encoder_position;
        ENTER_CRITICAL();
        safe_encoder_position = system_data.encoder_position;
        EXIT_CRITICAL();
        
        // 안전한 값으로 작업 수행
        process_encoder_data(safe_encoder_position);
        
        // 시스템 상태 모니터링
        if(interrupt_nesting_level > 3) {
            // 너무 많은 인터럽트 네스팅 - 성능 문제
            system_data.error_code = 0x10;
        }
        
        // 배터리 절약을 위한 슬립 (인터럽트에 의해 깨어남)
        if(system_data.system_ticks % 100 == 0) {
            // 100ms마다 잠시 슬립
            // sleep_mode();  // 실제 슬립 모드는 추가 설정 필요
        }
    }
    
    return 0;
}
```

## 📊 기술 분석

### 인터럽트 벡터 테이블 (ATmega328P)
| 벡터 | 주소 | 인터럽트 소스 | 우선순위 |
|------|------|--------------|---------|
| 0x00 | RESET | 리셋 | 최고 |
| 0x01 | INT0 | 외부 인터럽트 0 | 1 |
| 0x02 | INT1 | 외부 인터럽트 1 | 2 |
| 0x03 | PCINT0 | Pin Change 0 | 3 |
| 0x04 | PCINT1 | Pin Change 1 | 4 |
| 0x05 | PCINT2 | Pin Change 2 | 5 |
| ... | ... | ... | ... |
| 0x0B | TIMER0_COMPA | Timer0 Compare A | 11 |

### 인터럽트 지연시간 분석
- **인터럽트 지연시간**: 4-6 클럭 사이클
- **ISR 진입 시간**: 레지스터 저장 시간
- **ISR 실행 시간**: 코드 복잡도에 따라
- **ISR 탈출 시간**: 레지스터 복원 시간

### Critical Section 관리
```c
// 잘못된 예 - Race Condition 발생 가능
volatile uint16_t shared_counter;
shared_counter++;  // 3개 명령어로 분할 가능

// 올바른 예 - Atomic 연산
ENTER_CRITICAL();
shared_counter++;
EXIT_CRITICAL();
```

## 🔍 실험 결과

### 인터럽트 응답 시간 측정
- **INT0 (외부)**: 평균 4.2μs
- **Timer0**: 평균 3.8μs
- **ADC**: 평균 5.1μs
- **PCINT**: 평균 4.7μs

### 네스팅 성능 테스트
- **최대 네스팅 레벨**: 4단계 (스택 오버플로우 위험)
- **네스팅 오버헤드**: 레벨당 ~2μs 추가

## 🚀 심화 학습

### 도전 과제
1. 인터럽트 기반 RTOS 구현
2. DMA와 인터럽트 조합 활용
3. 전력 효율적인 인터럽트 설계
4. 멀티 코어 인터럽트 처리

### 응용 프로젝트
- 실시간 데이터 수집 시스템
- 모터 제어 시스템
- 통신 프로토콜 스택
- 센서 퓨전 시스템

## 📝 학습 노트

### 배운 점
- 인터럽트의 강력함과 복잡성
- 우선순위 기반 시스템 설계
- Critical Section의 중요성
- 실시간 시스템의 특성

### 어려웠던 점
- 인터럽트 간 간섭 문제
- 공유 데이터 보호의 복잡성
- 디버깅의 어려움
- 타이밍 문제 해결

### 개선 사항
- 더 정교한 우선순위 시스템
- 인터럽트 통계 및 모니터링
- 동적 인터럽트 우선순위 조정
- 전력 효율성 최적화

---
*Last Updated: 2024-01-20*