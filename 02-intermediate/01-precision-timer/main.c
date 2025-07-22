/*
 * Precision Timer Control System
 * 
 * Hardware: ATmega328P with 16MHz crystal
 * Outputs: PB0-PB3 (LEDs), PD4-PD7 (7-segment)
 * Inputs: PD2 (INT0), PD3 (INT1), PC0-PC2 (buttons)
 * 
 * This program demonstrates:
 * - Multiple timer configuration and synchronization
 * - CTC (Clear Timer on Compare) mode
 * - Precise frequency generation (1Hz to 10kHz)
 * - Real-time clock with stopwatch and countdown timer
 * - Microsecond precision timing functions
 * - Frequency measurement and validation
 * 
 * Features:
 * - Clock mode with hours:minutes:seconds
 * - Stopwatch with centisecond precision
 * - Countdown timer with alarm
 * - Frequency generator for testing
 * - Multiple LED indicators at different frequencies
 * 
 * Author: [Your Name]
 * Date: 2024-01-20
 */

#include <avr/io.h>
#include <avr/interrupt.h>
#include <avr/pgmspace.h>
#include <stdbool.h>
#include <stdint.h>

// 하드웨어 핀 정의
#define LED_1HZ     PB0
#define LED_10HZ    PB1
#define LED_100HZ   PB2  
#define LED_1KHZ    PB3

#define SEG_A       PD4
#define SEG_B       PD5
#define SEG_C       PD6
#define SEG_D       PD7

#define BTN_MODE    PD2  // INT0
#define BTN_SET     PD3  // INT1
#define BTN_START   PC0
#define BTN_RESET   PC1

// 시스템 상수
#define F_CPU       16000000UL
#define TIMER_1MS   249    // Timer0: 16MHz/64/1000Hz - 1
#define TIMER_10MS  2499   // Timer1: 16MHz/64/100Hz - 1
#define TIMER_100US 24     // Timer2: 16MHz/64/10000Hz - 1

// 애플리케이션 모드
typedef enum {
    MODE_CLOCK = 0,
    MODE_STOPWATCH,
    MODE_COUNTDOWN,
    MODE_FREQ_GEN,
    MODE_COUNT
} app_mode_t;

// 시간 구조체
typedef struct {
    volatile uint8_t hours;
    volatile uint8_t minutes;
    volatile uint8_t seconds;
    volatile uint8_t centiseconds;
    volatile uint32_t total_centiseconds;
} time_t;

// 주파수 생성 설정
typedef struct {
    uint16_t frequency;
    uint16_t prescaler;
    uint16_t compare_value;
    uint8_t pin;
} freq_config_t;

// 7-segment 디스플레이 패턴 (0-9, A-F)
const uint8_t seven_seg_patterns[16] PROGMEM = {
    0b11111100,  // 0
    0b01100000,  // 1
    0b11011010,  // 2
    0b11110010,  // 3
    0b01100110,  // 4
    0b10110110,  // 5
    0b10111110,  // 6
    0b11100000,  // 7
    0b11111110,  // 8
    0b11110110,  // 9
    0b11101110,  // A
    0b00111110,  // b
    0b10011100,  // C
    0b01111010,  // d
    0b10011110,  // E
    0b10001110   // F
};

// 전역 변수
volatile app_mode_t current_mode = MODE_CLOCK;
volatile time_t system_clock = {0, 0, 0, 0, 0};
volatile time_t stopwatch = {0, 0, 0, 0, 0};
volatile time_t countdown = {0, 5, 0, 0, 30000};  // 5분 초기값

volatile bool stopwatch_running = false;
volatile bool countdown_running = false;
volatile bool countdown_expired = false;
volatile bool display_update = false;
volatile bool button_pressed = false;

volatile uint32_t microsecond_counter = 0;
volatile uint16_t freq_gen_counter = 0;

// 디스플레이 멀티플렉싱 변수
volatile uint8_t display_digit = 0;
volatile uint8_t display_data[4] = {0, 0, 0, 0};

// 인터럽트 서비스 루틴들

// Timer0 - 시스템 메인 클럭 (1ms)
ISR(TIMER0_COMPA_vect) {
    static uint8_t ms_counter = 0;
    
    // 마이크로초 카운터 업데이트
    microsecond_counter += 1000;
    
    // 10ms마다 센티초 업데이트
    if(++ms_counter >= 10) {
        ms_counter = 0;
        
        // 시스템 클럭 업데이트
        system_clock.total_centiseconds++;
        if(++system_clock.centiseconds >= 100) {
            system_clock.centiseconds = 0;
            
            if(++system_clock.seconds >= 60) {
                system_clock.seconds = 0;
                
                if(++system_clock.minutes >= 60) {
                    system_clock.minutes = 0;
                    
                    if(++system_clock.hours >= 24) {
                        system_clock.hours = 0;
                    }
                }
            }
        }
        
        // 스톱워치 업데이트
        if(stopwatch_running) {
            stopwatch.total_centiseconds++;
            if(++stopwatch.centiseconds >= 100) {
                stopwatch.centiseconds = 0;
                
                if(++stopwatch.seconds >= 60) {
                    stopwatch.seconds = 0;
                    
                    if(++stopwatch.minutes >= 60) {
                        stopwatch.minutes = 0;
                        
                        if(++stopwatch.hours >= 24) {
                            stopwatch.hours = 0;
                        }
                    }
                }
            }
        }
        
        // 카운트다운 타이머 업데이트
        if(countdown_running && countdown.total_centiseconds > 0) {
            countdown.total_centiseconds--;
            
            if(countdown.centiseconds > 0) {
                countdown.centiseconds--;
            } else {
                countdown.centiseconds = 99;
                
                if(countdown.seconds > 0) {
                    countdown.seconds--;
                } else {
                    countdown.seconds = 59;
                    
                    if(countdown.minutes > 0) {
                        countdown.minutes--;
                    } else {
                        countdown.minutes = 59;
                        
                        if(countdown.hours > 0) {
                            countdown.hours--;
                        } else {
                            // 타이머 만료
                            countdown_running = false;
                            countdown_expired = true;
                        }
                    }
                }
            }
            
            // 만료 체크
            if(countdown.total_centiseconds == 0) {
                countdown_running = false;
                countdown_expired = true;
            }
        }
        
        display_update = true;
    }
}

// Timer1 - 디스플레이 멀티플렉싱 (2ms)
ISR(TIMER1_COMPA_vect) {
    // 현재 디지트 끄기
    PORTC &= ~(0x0F);  // PC0-PC3을 디지트 선택으로 사용
    
    // 다음 디지트 선택
    display_digit = (display_digit + 1) % 4;
    
    // 새 디지트 데이터 출력
    uint8_t pattern = pgm_read_byte(&seven_seg_patterns[display_data[display_digit] & 0x0F]);
    PORTD = (PORTD & 0x0F) | (pattern & 0xF0);
    
    // 해당 디지트 활성화
    PORTC |= (1 << display_digit);
}

// Timer2 - 고속 LED 제어 (100μs)
ISR(TIMER2_COMPA_vect) {
    static uint16_t counter_100us = 0;
    
    microsecond_counter += 100;
    counter_100us++;
    
    // 다양한 주파수 LED 제어
    if(counter_100us % 50000 == 0) {      // 1Hz (0.5초 ON/OFF)
        PORTB ^= (1 << LED_1HZ);
    }
    if(counter_100us % 5000 == 0) {       // 10Hz
        PORTB ^= (1 << LED_10HZ);
    }
    if(counter_100us % 500 == 0) {        // 100Hz
        PORTB ^= (1 << LED_100HZ);
    }
    if(counter_100us % 50 == 0) {         // 1kHz
        PORTB ^= (1 << LED_1KHZ);
    }
    
    // 카운터 오버플로우 방지
    if(counter_100us >= 50000) {
        counter_100us = 0;
    }
}

// 외부 인터럽트 - 모드 변경 버튼
ISR(INT0_vect) {
    static uint32_t last_press_time = 0;
    
    // 디바운싱 (50ms)
    if((microsecond_counter - last_press_time) > 50000) {
        current_mode = (current_mode + 1) % MODE_COUNT;
        button_pressed = true;
        last_press_time = microsecond_counter;
    }
}

// 외부 인터럽트 - 설정/시작 버튼
ISR(INT1_vect) {
    static uint32_t last_press_time = 0;
    
    // 디바운싱 (50ms)
    if((microsecond_counter - last_press_time) > 50000) {
        switch(current_mode) {
            case MODE_STOPWATCH:
                stopwatch_running = !stopwatch_running;
                break;
                
            case MODE_COUNTDOWN:
                if(countdown.total_centiseconds > 0) {
                    countdown_running = !countdown_running;
                    countdown_expired = false;
                }
                break;
                
            default:
                break;
        }
        
        button_pressed = true;
        last_press_time = microsecond_counter;
    }
}

// 함수 정의

void system_init(void) {
    // GPIO 초기화
    // LED 출력 (PB0-PB3)
    DDRB |= 0x0F;
    PORTB &= ~0x0F;
    
    // 7-segment 출력 (PD4-PD7)
    DDRD |= 0xF0;
    PORTD &= ~0xF0;
    
    // 디지트 선택 출력 (PC0-PC3)
    DDRC |= 0x0F;
    PORTC &= ~0x0F;
    
    // 버튼 입력 설정
    DDRD &= ~((1 << BTN_MODE) | (1 << BTN_SET));  // PD2, PD3 입력
    PORTD |= (1 << BTN_MODE) | (1 << BTN_SET);    // 풀업 활성화
    
    DDRC &= ~((1 << BTN_START) | (1 << BTN_RESET));  // PC0, PC1 입력
    PORTC |= (1 << BTN_START) | (1 << BTN_RESET);    // 풀업 활성화
}

void timers_init(void) {
    // Timer0: 1ms 시스템 클럭
    TCCR0A = (1 << WGM01);                    // CTC 모드
    TCCR0B = (1 << CS01) | (1 << CS00);       // 프리스케일러 64
    OCR0A = TIMER_1MS;
    TIMSK0 = (1 << OCIE0A);
    
    // Timer1: 2ms 디스플레이 멀티플렉싱
    TCCR1B = (1 << WGM12);                    // CTC 모드
    TCCR1B |= (1 << CS11) | (1 << CS10);      // 프리스케일러 64
    OCR1A = 499;  // 2ms
    TIMSK1 = (1 << OCIE1A);
    
    // Timer2: 100μs 고속 타이밍
    TCCR2A = (1 << WGM21);                    // CTC 모드
    TCCR2B = (1 << CS22);                     // 프리스케일러 64
    OCR2A = TIMER_100US;
    TIMSK2 = (1 << OCIE2A);
}

void external_interrupt_init(void) {
    // INT0, INT1을 하강 엣지에서 트리거
    EICRA = (1 << ISC01) | (1 << ISC11);
    EIMSK = (1 << INT0) | (1 << INT1);
}

void update_display(void) {
    switch(current_mode) {
        case MODE_CLOCK:
            display_data[0] = system_clock.hours / 10;
            display_data[1] = system_clock.hours % 10;
            display_data[2] = system_clock.minutes / 10;
            display_data[3] = system_clock.minutes % 10;
            break;
            
        case MODE_STOPWATCH:
            display_data[0] = stopwatch.minutes / 10;
            display_data[1] = stopwatch.minutes % 10;
            display_data[2] = stopwatch.seconds / 10;
            display_data[3] = stopwatch.seconds % 10;
            break;
            
        case MODE_COUNTDOWN:
            if(countdown.hours > 0) {
                display_data[0] = countdown.hours / 10;
                display_data[1] = countdown.hours % 10;
                display_data[2] = countdown.minutes / 10;
                display_data[3] = countdown.minutes % 10;
            } else {
                display_data[0] = countdown.minutes / 10;
                display_data[1] = countdown.minutes % 10;
                display_data[2] = countdown.seconds / 10;
                display_data[3] = countdown.seconds % 10;
            }
            break;
            
        case MODE_FREQ_GEN:
            // 주파수 값을 16진수로 표시
            uint16_t freq = 1000;  // 예시: 1kHz
            display_data[0] = (freq >> 12) & 0x0F;
            display_data[1] = (freq >> 8) & 0x0F;
            display_data[2] = (freq >> 4) & 0x0F;
            display_data[3] = freq & 0x0F;
            break;
    }
}

// 정밀 마이크로초 지연 함수
void delay_us_precise(uint16_t microseconds) {
    uint32_t start_time = microsecond_counter;
    
    while((microsecond_counter - start_time) < microseconds) {
        // 타이머 인터럽트 기반 대기
    }
}

// 주파수 측정 함수
uint32_t measure_frequency(uint8_t pin, uint16_t measurement_time_ms) {
    volatile uint32_t edge_count = 0;
    bool last_state = (PINB & (1 << pin)) != 0;
    uint32_t start_time = microsecond_counter;
    uint32_t end_time = start_time + (measurement_time_ms * 1000UL);
    
    while(microsecond_counter < end_time) {
        bool current_state = (PINB & (1 << pin)) != 0;
        
        if(current_state != last_state) {
            if(current_state) {  // 상승 엣지
                edge_count++;
            }
            last_state = current_state;
        }
    }
    
    // Hz 계산
    return (edge_count * 1000UL) / measurement_time_ms;
}

// 스톱워치 리셋
void reset_stopwatch(void) {
    stopwatch.hours = 0;
    stopwatch.minutes = 0;
    stopwatch.seconds = 0;
    stopwatch.centiseconds = 0;
    stopwatch.total_centiseconds = 0;
    stopwatch_running = false;
}

// 카운트다운 타이머 리셋
void reset_countdown(void) {
    countdown.hours = 0;
    countdown.minutes = 5;  // 5분으로 리셋
    countdown.seconds = 0;
    countdown.centiseconds = 0;
    countdown.total_centiseconds = 30000;  // 5분 = 30000 센티초
    countdown_running = false;
    countdown_expired = false;
}

int main(void) {
    system_init();
    timers_init();
    external_interrupt_init();
    
    sei();  // 전역 인터럽트 활성화
    
    while(1) {
        // 디스플레이 업데이트
        if(display_update) {
            display_update = false;
            update_display();
        }
        
        // 버튼 처리 (폴링)
        static bool last_start_state = false;
        static bool last_reset_state = false;
        
        bool start_pressed = !(PINC & (1 << BTN_START));
        bool reset_pressed = !(PINC & (1 << BTN_RESET));
        
        // 스타트 버튼 처리
        if(start_pressed && !last_start_state) {
            delay_us_precise(20000);  // 20ms 디바운싱
            
            if(!(PINC & (1 << BTN_START))) {  // 여전히 눌려있는지 확인
                switch(current_mode) {
                    case MODE_STOPWATCH:
                        stopwatch_running = !stopwatch_running;
                        break;
                        
                    case MODE_COUNTDOWN:
                        if(countdown.total_centiseconds > 0) {
                            countdown_running = !countdown_running;
                            countdown_expired = false;
                        }
                        break;
                        
                    default:
                        break;
                }
            }
        }
        last_start_state = start_pressed;
        
        // 리셋 버튼 처리
        if(reset_pressed && !last_reset_state) {
            delay_us_precise(20000);  // 20ms 디바운싱
            
            if(!(PINC & (1 << BTN_RESET))) {  // 여전히 눌려있는지 확인
                switch(current_mode) {
                    case MODE_STOPWATCH:
                        reset_stopwatch();
                        break;
                        
                    case MODE_COUNTDOWN:
                        reset_countdown();
                        break;
                        
                    default:
                        break;
                }
            }
        }
        last_reset_state = reset_pressed;
        
        // 카운트다운 만료 알림
        if(countdown_expired) {
            // LED로 알림 (빠른 깜빡임)
            static uint8_t alarm_counter = 0;
            if(++alarm_counter > 10) {
                alarm_counter = 0;
                PORTB ^= 0x0F;  // 모든 LED 토글
            }
        }
        
        // 모드별 특별 처리
        switch(current_mode) {
            case MODE_FREQ_GEN:
                // 주파수 생성 모드에서는 LED를 다른 용도로 사용
                static uint16_t freq_counter = 0;
                if(++freq_counter >= 1000) {
                    freq_counter = 0;
                    
                    // 각 LED에서 다른 주파수 측정 (예시)
                    uint32_t freq_1hz = measure_frequency(LED_1HZ, 2000);
                    uint32_t freq_10hz = measure_frequency(LED_10HZ, 1000);
                    
                    // 측정 결과를 디스플레이에 반영 (실제로는 UART 등으로 출력)
                }
                break;
                
            default:
                break;
        }
    }
    
    return 0;
}