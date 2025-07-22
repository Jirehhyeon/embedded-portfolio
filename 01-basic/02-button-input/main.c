/*
 * Button Input with Debouncing
 * 
 * Hardware: ATmega328P
 * Buttons: PD2 (INT0), PD3 (INT1)
 * LED: PB5 (Arduino Pin 13)
 * 
 * This program demonstrates:
 * - Digital input reading with internal pull-up
 * - Software debouncing using state machine
 * - Interrupt-based button handling
 * - Edge detection (rising/falling)
 * 
 * Author: [Your Name]
 * Date: 2024-01-20
 */

#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>
#include <stdbool.h>
#include <stdint.h>

// 핀 정의
#define BUTTON1_PIN     PD2     // INT0
#define BUTTON1_PORT    PORTD
#define BUTTON1_DDR     DDRD
#define BUTTON1_PINS    PIND

#define BUTTON2_PIN     PD3     // INT1
#define BUTTON2_PORT    PORTD
#define BUTTON2_DDR     DDRD
#define BUTTON2_PINS    PIND

#define LED_PIN         PB5
#define LED_PORT        PORTB
#define LED_DDR         DDRB

// 디바운싱 설정
#define DEBOUNCE_TIME   50      // 50ms
#define LONG_PRESS_TIME 1000    // 1초

// 버튼 상태 정의
typedef enum {
    BTN_IDLE,
    BTN_DEBOUNCE,
    BTN_PRESSED,
    BTN_HELD,
    BTN_RELEASE_DEBOUNCE
} button_state_t;

// 버튼 이벤트 타입
typedef enum {
    BTN_EVENT_NONE,
    BTN_EVENT_PRESS,
    BTN_EVENT_RELEASE,
    BTN_EVENT_LONG_PRESS,
    BTN_EVENT_DOUBLE_CLICK
} button_event_t;

// 버튼 구조체
typedef struct {
    button_state_t state;
    uint16_t debounce_counter;
    uint16_t hold_counter;
    uint16_t click_counter;
    uint16_t double_click_timer;
    bool current_value;
    bool last_value;
} button_t;

// 전역 변수
volatile bool timer_flag = false;
button_t button1 = {0};
button_t button2 = {0};
uint8_t led_mode = 0;  // 0: OFF, 1: ON, 2: SLOW_BLINK, 3: FAST_BLINK

// 함수 프로토타입
void system_init(void);
void timer0_init(void);
button_event_t process_button(button_t *btn, bool current_state);
void handle_button_event(uint8_t button_num, button_event_t event);
void update_led(void);

// Timer0 Compare Match A 인터럽트 (1ms 주기)
ISR(TIMER0_COMPA_vect) {
    timer_flag = true;
}

int main(void) {
    system_init();
    timer0_init();
    
    uint16_t blink_counter = 0;
    
    while(1) {
        // 1ms 타이머 플래그 확인
        if(timer_flag) {
            timer_flag = false;
            
            // 버튼 상태 읽기
            bool btn1_state = !(BUTTON1_PINS & (1 << BUTTON1_PIN));
            bool btn2_state = !(BUTTON2_PINS & (1 << BUTTON2_PIN));
            
            // 버튼 처리
            button_event_t event1 = process_button(&button1, btn1_state);
            button_event_t event2 = process_button(&button2, btn2_state);
            
            // 이벤트 처리
            if(event1 != BTN_EVENT_NONE) {
                handle_button_event(1, event1);
            }
            if(event2 != BTN_EVENT_NONE) {
                handle_button_event(2, event2);
            }
            
            // LED 업데이트
            blink_counter++;
            if(led_mode == 2 && blink_counter >= 500) {  // Slow blink
                LED_PORT ^= (1 << LED_PIN);
                blink_counter = 0;
            } else if(led_mode == 3 && blink_counter >= 100) {  // Fast blink
                LED_PORT ^= (1 << LED_PIN);
                blink_counter = 0;
            }
        }
    }
    
    return 0;
}

/**
 * 시스템 초기화
 */
void system_init(void) {
    // LED 출력 설정
    LED_DDR |= (1 << LED_PIN);
    LED_PORT &= ~(1 << LED_PIN);  // 초기값 OFF
    
    // 버튼 입력 설정 (내부 풀업 활성화)
    BUTTON1_DDR &= ~(1 << BUTTON1_PIN);
    BUTTON1_PORT |= (1 << BUTTON1_PIN);
    
    BUTTON2_DDR &= ~(1 << BUTTON2_PIN);
    BUTTON2_PORT |= (1 << BUTTON2_PIN);
    
    // 전역 인터럽트 활성화
    sei();
}

/**
 * Timer0 초기화 (1ms 인터럽트)
 */
void timer0_init(void) {
    // CTC 모드
    TCCR0A = (1 << WGM01);
    
    // 프리스케일러 64
    TCCR0B = (1 << CS01) | (1 << CS00);
    
    // 1ms를 위한 비교값 (16MHz / 64 / 1000 = 250)
    OCR0A = 249;
    
    // 비교 매치 인터럽트 활성화
    TIMSK0 = (1 << OCIE0A);
}

/**
 * 버튼 상태 처리 (상태 머신)
 */
button_event_t process_button(button_t *btn, bool current_state) {
    button_event_t event = BTN_EVENT_NONE;
    
    btn->current_value = current_state;
    
    // 더블클릭 타이머 처리
    if(btn->double_click_timer > 0) {
        btn->double_click_timer--;
    }
    
    switch(btn->state) {
        case BTN_IDLE:
            if(btn->current_value && !btn->last_value) {  // Rising edge
                btn->state = BTN_DEBOUNCE;
                btn->debounce_counter = 0;
            }
            break;
            
        case BTN_DEBOUNCE:
            btn->debounce_counter++;
            if(btn->debounce_counter >= DEBOUNCE_TIME) {
                if(btn->current_value) {
                    btn->state = BTN_PRESSED;
                    btn->hold_counter = 0;
                    event = BTN_EVENT_PRESS;
                    
                    // 더블클릭 체크
                    if(btn->double_click_timer > 0) {
                        event = BTN_EVENT_DOUBLE_CLICK;
                        btn->double_click_timer = 0;
                    } else {
                        btn->double_click_timer = 300;  // 300ms 내에 다시 클릭
                    }
                } else {
                    btn->state = BTN_IDLE;
                }
            }
            break;
            
        case BTN_PRESSED:
            if(!btn->current_value) {  // 버튼 뗌
                btn->state = BTN_RELEASE_DEBOUNCE;
                btn->debounce_counter = 0;
            } else {
                btn->hold_counter++;
                if(btn->hold_counter >= LONG_PRESS_TIME) {
                    btn->state = BTN_HELD;
                    event = BTN_EVENT_LONG_PRESS;
                }
            }
            break;
            
        case BTN_HELD:
            if(!btn->current_value) {
                btn->state = BTN_RELEASE_DEBOUNCE;
                btn->debounce_counter = 0;
            }
            break;
            
        case BTN_RELEASE_DEBOUNCE:
            btn->debounce_counter++;
            if(btn->debounce_counter >= DEBOUNCE_TIME) {
                if(!btn->current_value) {
                    btn->state = BTN_IDLE;
                    event = BTN_EVENT_RELEASE;
                } else {
                    btn->state = BTN_PRESSED;
                }
            }
            break;
    }
    
    btn->last_value = btn->current_value;
    return event;
}

/**
 * 버튼 이벤트 처리
 */
void handle_button_event(uint8_t button_num, button_event_t event) {
    switch(button_num) {
        case 1:  // Button 1
            switch(event) {
                case BTN_EVENT_PRESS:
                    // LED 모드 변경
                    led_mode = (led_mode + 1) % 4;
                    update_led();
                    break;
                    
                case BTN_EVENT_LONG_PRESS:
                    // 모든 모드 리셋
                    led_mode = 0;
                    update_led();
                    break;
                    
                case BTN_EVENT_DOUBLE_CLICK:
                    // 빠른 점멸 모드
                    led_mode = 3;
                    break;
                    
                default:
                    break;
            }
            break;
            
        case 2:  // Button 2
            switch(event) {
                case BTN_EVENT_PRESS:
                    // LED 토글
                    LED_PORT ^= (1 << LED_PIN);
                    break;
                    
                case BTN_EVENT_LONG_PRESS:
                    // 특수 패턴 (SOS)
                    for(uint8_t i = 0; i < 3; i++) {
                        // S
                        for(uint8_t j = 0; j < 3; j++) {
                            LED_PORT |= (1 << LED_PIN);
                            _delay_ms(100);
                            LED_PORT &= ~(1 << LED_PIN);
                            _delay_ms(100);
                        }
                        _delay_ms(200);
                        
                        // O
                        for(uint8_t j = 0; j < 3; j++) {
                            LED_PORT |= (1 << LED_PIN);
                            _delay_ms(300);
                            LED_PORT &= ~(1 << LED_PIN);
                            _delay_ms(100);
                        }
                        _delay_ms(200);
                        
                        // S
                        for(uint8_t j = 0; j < 3; j++) {
                            LED_PORT |= (1 << LED_PIN);
                            _delay_ms(100);
                            LED_PORT &= ~(1 << LED_PIN);
                            _delay_ms(100);
                        }
                        _delay_ms(500);
                    }
                    break;
                    
                default:
                    break;
            }
            break;
    }
}

/**
 * LED 상태 업데이트
 */
void update_led(void) {
    switch(led_mode) {
        case 0:  // OFF
            LED_PORT &= ~(1 << LED_PIN);
            break;
            
        case 1:  // ON
            LED_PORT |= (1 << LED_PIN);
            break;
            
        case 2:  // SLOW BLINK
        case 3:  // FAST BLINK
            // 타이머 인터럽트에서 처리
            break;
    }
}