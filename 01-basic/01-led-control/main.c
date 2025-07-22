/*
 * LED Control - Basic GPIO Example
 * 
 * Hardware: ATmega328P
 * Pin: PB5 (Arduino Pin 13)
 * 
 * This program demonstrates basic GPIO control by blinking an LED
 * connected to PB5. It directly manipulates AVR registers without
 * using Arduino libraries.
 * 
 * Author: [Your Name]
 * Date: 2024-01-20
 */

#include <avr/io.h>
#include <util/delay.h>

// LED 연결 핀 정의
#define LED_PIN     PB5
#define LED_PORT    PORTB
#define LED_DDR     DDRB

// LED 제어 매크로
#define LED_ON()    (LED_PORT |= (1 << LED_PIN))
#define LED_OFF()   (LED_PORT &= ~(1 << LED_PIN))
#define LED_TOGGLE() (LED_PORT ^= (1 << LED_PIN))

// 함수 프로토타입
void led_init(void);
void blink_pattern(uint8_t times, uint16_t on_time, uint16_t off_time);
void morse_code_sos(void);

int main(void) {
    // LED 초기화
    led_init();
    
    // 메인 루프
    while(1) {
        // 기본 깜빡임 패턴
        blink_pattern(3, 100, 100);
        _delay_ms(500);
        
        // SOS 모스 부호
        morse_code_sos();
        _delay_ms(2000);
        
        // 천천히 깜빡임
        blink_pattern(2, 500, 500);
        _delay_ms(1000);
    }
    
    return 0;
}

/**
 * LED 초기화 함수
 * PB5를 출력으로 설정하고 LED를 끔
 */
void led_init(void) {
    // PB5를 출력으로 설정
    LED_DDR |= (1 << LED_PIN);
    
    // LED 초기 상태는 OFF
    LED_OFF();
}

/**
 * LED 깜빡임 패턴 생성
 * @param times: 깜빡임 횟수
 * @param on_time: LED ON 시간 (ms)
 * @param off_time: LED OFF 시간 (ms)
 */
void blink_pattern(uint8_t times, uint16_t on_time, uint16_t off_time) {
    for(uint8_t i = 0; i < times; i++) {
        LED_ON();
        
        // on_time을 10ms 단위로 분할하여 딜레이
        for(uint16_t j = 0; j < on_time/10; j++) {
            _delay_ms(10);
        }
        
        LED_OFF();
        
        // off_time을 10ms 단위로 분할하여 딜레이
        for(uint16_t j = 0; j < off_time/10; j++) {
            _delay_ms(10);
        }
    }
}

/**
 * SOS 모스 부호 출력
 * S: 짧게 3번 (・・・)
 * O: 길게 3번 (－－－)
 * S: 짧게 3번 (・・・)
 */
void morse_code_sos(void) {
    // S - 짧은 신호 3번
    for(uint8_t i = 0; i < 3; i++) {
        LED_ON();
        _delay_ms(100);
        LED_OFF();
        _delay_ms(100);
    }
    
    _delay_ms(300); // 글자 간 간격
    
    // O - 긴 신호 3번
    for(uint8_t i = 0; i < 3; i++) {
        LED_ON();
        _delay_ms(300);
        LED_OFF();
        _delay_ms(100);
    }
    
    _delay_ms(300); // 글자 간 간격
    
    // S - 짧은 신호 3번
    for(uint8_t i = 0; i < 3; i++) {
        LED_ON();
        _delay_ms(100);
        LED_OFF();
        _delay_ms(100);
    }
}