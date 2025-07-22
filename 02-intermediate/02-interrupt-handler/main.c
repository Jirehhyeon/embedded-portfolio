/*
 * Advanced Interrupt Handler System
 * 
 * Hardware: ATmega328P
 * Features: Multi-source interrupt handling with priority management
 * 
 * Interrupt Sources:
 * - INT0 (PD2): Rotary encoder A phase
 * - INT1 (PD3): Rotary encoder B phase / Emergency stop
 * - PCINT20 (PD4): Ultrasonic sensor echo
 * - Timer0 Compare A: System tick (1ms)
 * - Timer1 Compare A: Real-time control loop (1ms)  
 * - ADC Complete: Temperature sensor reading
 * 
 * This program demonstrates:
 * - Interrupt vector table utilization
 * - Interrupt priority and nesting
 * - Critical section management
 * - Real-time system design
 * - Atomic operations
 * - Context switching
 * 
 * Author: [Your Name]
 * Date: 2024-01-20
 */

#include <avr/io.h>
#include <avr/interrupt.h>
#include <avr/sleep.h>
#include <util/delay.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

// Hardware pin definitions
#define ENCODER_A_PIN   PD2  // INT0
#define ENCODER_B_PIN   PD3  // INT1  
#define ECHO_PIN        PD4  // PCINT20
#define TRIGGER_PIN     PB1
#define BUZZER_PIN      PD6  // OC0A
#define PWM_LED_PIN     PD5  // OC0B

#define STATUS_LED      PB0
#define ERROR_LED       PB2
#define WARNING_LED     PB3
#define ACTIVITY_LED    PB4

// System constants
#define MAX_INTERRUPT_NESTING   4
#define CRITICAL_TEMP_THRESHOLD 50
#define WARNING_TEMP_THRESHOLD  40
#define PROXIMITY_THRESHOLD     10
#define EMERGENCY_DEBOUNCE_MS   100

// Interrupt priority levels
typedef enum {
    PRIORITY_CRITICAL = 0,   // Emergency stop, safety
    PRIORITY_HIGH = 1,       // Real-time control
    PRIORITY_NORMAL = 2,     // Sensor I/O
    PRIORITY_LOW = 3         // Background tasks
} interrupt_priority_t;

// System state structure
typedef struct {
    volatile int16_t encoder_position;
    volatile uint16_t distance_cm;
    volatile uint16_t temperature_c;
    volatile uint32_t system_ticks;
    volatile uint8_t error_code;
    volatile bool emergency_stop_active;
} system_state_t;

// Interrupt statistics
typedef struct {
    volatile uint32_t int0_count;
    volatile uint32_t int1_count;
    volatile uint32_t pcint_count;
    volatile uint32_t timer0_count;
    volatile uint32_t timer1_count;
    volatile uint32_t adc_count;
    volatile uint32_t total_interrupts;
    volatile uint8_t max_nesting_level;
} interrupt_stats_t;

// Interrupt flags structure
typedef struct {
    volatile bool encoder_changed : 1;
    volatile bool ultrasonic_ready : 1;
    volatile bool adc_complete : 1;
    volatile bool timer_tick : 1;
    volatile bool emergency_stop : 1;
    volatile bool system_error : 1;
    volatile bool temperature_warning : 1;
    volatile bool proximity_warning : 1;
} interrupt_flags_t;

// PID controller structure
typedef struct {
    float kp, ki, kd;
    float setpoint;
    float integral;
    float last_error;
    float output;
} pid_controller_t;

// Global variables
system_state_t system_state = {0};
interrupt_stats_t interrupt_stats = {0};
interrupt_flags_t interrupt_flags = {0};
pid_controller_t pid_controller = {2.0, 0.1, 0.5, 0.0, 0.0, 0.0, 0.0};

volatile interrupt_priority_t current_priority = PRIORITY_LOW;
volatile uint8_t interrupt_nesting_level = 0;

// Critical section macros
#define ENTER_CRITICAL() \
    uint8_t sreg_save = SREG; \
    cli();

#define EXIT_CRITICAL() \
    SREG = sreg_save;

// Interrupt priority management
#define ISR_ENTER(priority) \
    interrupt_stats.total_interrupts++; \
    interrupt_nesting_level++; \
    if(interrupt_nesting_level > interrupt_stats.max_nesting_level) { \
        interrupt_stats.max_nesting_level = interrupt_nesting_level; \
    } \
    interrupt_priority_t old_priority = current_priority; \
    if(priority < current_priority) { \
        current_priority = priority; \
        sei(); \
    }

#define ISR_EXIT() \
    cli(); \
    current_priority = old_priority; \
    interrupt_nesting_level--; \

// External Interrupt INT0 - Rotary Encoder A Phase
ISR(INT0_vect) {
    ISR_ENTER(PRIORITY_NORMAL);
    interrupt_stats.int0_count++;
    
    // Read encoder state
    bool a_state = (PIND & (1 << ENCODER_A_PIN)) != 0;
    bool b_state = (PIND & (1 << ENCODER_B_PIN)) != 0;
    
    // Determine direction (quadrature decoding)
    if(a_state == b_state) {
        system_state.encoder_position++;
    } else {
        system_state.encoder_position--;
    }
    
    // Limit range
    if(system_state.encoder_position > 255) system_state.encoder_position = 255;
    if(system_state.encoder_position < -255) system_state.encoder_position = -255;
    
    interrupt_flags.encoder_changed = true;
    
    ISR_EXIT();
}

// External Interrupt INT1 - Emergency Stop / Encoder B Phase
ISR(INT1_vect) {
    ISR_ENTER(PRIORITY_CRITICAL);
    interrupt_stats.int1_count++;
    
    static uint32_t last_interrupt_time = 0;
    
    // Emergency stop detection (rapid successive interrupts)
    if((system_state.system_ticks - last_interrupt_time) < EMERGENCY_DEBOUNCE_MS) {
        // Immediate safety actions
        OCR0A = 0;  // Stop buzzer
        OCR0B = 0;  // Stop PWM LED
        OCR1A = 0;  // Stop any motor PWM
        
        system_state.emergency_stop_active = true;
        system_state.error_code = 0x01;
        interrupt_flags.emergency_stop = true;
        
        // Set error LED
        PORTB |= (1 << ERROR_LED);
    }
    
    last_interrupt_time = system_state.system_ticks;
    
    ISR_EXIT();
}

// Pin Change Interrupt - Ultrasonic Sensor Echo
ISR(PCINT2_vect) {
    ISR_ENTER(PRIORITY_HIGH);
    interrupt_stats.pcint_count++;
    
    static uint32_t echo_start_time = 0;
    static bool echo_active = false;
    
    bool echo_state = (PIND & (1 << ECHO_PIN)) != 0;
    
    if(echo_state && !echo_active) {
        // Rising edge - start measurement
        echo_start_time = system_state.system_ticks * 1000; // Convert to microseconds
        echo_active = true;
    } else if(!echo_state && echo_active) {
        // Falling edge - calculate distance
        uint32_t echo_duration = (system_state.system_ticks * 1000) - echo_start_time;
        
        // Distance calculation: time_us / 58 = distance_cm
        system_state.distance_cm = echo_duration / 58;
        
        // Limit reasonable range
        if(system_state.distance_cm > 400) system_state.distance_cm = 400;
        
        echo_active = false;
        interrupt_flags.ultrasonic_ready = true;
        
        // Proximity warning check
        if(system_state.distance_cm < PROXIMITY_THRESHOLD) {
            interrupt_flags.proximity_warning = true;
            system_state.error_code = 0x02;
        }
    }
    
    ISR_EXIT();
}

// Timer0 Compare Match A - System Tick (1ms)
ISR(TIMER0_COMPA_vect) {
    ISR_ENTER(PRIORITY_HIGH);
    interrupt_stats.timer0_count++;
    
    system_state.system_ticks++;
    interrupt_flags.timer_tick = true;
    
    // Periodic ADC trigger (every 100ms)
    if(system_state.system_ticks % 100 == 0) {
        ADCSRA |= (1 << ADSC);
    }
    
    // Activity LED heartbeat (every 1 second)
    if(system_state.system_ticks % 1000 == 0) {
        PORTB ^= (1 << ACTIVITY_LED);
    }
    
    // Trigger next ultrasonic measurement (every 50ms)
    static uint32_t last_ultrasonic_trigger = 0;
    if((system_state.system_ticks - last_ultrasonic_trigger) >= 50) {
        // Trigger ultrasonic sensor
        PORTB |= (1 << TRIGGER_PIN);
        _delay_us(10);
        PORTB &= ~(1 << TRIGGER_PIN);
        last_ultrasonic_trigger = system_state.system_ticks;
    }
    
    ISR_EXIT();
}

// Timer1 Compare Match A - Real-time Control Loop (1ms)
ISR(TIMER1_COMPA_vect) {
    ISR_ENTER(PRIORITY_HIGH);
    interrupt_stats.timer1_count++;
    
    // PID controller for distance control
    float setpoint = (float)abs(system_state.encoder_position);
    float process_value = (float)system_state.distance_cm;
    float error = setpoint - process_value;
    
    // PID calculation
    pid_controller.integral += error * 0.001f; // dt = 1ms
    float derivative = (error - pid_controller.last_error) / 0.001f;
    
    // Anti-windup
    if(pid_controller.integral > 100.0f) pid_controller.integral = 100.0f;
    if(pid_controller.integral < -100.0f) pid_controller.integral = -100.0f;
    
    pid_controller.output = (pid_controller.kp * error) +
                           (pid_controller.ki * pid_controller.integral) +
                           (pid_controller.kd * derivative);
    
    // Output limiting
    if(pid_controller.output > 255.0f) pid_controller.output = 255.0f;
    if(pid_controller.output < 0.0f) pid_controller.output = 0.0f;
    
    // Apply PID output to PWM (only if not in emergency stop)
    if(!system_state.emergency_stop_active) {
        OCR0B = (uint8_t)pid_controller.output;
    }
    
    pid_controller.last_error = error;
    
    ISR_EXIT();
}

// ADC Conversion Complete - Temperature Reading
ISR(ADC_vect) {
    ISR_ENTER(PRIORITY_NORMAL);
    interrupt_stats.adc_count++;
    
    // LM35 temperature calculation: 10mV/°C, 5V reference
    // Temperature = ADC * 5000mV / 1024 / 10mV/°C
    system_state.temperature_c = (ADC * 500UL) / 1024;
    
    interrupt_flags.adc_complete = true;
    
    // Temperature warnings
    if(system_state.temperature_c >= CRITICAL_TEMP_THRESHOLD) {
        system_state.error_code = 0x03;
        interrupt_flags.system_error = true;
    } else if(system_state.temperature_c >= WARNING_TEMP_THRESHOLD) {
        interrupt_flags.temperature_warning = true;
    }
    
    ISR_EXIT();
}

// System initialization functions
void init_gpio(void) {
    // LED outputs
    DDRB |= (1 << STATUS_LED) | (1 << ERROR_LED) | (1 << WARNING_LED) | (1 << ACTIVITY_LED);
    PORTB &= ~((1 << STATUS_LED) | (1 << ERROR_LED) | (1 << WARNING_LED) | (1 << ACTIVITY_LED));
    
    // Ultrasonic sensor
    DDRB |= (1 << TRIGGER_PIN);   // Trigger output
    DDRD &= ~(1 << ECHO_PIN);     // Echo input
    
    // Rotary encoder inputs with pull-ups
    DDRD &= ~((1 << ENCODER_A_PIN) | (1 << ENCODER_B_PIN));
    PORTD |= (1 << ENCODER_A_PIN) | (1 << ENCODER_B_PIN);
    
    // PWM outputs
    DDRD |= (1 << BUZZER_PIN) | (1 << PWM_LED_PIN);
}

void init_timers(void) {
    // Timer0: Fast PWM for buzzer and LED + CTC for system tick
    TCCR0A = (1 << WGM01) | (1 << WGM00) | (1 << COM0A1) | (1 << COM0B1);
    TCCR0B = (1 << CS01) | (1 << CS00);  // Prescaler 64
    OCR0A = 0;  // Buzzer off initially
    OCR0B = 0;  // LED off initially
    TIMSK0 = (1 << OCIE0A);
    
    // Timer1: CTC mode for real-time control (1ms)
    TCCR1B = (1 << WGM12) | (1 << CS11) | (1 << CS10);  // CTC, prescaler 64
    OCR1A = 249;  // 1ms period
    TIMSK1 = (1 << OCIE1A);
    
    // Timer2: Available for future use
}

void init_adc(void) {
    // ADC setup for temperature sensor on PC0
    ADMUX = (1 << REFS0);  // AVCC reference, ADC0 channel
    ADCSRA = (1 << ADEN) |                    // Enable ADC
             (1 << ADIE) |                    // Enable interrupt
             (1 << ADPS2) | (1 << ADPS1) | (1 << ADPS0);  // Prescaler 128
}

void init_interrupts(void) {
    // External interrupts
    EICRA = (1 << ISC00) | (1 << ISC10);  // Any edge for both INT0 and INT1
    EIMSK = (1 << INT0) | (1 << INT1);    // Enable INT0 and INT1
    
    // Pin change interrupt for echo pin
    PCICR = (1 << PCIE2);        // Enable PCINT2 group
    PCMSK2 = (1 << PCINT20);     // Enable PD4 (echo pin)
    
    // Set initial priority
    current_priority = PRIORITY_LOW;
    interrupt_nesting_level = 0;
}

// Utility functions
void atomic_increment_32(volatile uint32_t *value) {
    ENTER_CRITICAL();
    (*value)++;
    EXIT_CRITICAL();
}

bool atomic_test_and_clear_flag(volatile bool *flag) {
    bool result;
    ENTER_CRITICAL();
    result = *flag;
    *flag = false;
    EXIT_CRITICAL();
    return result;
}

void handle_emergency_stop(void) {
    // All safety-critical shutdowns
    OCR0A = 0;  // Stop buzzer
    OCR0B = 0;  // Stop LED PWM
    OCR1A = 0;  // Stop any motor control
    
    // Visual and audible alarm
    PORTB |= (1 << ERROR_LED);
    
    // Emergency beep pattern
    for(uint8_t i = 0; i < 5; i++) {
        OCR0A = 128;
        _delay_ms(100);
        OCR0A = 0;
        _delay_ms(100);
    }
    
    // Clear emergency flag
    interrupt_flags.emergency_stop = false;
}

void update_system_leds(void) {
    // Status LED based on encoder position
    if(system_state.encoder_position > 0) {
        PORTB |= (1 << STATUS_LED);
    } else {
        PORTB &= ~(1 << STATUS_LED);
    }
    
    // Warning LED for temperature
    if(interrupt_flags.temperature_warning) {
        PORTB |= (1 << WARNING_LED);
        interrupt_flags.temperature_warning = false;
    } else {
        PORTB &= ~(1 << WARNING_LED);
    }
    
    // Proximity warning (flashing error LED)
    static uint8_t proximity_flash_counter = 0;
    if(interrupt_flags.proximity_warning) {
        if(++proximity_flash_counter > 10) {
            PORTB ^= (1 << ERROR_LED);
            proximity_flash_counter = 0;
        }
    }
}

void process_sensor_data(void) {
    // Distance-based buzzer control
    if(system_state.distance_cm < 50 && !system_state.emergency_stop_active) {
        // Closer distance = higher frequency
        uint8_t buzz_intensity = 255 - (system_state.distance_cm * 5);
        OCR0A = buzz_intensity;
    } else {
        OCR0A = 0;
    }
}

void print_system_status(void) {
    // This would typically go to UART for debugging
    // For now, just update some diagnostic variables
    static uint32_t last_print_time = 0;
    
    if((system_state.system_ticks - last_print_time) >= 1000) {
        last_print_time = system_state.system_ticks;
        
        // System status could be displayed on LCD or sent via UART
        // printf("ENC:%d DIST:%dcm TEMP:%dC ERR:0x%02X\n",
        //        system_state.encoder_position,
        //        system_state.distance_cm, 
        //        system_state.temperature_c,
        //        system_state.error_code);
        
        // Reset error code after reporting
        system_state.error_code = 0;
    }
}

int main(void) {
    // Initialize all subsystems
    init_gpio();
    init_timers();
    init_adc();
    init_interrupts();
    
    // Enable global interrupts
    sei();
    
    // Initial ultrasonic trigger
    PORTB |= (1 << TRIGGER_PIN);
    _delay_us(10);
    PORTB &= ~(1 << TRIGGER_PIN);
    
    // Main application loop
    while(1) {
        // Critical priority: Emergency stop
        if(atomic_test_and_clear_flag(&interrupt_flags.emergency_stop)) {
            handle_emergency_stop();
        }
        
        // High priority: Real-time control updates
        if(atomic_test_and_clear_flag(&interrupt_flags.encoder_changed)) {
            // Encoder position affects PID setpoint
            pid_controller.setpoint = (float)abs(system_state.encoder_position);
        }
        
        // Normal priority: Sensor data processing
        if(atomic_test_and_clear_flag(&interrupt_flags.ultrasonic_ready)) {
            process_sensor_data();
        }
        
        if(atomic_test_and_clear_flag(&interrupt_flags.adc_complete)) {
            // Temperature data is already processed in ISR
            // Additional processing could go here
        }
        
        // Low priority: System monitoring and display
        if(atomic_test_and_clear_flag(&interrupt_flags.timer_tick)) {
            update_system_leds();
            print_system_status();
            
            // Check for interrupt overflow conditions
            if(interrupt_stats.max_nesting_level >= MAX_INTERRUPT_NESTING) {
                system_state.error_code = 0x10;  // Interrupt overflow warning
                interrupt_stats.max_nesting_level = 0;  // Reset counter
            }
        }
        
        // System error handling
        if(atomic_test_and_clear_flag(&interrupt_flags.system_error)) {
            switch(system_state.error_code) {
                case 0x03:  // Critical temperature
                    // Emergency cooling protocol
                    OCR0B = 255;  // Maximum cooling fan speed
                    PORTB |= (1 << ERROR_LED);
                    break;
                    
                default:
                    break;
            }
        }
        
        // Power management: Sleep when idle
        if(interrupt_nesting_level == 0 && 
           system_state.system_ticks % 10 == 0) {
            // Enter idle mode briefly to save power
            // Will wake up on any interrupt
            set_sleep_mode(SLEEP_MODE_IDLE);
            sleep_mode();
        }
    }
    
    return 0;
}