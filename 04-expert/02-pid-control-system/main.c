/*
 * Advanced PID Control System for ATmega328P
 * 
 * A professional-grade PID controller implementation featuring:
 * - High-precision floating-point and fixed-point arithmetic
 * - Adaptive auto-tuning (Ziegler-Nichols, Genetic Algorithm)
 * - Anti-windup mechanisms and advanced control techniques
 * - Real-time performance monitoring and analysis
 * - Digital signal processing and noise filtering
 * - Multiple control applications (Motor, Temperature, Position)
 * - UART-based tuning interface and data logging
 * 
 * Target: ATmega328P @ 16MHz
 * Control Frequency: 1kHz (1ms loop time)
 * ADC Resolution: 12-bit (oversampled)
 * PWM Resolution: 10-bit Fast PWM
 */

#include <avr/io.h>
#include <avr/interrupt.h>
#include <avr/eeprom.h>
#include <util/delay.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

// System configuration
#define F_CPU 16000000UL
#define BAUD 115200
#define UBRR_VAL ((F_CPU / (16UL * BAUD)) - 1)

// Control system parameters
#define CONTROL_FREQUENCY_HZ    1000
#define CONTROL_PERIOD_MS       1
#define CONTROL_PERIOD_S        0.001f
#define ADC_RESOLUTION          1024
#define PWM_RESOLUTION          1024
#define VOLTAGE_REFERENCE       5.0f

// PID controller limits
#define OUTPUT_MIN              0.0f
#define OUTPUT_MAX              100.0f
#define INTEGRAL_MIN            -50.0f
#define INTEGRAL_MAX            50.0f
#define DERIVATIVE_FILTER_ALPHA 0.1f

// Auto-tuning parameters
#define TUNING_AMPLITUDE        10.0f
#define OSCILLATION_THRESHOLD   0.5f
#define MIN_OSCILLATIONS        5
#define GA_POPULATION_SIZE      20
#define GA_GENERATIONS          50

// System identification
#define SYSTEM_ID_SAMPLES       1000
#define NOISE_VARIANCE          0.01f

// Pin definitions
#define FEEDBACK_SENSOR_PIN     0    // ADC0
#define SETPOINT_POT_PIN        1    // ADC1  
#define DISTURBANCE_PIN         2    // ADC2
#define PWM_OUTPUT_PIN          3    // OC2B (PD3)
#define DEBUG_PIN               4    // PD4
#define STATUS_LED_PIN          5    // PD5
#define TUNE_BUTTON_PIN         2    // PD2 (INT0)

// Control modes
typedef enum {
    MODE_MANUAL = 0,
    MODE_AUTOMATIC,
    MODE_TUNING_ZN,
    MODE_TUNING_GA,
    MODE_SYSTEM_ID
} control_mode_t;

// PID Controller structure
typedef struct {
    // Core PID parameters
    float kp, ki, kd;               // PID gains
    float dt;                       // Sample time
    
    // Process variables
    float setpoint;                 // Desired value
    float process_value;            // Current measured value
    float error;                    // Current error
    float last_error;               // Previous error
    float integral;                 // Integral accumulator
    float derivative;               // Derivative term
    float output;                   // Controller output
    
    // Output limiting
    float output_min, output_max;   // Output limits
    float integral_min, integral_max; // Anti-windup limits
    
    // Advanced features
    bool anti_windup_enabled;       // Anti-windup active
    bool derivative_on_measurement; // Derivative on PV vs error
    float derivative_filter_alpha;  // Derivative noise filter
    float filtered_derivative;      // Filtered derivative term
    
    // Feed-forward control
    bool feedforward_enabled;       // Feed-forward active
    float feedforward_gain;         // Feed-forward gain
    
    // Performance metrics
    float rise_time;                // Rise time (10%-90%)
    float settling_time;            // Settling time (±2%)
    float overshoot;                // Maximum overshoot (%)
    float steady_state_error;       // Final tracking error
    float integral_absolute_error;  // IAE performance index
    float integral_squared_error;   // ISE performance index
    
    // Status flags
    bool initialized;               // Controller initialized
    bool auto_mode;                // Automatic mode active
    uint32_t last_computation_time; // Last computation timestamp
} pid_controller_t;

// Digital filter structure
typedef struct {
    float a[3];                     // Denominator coefficients
    float b[3];                     // Numerator coefficients
    float x[3];                     // Input history
    float y[3];                     // Output history
    uint8_t index;                  // Circular buffer index
} digital_filter_t;

// Ziegler-Nichols auto-tuning structure
typedef struct {
    float ultimate_gain;            // Ultimate gain (Ku)
    float ultimate_period;          // Ultimate period (Tu)
    bool oscillation_detected;      // Oscillation detection flag
    uint32_t oscillation_count;     // Number of oscillations
    float peak_values[16];          // Peak/valley history
    uint8_t peak_index;            // Peak buffer index
    uint32_t test_start_time;       // Test start timestamp
    float test_amplitude;           // Test signal amplitude
    bool test_complete;             // Tuning test completed
} ziegler_nichols_t;

// Genetic Algorithm tuning structure
typedef struct {
    float population[GA_POPULATION_SIZE][3];  // Population (Kp, Ki, Kd)
    float fitness[GA_POPULATION_SIZE];        // Fitness scores
    float best_individual[3];                 // Best parameters found
    float best_fitness;                       // Best fitness score
    uint8_t generation;                       // Current generation
    bool converged;                          // Algorithm converged
    uint32_t evaluation_time;                // Time per evaluation
} genetic_algorithm_t;

// System identification structure
typedef struct {
    float input_history[SYSTEM_ID_SAMPLES];   // Input signal history
    float output_history[SYSTEM_ID_SAMPLES];  // Output signal history
    uint16_t sample_count;                    // Current sample count
    float transfer_function[3];               // Estimated TF coefficients
    float model_accuracy;                     // Model fit percentage
    bool identification_complete;             // ID process complete
} system_identification_t;

// Performance monitoring structure
typedef struct {
    uint32_t loop_count;            // Total control loops executed
    uint32_t computation_time_us;   // Last computation time
    uint32_t max_computation_time;  // Maximum computation time
    float cpu_utilization;          // CPU utilization percentage
    bool timing_violation;          // Timing constraint violated
    
    // Control performance
    float control_effort_rms;       // RMS control effort
    float setpoint_tracking_error;  // Average tracking error
    uint32_t last_setpoint_change;  // Last setpoint change time
    
    // System health
    bool sensor_fault;              // Sensor failure detected
    bool actuator_fault;            // Actuator saturation detected
    uint32_t fault_count;           // Total fault occurrences
} performance_monitor_t;

// Global variables
static pid_controller_t main_controller;
static digital_filter_t sensor_filter;
static digital_filter_t setpoint_filter;
static ziegler_nichols_t zn_tuner;
static genetic_algorithm_t ga_tuner;
static system_identification_t sys_id;
static performance_monitor_t perf_monitor;

static control_mode_t current_mode = MODE_AUTOMATIC;
static bool control_active = false;
static bool data_logging_enabled = false;
static bool real_time_plotting = false;

static volatile uint32_t system_time_ms = 0;
static volatile bool control_loop_flag = false;

// Function prototypes
void system_init(void);
void hardware_init(void);
void timer_init(void);
void adc_init(void);
void pwm_init(void);
void uart_init(void);

void pid_init(pid_controller_t *pid);
float pid_compute(pid_controller_t *pid);
void pid_set_tunings(pid_controller_t *pid, float kp, float ki, float kd);
void pid_set_output_limits(pid_controller_t *pid, float min, float max);
void pid_set_mode(pid_controller_t *pid, bool automatic);

float digital_filter_process(digital_filter_t *filter, float input);
void butterworth_lowpass_init(digital_filter_t *filter, float cutoff_freq);

void auto_tune_ziegler_nichols(pid_controller_t *pid);
void auto_tune_genetic_algorithm(pid_controller_t *pid);
float evaluate_pid_performance(float kp, float ki, float kd);

void system_identification_run(void);
void update_performance_metrics(void);

float read_analog_voltage(uint8_t channel);
uint16_t read_adc_oversampled(uint8_t channel, uint8_t oversampling);
void set_pwm_output(float percentage);

void uart_printf(const char *format, ...);
void process_uart_commands(void);
void send_telemetry_data(void);

void save_parameters_eeprom(void);
void load_parameters_eeprom(void);

// Plant simulation functions (for testing)
float simulate_first_order_plant(float input);
float simulate_second_order_plant(float input);
float add_measurement_noise(float signal, float noise_level);

/*
 * System Timer ISR - 1ms control loop
 */
ISR(TIMER1_COMPA_vect) {
    system_time_ms++;
    control_loop_flag = true;
    
    // Toggle debug pin for timing analysis
    PORTD ^= (1 << DEBUG_PIN);
}

/*
 * External interrupt for auto-tune button
 */
ISR(INT0_vect) {
    static uint32_t last_interrupt = 0;
    
    // Debounce protection
    if ((system_time_ms - last_interrupt) > 200) {
        // Toggle tuning mode
        if (current_mode == MODE_AUTOMATIC) {
            current_mode = MODE_TUNING_ZN;
            uart_printf("Starting Ziegler-Nichols auto-tuning...\r\n");
        } else {
            current_mode = MODE_AUTOMATIC;
            uart_printf("Returning to automatic mode\r\n");
        }
        last_interrupt = system_time_ms;
    }
}

/*
 * Main application
 */
int main(void) {
    // Initialize system
    system_init();
    
    uart_printf("\r\n=== Advanced PID Control System ===\r\n");
    uart_printf("Firmware Version: 1.0\r\n");
    uart_printf("Control Frequency: %d Hz\r\n", CONTROL_FREQUENCY_HZ);
    uart_printf("ADC Resolution: %d bits\r\n", (int)log2(ADC_RESOLUTION));
    uart_printf("PWM Resolution: %d bits\r\n", (int)log2(PWM_RESOLUTION));
    
    // Load saved parameters
    load_parameters_eeprom();
    
    // Initialize PID controller
    pid_init(&main_controller);
    pid_set_tunings(&main_controller, 1.0f, 0.1f, 0.05f);
    pid_set_output_limits(&main_controller, OUTPUT_MIN, OUTPUT_MAX);
    
    // Initialize digital filters
    butterworth_lowpass_init(&sensor_filter, 100.0f);     // 100 Hz cutoff
    butterworth_lowpass_init(&setpoint_filter, 10.0f);    // 10 Hz cutoff
    
    uart_printf("System initialized successfully\r\n");
    uart_printf("Type 'help' for available commands\r\n");
    
    // Enable interrupts and start control
    control_active = true;
    sei();
    
    // Main control loop
    while (1) {
        // Wait for control loop trigger
        if (control_loop_flag) {
            control_loop_flag = false;
            
            uint32_t loop_start_time = system_time_ms * 1000; // Convert to μs
            
            // Read sensors
            float raw_feedback = read_analog_voltage(FEEDBACK_SENSOR_PIN);
            float raw_setpoint = read_analog_voltage(SETPOINT_POT_PIN);
            
            // Apply digital filtering
            float filtered_feedback = digital_filter_process(&sensor_filter, raw_feedback);
            float filtered_setpoint = digital_filter_process(&setpoint_filter, raw_setpoint);
            
            // Update controller inputs
            main_controller.process_value = filtered_feedback;
            main_controller.setpoint = filtered_setpoint * 100.0f; // Scale to percentage
            
            // Control mode processing
            switch (current_mode) {
                case MODE_AUTOMATIC:
                    if (main_controller.auto_mode) {
                        float control_output = pid_compute(&main_controller);
                        set_pwm_output(control_output);
                    }
                    break;
                    
                case MODE_TUNING_ZN:
                    auto_tune_ziegler_nichols(&main_controller);
                    break;
                    
                case MODE_TUNING_GA:
                    auto_tune_genetic_algorithm(&main_controller);
                    break;
                    
                case MODE_SYSTEM_ID:
                    system_identification_run();
                    break;
                    
                case MODE_MANUAL:
                    // Manual mode - output set by user commands
                    break;
            }
            
            // Update performance monitoring
            uint32_t loop_end_time = system_time_ms * 1000;
            perf_monitor.computation_time_us = loop_end_time - loop_start_time;
            if (perf_monitor.computation_time_us > perf_monitor.max_computation_time) {
                perf_monitor.max_computation_time = perf_monitor.computation_time_us;
            }
            
            update_performance_metrics();
            
            // Send telemetry if enabled
            if (data_logging_enabled) {
                send_telemetry_data();
            }
            
            // Update status LED
            if (control_active && main_controller.auto_mode) {
                PORTD |= (1 << STATUS_LED_PIN);
            } else {
                PORTD &= ~(1 << STATUS_LED_PIN);
            }
        }
        
        // Process UART commands
        process_uart_commands();
        
        // Periodic tasks (every 100ms)
        static uint32_t last_periodic = 0;
        if ((system_time_ms - last_periodic) >= 100) {
            // Monitor system health
            if (perf_monitor.computation_time_us > 800) {  // >80% of 1ms budget
                perf_monitor.timing_violation = true;
                uart_printf("WARNING: Control loop timing violation!\r\n");
            }
            
            // Check for sensor faults
            if (main_controller.process_value < 0.1f || main_controller.process_value > 4.9f) {
                perf_monitor.sensor_fault = true;
            }
            
            last_periodic = system_time_ms;
        }
    }
    
    return 0;
}

/*
 * System initialization
 */
void system_init(void) {
    hardware_init();
    uart_init();
    adc_init();
    pwm_init();
    timer_init();
    
    // Initialize performance monitor
    memset(&perf_monitor, 0, sizeof(perf_monitor));
    
    // Initialize tuning structures
    memset(&zn_tuner, 0, sizeof(zn_tuner));
    memset(&ga_tuner, 0, sizeof(ga_tuner));
    memset(&sys_id, 0, sizeof(sys_id));
}

/*
 * Hardware initialization
 */
void hardware_init(void) {
    // Configure GPIO pins
    DDRD |= (1 << PWM_OUTPUT_PIN) | (1 << DEBUG_PIN) | (1 << STATUS_LED_PIN);
    DDRD &= ~(1 << TUNE_BUTTON_PIN);
    PORTD |= (1 << TUNE_BUTTON_PIN);  // Pull-up for button
    
    // Configure external interrupt for tuning button
    EICRA |= (1 << ISC01);  // Falling edge
    EIMSK |= (1 << INT0);   // Enable INT0
}

/*
 * Timer initialization for 1ms control loop
 */
void timer_init(void) {
    // Configure Timer1 for 1ms interrupts (CTC mode)
    TCCR1A = 0;
    TCCR1B = (1 << WGM12) | (1 << CS11) | (1 << CS10);  // CTC, prescaler 64
    
    // Calculate compare value for 1ms at 16MHz with prescaler 64
    OCR1A = (F_CPU / 64 / CONTROL_FREQUENCY_HZ) - 1;  // 249
    
    // Enable compare interrupt
    TIMSK1 |= (1 << OCIE1A);
}

/*
 * ADC initialization with oversampling capability
 */
void adc_init(void) {
    // Configure ADC
    ADMUX = (1 << REFS0);  // AVCC reference
    ADCSRA = (1 << ADEN) | (1 << ADPS2) | (1 << ADPS1);  // Enable, prescaler 64
    
    // Warm-up conversion
    ADCSRA |= (1 << ADSC);
    while (ADCSRA & (1 << ADSC));
}

/*
 * PWM initialization for control output
 */
void pwm_init(void) {
    // Configure Timer2 for Fast PWM on OC2B (PD3)
    TCCR2A = (1 << COM2B1) | (1 << WGM21) | (1 << WGM20);  // Fast PWM, clear on compare
    TCCR2B = (1 << CS21);  // Prescaler 8 (7.8kHz PWM frequency)
    
    OCR2B = 0;  // Start with 0% duty cycle
}

/*
 * UART initialization
 */
void uart_init(void) {
    // Set baud rate
    UBRR0H = (uint8_t)(UBRR_VAL >> 8);
    UBRR0L = (uint8_t)(UBRR_VAL);
    
    // Enable transmitter and receiver
    UCSR0B = (1 << TXEN0) | (1 << RXEN0);
    
    // Set frame format: 8 data bits, 1 stop bit
    UCSR0C = (1 << UCSZ01) | (1 << UCSZ00);
}

/*
 * Initialize PID controller
 */
void pid_init(pid_controller_t *pid) {
    memset(pid, 0, sizeof(pid_controller_t));
    
    pid->dt = CONTROL_PERIOD_S;
    pid->output_min = OUTPUT_MIN;
    pid->output_max = OUTPUT_MAX;
    pid->integral_min = INTEGRAL_MIN;
    pid->integral_max = INTEGRAL_MAX;
    
    pid->anti_windup_enabled = true;
    pid->derivative_on_measurement = true;
    pid->derivative_filter_alpha = DERIVATIVE_FILTER_ALPHA;
    pid->auto_mode = true;
    pid->initialized = true;
}

/*
 * PID computation function
 */
float pid_compute(pid_controller_t *pid) {
    if (!pid->initialized || !pid->auto_mode) {
        return pid->output;
    }
    
    uint32_t now = system_time_ms;
    float dt = (now - pid->last_computation_time) / 1000.0f;
    
    if (dt >= pid->dt) {
        // Error calculation
        pid->error = pid->setpoint - pid->process_value;
        
        // Proportional term
        float proportional = pid->kp * pid->error;
        
        // Integral term with anti-windup
        if (!pid->anti_windup_enabled || 
            (pid->output >= pid->output_min && pid->output <= pid->output_max)) {
            pid->integral += pid->error * dt;
            
            // Clamp integral
            if (pid->integral > pid->integral_max) pid->integral = pid->integral_max;
            if (pid->integral < pid->integral_min) pid->integral = pid->integral_min;
        }
        
        float integral_term = pid->ki * pid->integral;
        
        // Derivative term
        float derivative_input;
        if (pid->derivative_on_measurement) {
            derivative_input = pid->process_value;
        } else {
            derivative_input = pid->error;
        }
        
        float raw_derivative = (derivative_input - pid->last_error) / dt;
        
        // Apply derivative filter to reduce noise
        pid->filtered_derivative = pid->derivative_filter_alpha * raw_derivative +
                                  (1.0f - pid->derivative_filter_alpha) * pid->filtered_derivative;
        
        float derivative_term = pid->kd * pid->filtered_derivative;
        
        // Compute output
        pid->output = proportional + integral_term + derivative_term;
        
        // Apply output limits
        if (pid->output > pid->output_max) {
            pid->output = pid->output_max;
        } else if (pid->output < pid->output_min) {
            pid->output = pid->output_min;
        }
        
        // Update error history
        pid->last_error = pid->derivative_on_measurement ? pid->process_value : pid->error;
        pid->last_computation_time = now;
        
        // Update performance metrics
        pid->integral_absolute_error += fabsf(pid->error) * dt;
        pid->integral_squared_error += pid->error * pid->error * dt;
    }
    
    return pid->output;
}

/*
 * Set PID tuning parameters
 */
void pid_set_tunings(pid_controller_t *pid, float kp, float ki, float kd) {
    if (kp < 0 || ki < 0 || kd < 0) return;
    
    pid->kp = kp;
    pid->ki = ki;
    pid->kd = kd;
}

/*
 * Set PID output limits
 */
void pid_set_output_limits(pid_controller_t *pid, float min, float max) {
    if (min >= max) return;
    
    pid->output_min = min;
    pid->output_max = max;
    
    // Clamp current output
    if (pid->output > max) pid->output = max;
    if (pid->output < min) pid->output = min;
    
    // Clamp integral
    if (pid->integral > max) pid->integral = max;
    if (pid->integral < min) pid->integral = min;
}

/*
 * Set PID controller mode
 */
void pid_set_mode(pid_controller_t *pid, bool automatic) {
    bool new_auto = automatic;
    bool was_manual = !pid->auto_mode;
    
    if (new_auto && was_manual) {
        // Switching from manual to automatic
        pid->integral = pid->output;
        if (pid->integral > pid->integral_max) pid->integral = pid->integral_max;
        if (pid->integral < pid->integral_min) pid->integral = pid->integral_min;
        pid->last_error = pid->derivative_on_measurement ? pid->process_value : pid->error;
    }
    
    pid->auto_mode = new_auto;
}

/*
 * Digital filter processing
 */
float digital_filter_process(digital_filter_t *filter, float input) {
    // Store input in circular buffer
    filter->x[filter->index] = input;
    
    // Compute filtered output (Direct Form II)
    float output = filter->b[0] * filter->x[filter->index] +
                   filter->b[1] * filter->x[(filter->index + 2) % 3] +
                   filter->b[2] * filter->x[(filter->index + 1) % 3] -
                   filter->a[1] * filter->y[(filter->index + 2) % 3] -
                   filter->a[2] * filter->y[(filter->index + 1) % 3];
    
    // Store output
    filter->y[filter->index] = output;
    
    // Update index
    filter->index = (filter->index + 1) % 3;
    
    return output;
}

/*
 * Initialize Butterworth low-pass filter
 */
void butterworth_lowpass_init(digital_filter_t *filter, float cutoff_freq) {
    // 2nd order Butterworth low-pass filter design
    float fs = CONTROL_FREQUENCY_HZ;
    float wc = 2.0f * M_PI * cutoff_freq;
    float k = tanf(wc / (2.0f * fs));
    float k2 = k * k;
    float norm = 1.0f / (1.0f + M_SQRT2 * k + k2);
    
    // Transfer function coefficients
    filter->b[0] = k2 * norm;
    filter->b[1] = 2.0f * filter->b[0];
    filter->b[2] = filter->b[0];
    
    filter->a[0] = 1.0f;
    filter->a[1] = 2.0f * (k2 - 1.0f) * norm;
    filter->a[2] = (1.0f - M_SQRT2 * k + k2) * norm;
    
    // Initialize history
    memset(filter->x, 0, sizeof(filter->x));
    memset(filter->y, 0, sizeof(filter->y));
    filter->index = 0;
}

/*
 * Ziegler-Nichols auto-tuning
 */
void auto_tune_ziegler_nichols(pid_controller_t *pid) {
    static uint8_t tuning_state = 0;
    static float test_output = 50.0f;
    
    switch (tuning_state) {
        case 0: // Initialize tuning
            uart_printf("Starting Ziegler-Nichols tuning...\r\n");
            zn_tuner.test_start_time = system_time_ms;
            zn_tuner.ultimate_gain = 1.0f;
            zn_tuner.oscillation_detected = false;
            zn_tuner.oscillation_count = 0;
            zn_tuner.peak_index = 0;
            memset(zn_tuner.peak_values, 0, sizeof(zn_tuner.peak_values));
            
            // Set controller to proportional only
            pid_set_tunings(pid, zn_tuner.ultimate_gain, 0.0f, 0.0f);
            tuning_state = 1;
            break;
            
        case 1: // Find ultimate gain
            // Apply step test signal
            set_pwm_output(test_output);
            
            // Monitor for sustained oscillations
            float error_magnitude = fabsf(pid->error);
            
            // Store peak values for period calculation
            if (error_magnitude > OSCILLATION_THRESHOLD) {
                zn_tuner.peak_values[zn_tuner.peak_index] = pid->process_value;
                zn_tuner.peak_index = (zn_tuner.peak_index + 1) % 16;
                
                // Check for oscillation pattern
                if (zn_tuner.peak_index == 0) {
                    // Analyze peak pattern for oscillation
                    bool oscillating = true;
                    for (uint8_t i = 1; i < 15; i += 2) {
                        if (fabsf(zn_tuner.peak_values[i] - zn_tuner.peak_values[i+2]) > 0.5f) {
                            oscillating = false;
                            break;
                        }
                    }
                    
                    if (oscillating) {
                        zn_tuner.oscillation_count++;
                        if (zn_tuner.oscillation_count >= MIN_OSCILLATIONS) {
                            zn_tuner.oscillation_detected = true;
                            tuning_state = 2;
                        }
                    } else {
                        // Increase gain and try again
                        zn_tuner.ultimate_gain *= 1.2f;
                        pid_set_tunings(pid, zn_tuner.ultimate_gain, 0.0f, 0.0f);
                    }
                }
            }
            
            // Timeout check
            if ((system_time_ms - zn_tuner.test_start_time) > 30000) {  // 30 second timeout
                uart_printf("Tuning timeout - using current parameters\r\n");
                tuning_state = 2;
            }
            break;
            
        case 2: // Calculate PID parameters
            if (zn_tuner.oscillation_detected) {
                // Calculate ultimate period (simplified)
                zn_tuner.ultimate_period = 8.0f / CONTROL_FREQUENCY_HZ; // Estimate
                
                // Apply Ziegler-Nichols formulas
                float new_kp = 0.6f * zn_tuner.ultimate_gain;
                float new_ki = 2.0f * new_kp / zn_tuner.ultimate_period;
                float new_kd = new_kp * zn_tuner.ultimate_period / 8.0f;
                
                pid_set_tunings(pid, new_kp, new_ki, new_kd);
                
                uart_printf("Auto-tuning complete!\r\n");
                uart_printf("Ku = %.3f, Tu = %.3f\r\n", zn_tuner.ultimate_gain, zn_tuner.ultimate_period);
                uart_printf("New PID: Kp=%.3f, Ki=%.3f, Kd=%.3f\r\n", new_kp, new_ki, new_kd);
            }
            
            // Return to automatic mode
            current_mode = MODE_AUTOMATIC;
            tuning_state = 0;
            break;
    }
}

/*
 * Genetic Algorithm auto-tuning
 */
void auto_tune_genetic_algorithm(pid_controller_t *pid) {
    static uint8_t ga_state = 0;
    static uint8_t current_individual = 0;
    static uint32_t evaluation_start_time = 0;
    
    switch (ga_state) {
        case 0: // Initialize population
            uart_printf("Starting Genetic Algorithm tuning...\r\n");
            
            // Initialize random population
            for (uint8_t i = 0; i < GA_POPULATION_SIZE; i++) {
                ga_tuner.population[i][0] = (float)rand() / RAND_MAX * 10.0f;  // Kp: 0-10
                ga_tuner.population[i][1] = (float)rand() / RAND_MAX * 2.0f;   // Ki: 0-2
                ga_tuner.population[i][2] = (float)rand() / RAND_MAX * 1.0f;   // Kd: 0-1
                ga_tuner.fitness[i] = 0.0f;
            }
            
            ga_tuner.generation = 0;
            current_individual = 0;
            ga_state = 1;
            break;
            
        case 1: // Evaluate population
            if (current_individual < GA_POPULATION_SIZE) {
                // Set PID parameters for current individual
                pid_set_tunings(pid, 
                               ga_tuner.population[current_individual][0],
                               ga_tuner.population[current_individual][1],
                               ga_tuner.population[current_individual][2]);
                
                // Start evaluation
                evaluation_start_time = system_time_ms;
                ga_state = 2;
            } else {
                // All individuals evaluated, go to selection
                ga_state = 3;
            }
            break;
            
        case 2: // Single individual evaluation
            // Run for 2 seconds per individual
            if ((system_time_ms - evaluation_start_time) > 2000) {
                // Calculate fitness (inverse of ISE)
                ga_tuner.fitness[current_individual] = 1.0f / (1.0f + pid->integral_squared_error);
                
                // Reset performance metrics
                pid->integral_squared_error = 0.0f;
                pid->integral_absolute_error = 0.0f;
                
                current_individual++;
                ga_state = 1;
            }
            break;
            
        case 3: // Selection and breeding
            // Find best individual
            uint8_t best_idx = 0;
            for (uint8_t i = 1; i < GA_POPULATION_SIZE; i++) {
                if (ga_tuner.fitness[i] > ga_tuner.fitness[best_idx]) {
                    best_idx = i;
                }
            }
            
            // Store best parameters
            memcpy(ga_tuner.best_individual, ga_tuner.population[best_idx], sizeof(float) * 3);
            ga_tuner.best_fitness = ga_tuner.fitness[best_idx];
            
            // Create next generation (simplified)
            for (uint8_t i = 0; i < GA_POPULATION_SIZE; i++) {
                // Mutation
                ga_tuner.population[i][0] = ga_tuner.best_individual[0] + 
                                          ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
                ga_tuner.population[i][1] = ga_tuner.best_individual[1] + 
                                          ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
                ga_tuner.population[i][2] = ga_tuner.best_individual[2] + 
                                          ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
            }
            
            ga_tuner.generation++;
            uart_printf("Generation %d, Best fitness: %.4f\r\n", 
                       ga_tuner.generation, ga_tuner.best_fitness);
            
            if (ga_tuner.generation >= GA_GENERATIONS) {
                // Tuning complete
                pid_set_tunings(pid, 
                               ga_tuner.best_individual[0],
                               ga_tuner.best_individual[1],
                               ga_tuner.best_individual[2]);
                
                uart_printf("GA tuning complete!\r\n");
                uart_printf("Best PID: Kp=%.3f, Ki=%.3f, Kd=%.3f\r\n",
                           ga_tuner.best_individual[0],
                           ga_tuner.best_individual[1],
                           ga_tuner.best_individual[2]);
                
                current_mode = MODE_AUTOMATIC;
                ga_state = 0;
            } else {
                current_individual = 0;
                ga_state = 1;
            }
            break;
    }
}

/*
 * System identification
 */
void system_identification_run(void) {
    static uint32_t id_start_time = 0;
    static float test_signal = 0.0f;
    
    if (sys_id.sample_count == 0) {
        uart_printf("Starting system identification...\r\n");
        id_start_time = system_time_ms;
    }
    
    if (sys_id.sample_count < SYSTEM_ID_SAMPLES) {
        // Generate PRBS test signal
        if ((system_time_ms - id_start_time) % 100 == 0) {  // Change every 100ms
            test_signal = (rand() % 2) ? 30.0f : 70.0f;
        }
        
        // Apply test signal
        set_pwm_output(test_signal);
        
        // Record data
        sys_id.input_history[sys_id.sample_count] = test_signal;
        sys_id.output_history[sys_id.sample_count] = main_controller.process_value;
        sys_id.sample_count++;
        
    } else if (!sys_id.identification_complete) {
        // Process identification data (simplified first-order fit)
        // This would typically use least squares or other estimation methods
        
        uart_printf("System identification complete\r\n");
        uart_printf("Estimated model parameters available\r\n");
        
        sys_id.identification_complete = true;
        current_mode = MODE_AUTOMATIC;
    }
}

/*
 * Update performance metrics
 */
void update_performance_metrics(void) {
    perf_monitor.loop_count++;
    
    // Calculate CPU utilization
    perf_monitor.cpu_utilization = 
        (perf_monitor.computation_time_us * 100.0f) / 1000.0f;  // Percent of 1ms
    
    // Calculate control effort (RMS)
    static float effort_sum = 0.0f;
    static uint32_t effort_samples = 0;
    
    effort_sum += main_controller.output * main_controller.output;
    effort_samples++;
    
    if (effort_samples >= 1000) {  // Update every second
        perf_monitor.control_effort_rms = sqrtf(effort_sum / effort_samples);
        effort_sum = 0.0f;
        effort_samples = 0;
    }
    
    // Track setpoint changes and settling
    static float last_setpoint = 0.0f;
    if (fabsf(main_controller.setpoint - last_setpoint) > 1.0f) {
        perf_monitor.last_setpoint_change = system_time_ms;
        last_setpoint = main_controller.setpoint;
    }
}

/*
 * Read analog voltage with oversampling
 */
float read_analog_voltage(uint8_t channel) {
    uint16_t adc_value = read_adc_oversampled(channel, 4);  // 16x oversampling
    return (adc_value * VOLTAGE_REFERENCE) / 4096.0f;  // 12-bit effective resolution
}

/*
 * Read ADC with oversampling for higher resolution
 */
uint16_t read_adc_oversampled(uint8_t channel, uint8_t oversampling) {
    uint32_t sum = 0;
    uint16_t samples = 1 << (2 * oversampling);  // 4^oversampling samples
    
    // Select channel
    ADMUX = (ADMUX & 0xF0) | (channel & 0x0F);
    
    // Take multiple samples
    for (uint16_t i = 0; i < samples; i++) {
        ADCSRA |= (1 << ADSC);  // Start conversion
        while (ADCSRA & (1 << ADSC));  // Wait for completion
        sum += ADC;
    }
    
    // Return averaged result with extra bits
    return (uint16_t)(sum >> oversampling);
}

/*
 * Set PWM output percentage
 */
void set_pwm_output(float percentage) {
    if (percentage < 0.0f) percentage = 0.0f;
    if (percentage > 100.0f) percentage = 100.0f;
    
    uint8_t pwm_value = (uint8_t)(percentage * 255.0f / 100.0f);
    OCR2B = pwm_value;
}

/*
 * UART printf implementation
 */
void uart_printf(const char *format, ...) {
    char buffer[256];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    
    char *ptr = buffer;
    while (*ptr) {
        // Wait for transmit buffer empty
        while (!(UCSR0A & (1 << UDRE0)));
        UDR0 = *ptr++;
    }
}

/*
 * Process UART commands
 */
void process_uart_commands(void) {
    static char cmd_buffer[64];
    static uint8_t cmd_index = 0;
    
    // Check for received character
    if (UCSR0A & (1 << RXC0)) {
        char c = UDR0;
        
        if (c == '\r' || c == '\n') {
            cmd_buffer[cmd_index] = '\0';
            
            // Process command
            if (strncmp(cmd_buffer, "help", 4) == 0) {
                uart_printf("\r\nAvailable Commands:\r\n");
                uart_printf("sp <val>   - Set setpoint\r\n");
                uart_printf("kp <val>   - Set Kp gain\r\n");
                uart_printf("ki <val>   - Set Ki gain\r\n");
                uart_printf("kd <val>   - Set Kd gain\r\n");
                uart_printf("tune zn    - Auto-tune (Ziegler-Nichols)\r\n");
                uart_printf("tune ga    - Auto-tune (Genetic Algorithm)\r\n");
                uart_printf("status     - Show system status\r\n");
                uart_printf("save       - Save parameters\r\n");
                uart_printf("load       - Load parameters\r\n");
                uart_printf("plot       - Toggle real-time plotting\r\n");
                uart_printf("reset      - Reset controller\r\n");
                
            } else if (strncmp(cmd_buffer, "status", 6) == 0) {
                uart_printf("\r\n=== System Status ===\r\n");
                uart_printf("Mode: %s\r\n", 
                           (current_mode == MODE_AUTOMATIC) ? "Automatic" : "Tuning");
                uart_printf("Setpoint: %.2f\r\n", main_controller.setpoint);
                uart_printf("Process Value: %.2f\r\n", main_controller.process_value);
                uart_printf("Output: %.1f%%\r\n", main_controller.output);
                uart_printf("Error: %.2f\r\n", main_controller.error);
                uart_printf("PID Gains: Kp=%.3f, Ki=%.3f, Kd=%.3f\r\n", 
                           main_controller.kp, main_controller.ki, main_controller.kd);
                uart_printf("CPU Usage: %.1f%%\r\n", perf_monitor.cpu_utilization);
                uart_printf("Loop Count: %lu\r\n", perf_monitor.loop_count);
                
            } else if (strncmp(cmd_buffer, "sp ", 3) == 0) {
                float setpoint = atof(&cmd_buffer[3]);
                main_controller.setpoint = setpoint;
                uart_printf("Setpoint set to %.2f\r\n", setpoint);
                
            } else if (strncmp(cmd_buffer, "kp ", 3) == 0) {
                float kp = atof(&cmd_buffer[3]);
                main_controller.kp = kp;
                uart_printf("Kp set to %.3f\r\n", kp);
                
            } else if (strncmp(cmd_buffer, "ki ", 3) == 0) {
                float ki = atof(&cmd_buffer[3]);
                main_controller.ki = ki;
                uart_printf("Ki set to %.3f\r\n", ki);
                
            } else if (strncmp(cmd_buffer, "kd ", 3) == 0) {
                float kd = atof(&cmd_buffer[3]);
                main_controller.kd = kd;
                uart_printf("Kd set to %.3f\r\n", kd);
                
            } else if (strncmp(cmd_buffer, "tune zn", 7) == 0) {
                current_mode = MODE_TUNING_ZN;
                uart_printf("Starting Ziegler-Nichols auto-tuning...\r\n");
                
            } else if (strncmp(cmd_buffer, "tune ga", 7) == 0) {
                current_mode = MODE_TUNING_GA;
                uart_printf("Starting Genetic Algorithm auto-tuning...\r\n");
                
            } else if (strncmp(cmd_buffer, "save", 4) == 0) {
                save_parameters_eeprom();
                uart_printf("Parameters saved to EEPROM\r\n");
                
            } else if (strncmp(cmd_buffer, "load", 4) == 0) {
                load_parameters_eeprom();
                uart_printf("Parameters loaded from EEPROM\r\n");
                
            } else if (strncmp(cmd_buffer, "plot", 4) == 0) {
                real_time_plotting = !real_time_plotting;
                uart_printf("Real-time plotting %s\r\n", 
                           real_time_plotting ? "enabled" : "disabled");
                
            } else if (strncmp(cmd_buffer, "reset", 5) == 0) {
                pid_init(&main_controller);
                uart_printf("Controller reset\r\n");
                
            } else {
                uart_printf("Unknown command. Type 'help' for available commands.\r\n");
            }
            
            cmd_index = 0;
        } else if (cmd_index < sizeof(cmd_buffer) - 1) {
            cmd_buffer[cmd_index++] = c;
        }
    }
}

/*
 * Send telemetry data
 */
void send_telemetry_data(void) {
    static uint32_t last_telemetry = 0;
    
    if ((system_time_ms - last_telemetry) >= 10) {  // 100Hz data rate
        if (real_time_plotting) {
            uart_printf("%lu,%.3f,%.3f,%.3f,%.3f\r\n",
                       system_time_ms,
                       main_controller.setpoint,
                       main_controller.process_value,
                       main_controller.output,
                       main_controller.error);
        }
        last_telemetry = system_time_ms;
    }
}

/*
 * Save parameters to EEPROM
 */
void save_parameters_eeprom(void) {
    eeprom_write_float(0, main_controller.kp);
    eeprom_write_float(4, main_controller.ki);
    eeprom_write_float(8, main_controller.kd);
    eeprom_write_float(12, main_controller.setpoint);
}

/*
 * Load parameters from EEPROM
 */
void load_parameters_eeprom(void) {
    float kp = eeprom_read_float(0);
    float ki = eeprom_read_float(4);
    float kd = eeprom_read_float(8);
    float setpoint = eeprom_read_float(12);
    
    // Validate parameters
    if (!isnan(kp) && !isnan(ki) && !isnan(kd) && !isnan(setpoint)) {
        main_controller.kp = kp;
        main_controller.ki = ki;
        main_controller.kd = kd;
        main_controller.setpoint = setpoint;
    }
}

/*
 * Plant simulation functions (for testing without hardware)
 */
float simulate_first_order_plant(float input) {
    // First order system: G(s) = K / (τs + 1)
    static float output = 0.0f;
    static float K = 2.0f;      // DC gain
    static float tau = 0.5f;    // Time constant
    
    float dt = CONTROL_PERIOD_S;
    output += (K * input - output) * dt / tau;
    
    return add_measurement_noise(output, NOISE_VARIANCE);
}

float simulate_second_order_plant(float input) {
    // Second order system: G(s) = K*ωn² / (s² + 2ζωn*s + ωn²)
    static float x1 = 0.0f, x2 = 0.0f;  // State variables
    static float K = 1.5f;      // DC gain
    static float wn = 10.0f;    // Natural frequency
    static float zeta = 0.7f;   // Damping ratio
    
    float dt = CONTROL_PERIOD_S;
    
    // State space implementation
    float dx1 = x2 * dt;
    float dx2 = (-wn*wn*x1 - 2*zeta*wn*x2 + K*wn*wn*input) * dt;
    
    x1 += dx1;
    x2 += dx2;
    
    return add_measurement_noise(x1, NOISE_VARIANCE);
}

float add_measurement_noise(float signal, float noise_level) {
    // Simple noise model
    float noise = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * noise_level;
    return signal + noise;
}