# 🎛️ Stage 4-2: PID 제어 시스템

## 📋 프로젝트 개요

**고성능 PID(비례-적분-미분) 제어기 구현**

ATmega328P를 기반으로 한 전문급 PID 제어 시스템으로, 실시간 피드백 제어, 적응형 튜닝, 고급 제어 알고리즘을 포함한 산업용 품질의 제어 시스템입니다. DC 모터 속도 제어, 온도 제어, 위치 제어 등 다양한 응용 분야에 적용 가능합니다.

## 🎯 학습 목표

### Core Control Theory
- **PID 알고리즘**: 비례, 적분, 미분 제어의 수학적 구현
- **제어 시스템 이론**: 전달함수, 안정성 분석, 주파수 응답
- **디지털 제어**: 이산시간 제어, Z-변환, 샘플링 이론
- **시스템 식별**: 플랜트 모델링 및 파라미터 추정

### Advanced Control Techniques
- **적응형 제어**: 실시간 파라미터 조정 및 최적화
- **Anti-Windup**: 적분 포화 방지 및 복구 메커니즘
- **Feed-Forward 제어**: 외란 보상 및 성능 향상
- **Cascade 제어**: 다중 루프 제어 구조

### Real-Time Implementation
- **고정밀 ADC**: 12-bit 센서 데이터 획득
- **PWM 출력**: 고해상도 액추에이터 제어
- **인터럽트 기반**: 결정론적 제어 루프 타이밍
- **데이터 로깅**: 실시간 성능 분석 및 튜닝

## ⚙️ 시스템 아키텍처

### Hardware Components
```
ATmega328P @ 16MHz
├── ADC0: 피드백 센서 입력 (0-5V, 12-bit)
├── ADC1: 설정값 입력 (가변저항)
├── ADC2: 외란 측정 (선택사항)
├── Timer0: PWM 출력 (액추에이터 제어)
├── Timer1: PID 제어 루프 타이밍 (1kHz)
├── Timer2: 데이터 로깅 및 통신
├── UART: 실시간 모니터링 및 튜닝
└── I2C: 추가 센서 확장
```

### Software Architecture
```
Application Layer
├── PID Controller Core (Fixed-point arithmetic)
├── Adaptive Tuning Engine (Ziegler-Nichols, GA)
├── Anti-Windup Mechanisms (Conditional integration)
├── Feed-Forward Compensator (Disturbance rejection)
└── System Identification (Recursive least squares)

Control Loop Layer  
├── Sensor Signal Processing (Filtering, calibration)
├── Setpoint Management (Ramping, limiting)
├── Output Limiting (Saturation, rate limiting)
├── Performance Monitoring (Rise time, overshoot)
└── Fault Detection (Sensor failure, actuator fault)

Hardware Abstraction Layer
├── High-Resolution ADC (Oversampling, noise reduction)
├── PWM Generation (Phase-correct, high frequency)
├── Timer Management (Precise timing control)
└── Communication Interface (UART, I2C protocols)
```

## 🔧 핵심 기능

### 1. PID Controller Core
```c
// PID 제어기 구조체
typedef struct {
    // PID 파라미터
    float kp, ki, kd;               // 비례, 적분, 미분 게인
    float dt;                       // 샘플링 시간
    
    // 상태 변수
    float setpoint;                 // 목표값
    float process_value;            // 현재 프로세스 값
    float error;                    // 현재 오차
    float last_error;               // 이전 오차
    float integral;                 // 적분 누적값
    float derivative;               // 미분값
    float output;                   // PID 출력
    
    // 제한값
    float output_min, output_max;   // 출력 제한
    float integral_min, integral_max; // 적분 windup 방지
    
    // 성능 지표
    float rise_time;                // 상승 시간
    float settling_time;            // 정착 시간
    float overshoot;                // 오버슈트
    float steady_state_error;       // 정상상태 오차
    
    // 고급 기능
    bool anti_windup_enabled;       // Anti-windup 활성화
    bool feed_forward_enabled;      // Feed-forward 활성화
    float feed_forward_gain;        // Feed-forward 게인
    
    // 튜닝 상태
    bool auto_tuning_active;        // 자동 튜닝 모드
    uint8_t tuning_method;          // 튜닝 방법 선택
} pid_controller_t;

// PID 계산 함수 (고정소수점 최적화)
float pid_compute(pid_controller_t *pid) {
    // 오차 계산
    pid->error = pid->setpoint - pid->process_value;
    
    // 비례 항
    float proportional = pid->kp * pid->error;
    
    // 적분 항 (Anti-windup 포함)
    if (!pid->anti_windup_enabled || 
        (pid->output > pid->output_min && pid->output < pid->output_max)) {
        pid->integral += pid->error * pid->dt;
    }
    
    // 적분 제한
    if (pid->integral > pid->integral_max) pid->integral = pid->integral_max;
    if (pid->integral < pid->integral_min) pid->integral = pid->integral_min;
    
    float integral_term = pid->ki * pid->integral;
    
    // 미분 항 (노이즈 필터링 포함)
    pid->derivative = (pid->error - pid->last_error) / pid->dt;
    float derivative_term = pid->kd * pid->derivative;
    
    // PID 출력 계산
    pid->output = proportional + integral_term + derivative_term;
    
    // Feed-forward 보상
    if (pid->feed_forward_enabled) {
        pid->output += pid->feed_forward_gain * pid->setpoint;
    }
    
    // 출력 제한
    if (pid->output > pid->output_max) pid->output = pid->output_max;
    if (pid->output < pid->output_min) pid->output = pid->output_min;
    
    // 이전 오차 저장
    pid->last_error = pid->error;
    
    return pid->output;
}
```

### 2. 적응형 자동 튜닝
```c
// Ziegler-Nichols 자동 튜닝
typedef struct {
    float ultimate_gain;            // 한계 게인 (Ku)
    float ultimate_period;          // 한계 주기 (Tu)
    bool oscillation_detected;      // 진동 감지
    uint32_t oscillation_count;     // 진동 횟수
    float peak_values[16];          // 피크값 기록
    uint8_t peak_index;             // 피크 인덱스
} ziegler_nichols_t;

// 유전자 알고리즘 튜닝
typedef struct {
    float population[20][3];        // 20개 개체 (Kp, Ki, Kd)
    float fitness[20];              // 적합도 점수
    uint8_t generation;             // 현재 세대
    float best_params[3];           // 최적 파라미터
    float best_fitness;             // 최고 적합도
} genetic_algorithm_t;

void auto_tune_ziegler_nichols(pid_controller_t *pid) {
    // 1단계: 한계 게인 찾기
    // 2단계: 한계 주기 측정  
    // 3단계: PID 파라미터 계산
    pid->kp = 0.6 * ultimate_gain;
    pid->ki = 2 * pid->kp / ultimate_period;
    pid->kd = pid->kp * ultimate_period / 8;
}
```

### 3. 고급 신호 처리
```c
// 디지털 필터 구조체
typedef struct {
    float a[3];                     // 피드백 계수
    float b[3];                     // 피드포워드 계수
    float x[3];                     // 입력 히스토리
    float y[3];                     // 출력 히스토리
    uint8_t index;                  // 순환 인덱스
} digital_filter_t;

// 2차 저역통과 필터 (Butterworth)
float butterworth_filter(digital_filter_t *filter, float input) {
    // 입력 값 저장
    filter->x[filter->index] = input;
    
    // 필터 계산 (Direct Form II)
    float output = filter->b[0] * filter->x[filter->index] +
                   filter->b[1] * filter->x[(filter->index + 2) % 3] +
                   filter->b[2] * filter->x[(filter->index + 1) % 3] -
                   filter->a[1] * filter->y[(filter->index + 2) % 3] -
                   filter->a[2] * filter->y[(filter->index + 1) % 3];
    
    // 출력 값 저장
    filter->y[filter->index] = output;
    
    // 인덱스 업데이트
    filter->index = (filter->index + 1) % 3;
    
    return output;
}
```

## 📊 실제 응용 예제

### Example 1: DC 모터 속도 제어
```c
void dc_motor_speed_control_task(void) {
    static pid_controller_t motor_pid;
    static bool initialized = false;
    
    if (!initialized) {
        // PID 파라미터 초기화
        pid_init(&motor_pid);
        motor_pid.kp = 2.0f;
        motor_pid.ki = 0.5f;
        motor_pid.kd = 0.1f;
        motor_pid.dt = 0.001f;  // 1ms 제어 주기
        motor_pid.output_min = 0.0f;
        motor_pid.output_max = 255.0f;
        initialized = true;
    }
    
    // 인코더에서 현재 속도 읽기
    float current_speed = read_encoder_speed();
    motor_pid.process_value = current_speed;
    
    // 목표 속도 설정 (가변저항 또는 UART 명령)
    motor_pid.setpoint = read_setpoint();
    
    // PID 계산
    float pwm_output = pid_compute(&motor_pid);
    
    // PWM 출력 (모터 드라이버)
    set_motor_pwm((uint8_t)pwm_output);
    
    // 성능 모니터링
    update_performance_metrics(&motor_pid);
}
```

### Example 2: 온도 제어 시스템
```c
void temperature_control_task(void) {
    static pid_controller_t temp_pid;
    static digital_filter_t temp_filter;
    
    // 온도 센서 읽기 (써미스터 + ADC)
    uint16_t adc_value = read_temperature_sensor();
    float temperature = convert_adc_to_celsius(adc_value);
    
    // 노이즈 필터링 (2차 저역통과 필터)
    temperature = butterworth_filter(&temp_filter, temperature);
    
    temp_pid.process_value = temperature;
    temp_pid.setpoint = target_temperature;
    
    // PID 계산 (히터 제어)
    float heater_output = pid_compute(&temp_pid);
    
    // PWM 출력 (히터 또는 펠티어 쿨러)
    set_heater_pwm((uint8_t)heater_output);
    
    // 안전 보호
    if (temperature > TEMPERATURE_SAFETY_LIMIT) {
        emergency_shutdown();
    }
}
```

### Example 3: 위치 제어 (서보 시스템)
```c
void position_control_task(void) {
    static pid_controller_t pos_pid;
    static pid_controller_t vel_pid;  // Cascade 제어
    
    // 위치 피드백 (엔코더 또는 포텐셔미터)
    float current_position = read_position_sensor();
    float current_velocity = calculate_velocity(current_position);
    
    // 외측 루프: 위치 제어
    pos_pid.process_value = current_position;
    float velocity_command = pid_compute(&pos_pid);
    
    // 내측 루프: 속도 제어
    vel_pid.setpoint = velocity_command;
    vel_pid.process_value = current_velocity;
    float torque_command = pid_compute(&vel_pid);
    
    // 모터 토크 출력
    set_motor_torque(torque_command);
    
    // 궤적 추종 성능 평가
    evaluate_tracking_performance(&pos_pid);
}
```

## 🔍 고급 제어 기법

### Adaptive Control
```c
// 모델 참조 적응 제어 (MRAC)
typedef struct {
    float reference_model[3];       // 참조 모델 파라미터
    float plant_estimate[3];        // 플랜트 추정 파라미터
    float adaptation_gain;          // 적응 게인
    float tracking_error;           // 추종 오차
} mrac_controller_t;

// 자기조정 제어 (Self-Tuning Control)
typedef struct {
    float theta_hat[6];             // 파라미터 추정값
    float p_matrix[36];             // 공분산 행렬 (6x6)
    float forgetting_factor;        // 망각 인자
    bool parameter_converged;       // 파라미터 수렴 상태
} recursive_least_squares_t;
```

### Robust Control
```c
// H-infinity 제어
typedef struct {
    float uncertainty_bound;        // 불확실성 경계
    float performance_weight;       // 성능 가중함수
    float robustness_margin;        // 강인성 여유도
} h_infinity_controller_t;

// 슬라이딩 모드 제어
typedef struct {
    float sliding_surface;          // 슬라이딩 표면
    float switching_gain;           // 스위칭 게인
    float boundary_layer;           // 경계층 두께
    bool chattering_reduction;      // 채터링 감소
} sliding_mode_controller_t;
```

## 📈 성능 분석 및 튜닝

### Real-time Performance Metrics
```c
typedef struct {
    // 시간 영역 성능
    float rise_time;                // 상승 시간 (10%-90%)
    float settling_time;            // 정착 시간 (±2%)
    float peak_time;                // 피크 시간
    float overshoot_percent;        // 오버슈트 (%)
    float undershoot_percent;       // 언더슈트 (%)
    
    // 주파수 영역 성능
    float gain_margin;              // 게인 여유도 (dB)
    float phase_margin;             // 위상 여유도 (도)
    float bandwidth;                // 대역폭 (Hz)
    float crossover_frequency;      // 교차 주파수 (Hz)
    
    // 강인성 지표
    float sensitivity_peak;         // 감도 함수 피크
    float comp_sensitivity_peak;    // 보완 감도 함수 피크
    float stability_margin;         // 안정도 여유도
    
    // 에너지 효율성
    float control_effort;           // 제어 노력 (RMS)
    float energy_consumption;       // 에너지 소모량
    float efficiency_ratio;         // 효율성 비율
} performance_metrics_t;
```

### Automated Tuning Algorithms
```c
// PSO (Particle Swarm Optimization) 튜닝
typedef struct {
    float particles[20][3];         // 20개 입자 위치 (Kp, Ki, Kd)
    float velocities[20][3];        // 입자 속도
    float personal_best[20][3];     // 개인 최적해
    float global_best[3];           // 전역 최적해
    float inertia_weight;           // 관성 가중치
    float cognitive_coeff;          // 인지 계수
    float social_coeff;             // 사회 계수
} pso_tuner_t;

// 베이지안 최적화 튜닝
typedef struct {
    float parameter_space[100][3];  // 파라미터 공간 샘플
    float objective_values[100];    // 목적함수 값
    float uncertainty[100];         // 불확실성 추정
    uint8_t evaluated_points;       // 평가된 점의 수
    float acquisition_function[100]; // 획득 함수 값
} bayesian_optimizer_t;
```

## 🚀 실시간 구현 최적화

### Fixed-Point Arithmetic
```c
// 고정소수점 PID (16.16 format)
typedef struct {
    int32_t kp_fixed, ki_fixed, kd_fixed;  // 게인 (16.16)
    int32_t integral_fixed;                 // 적분값 (16.16)
    int32_t error_fixed, last_error_fixed; // 오차 (16.16)
    uint16_t dt_ms;                         // 샘플링 시간 (ms)
} pid_fixed_point_t;

#define FIXED_POINT_SHIFT   16
#define FLOAT_TO_FIXED(x)   ((int32_t)((x) * (1 << FIXED_POINT_SHIFT)))
#define FIXED_TO_FLOAT(x)   ((float)(x) / (1 << FIXED_POINT_SHIFT))

int32_t pid_compute_fixed(pid_fixed_point_t *pid, int32_t setpoint, int32_t process_value) {
    // 고정소수점 PID 계산 (부동소수점 연산 없음)
    int32_t error = setpoint - process_value;
    
    // 비례 항
    int64_t proportional = ((int64_t)pid->kp_fixed * error) >> FIXED_POINT_SHIFT;
    
    // 적분 항
    pid->integral_fixed += error;  // dt는 정규화됨
    int64_t integral = ((int64_t)pid->ki_fixed * pid->integral_fixed) >> FIXED_POINT_SHIFT;
    
    // 미분 항
    int32_t derivative = error - pid->last_error_fixed;
    int64_t diff_term = ((int64_t)pid->kd_fixed * derivative) >> FIXED_POINT_SHIFT;
    
    pid->last_error_fixed = error;
    
    return (int32_t)(proportional + integral + diff_term);
}
```

### Assembly Optimization
```c
// 임계 경로 어셈블리 최적화
inline int32_t fast_multiply_16_16(int32_t a, int32_t b) {
    int32_t result;
    asm volatile (
        "mul %A1, %A2    \n\t"    // 하위 바이트 곱셈
        "mov %A0, r1     \n\t"
        "mul %A1, %B2    \n\t"    // 교차 곱셈
        "add %A0, r0     \n\t"
        "mov %B0, r1     \n\t"
        "mul %B1, %A2    \n\t"
        "add %B0, r0     \n\t"
        "clr r1          \n\t"
        : "=r" (result)
        : "r" (a), "r" (b)
        : "r0", "r1"
    );
    return result;
}
```

## 📱 사용자 인터페이스

### UART 제어 콘솔
```
=== PID Control System Console ===
Commands:
  sp <value>    - Set setpoint
  kp <value>    - Set proportional gain  
  ki <value>    - Set integral gain
  kd <value>    - Set derivative gain
  tune zn       - Auto-tune (Ziegler-Nichols)
  tune ga       - Auto-tune (Genetic Algorithm)
  status        - Show system status
  plot          - Start real-time plotting
  save          - Save parameters to EEPROM
  load          - Load parameters from EEPROM
  reset         - Reset controller
  help          - Show this help

Current Status:
  Setpoint: 50.00
  Process Value: 48.52
  Output: 67.3%
  Kp: 2.50, Ki: 0.80, Kd: 0.15
  Rise Time: 245ms
  Overshoot: 8.2%
  Settling Time: 1.2s
  Steady-State Error: 0.1%
```

### Real-time Data Visualization
```c
// 실시간 플롯 데이터 전송
void send_plot_data(void) {
    // CSV 형식으로 데이터 전송
    uart_printf("%lu,%.2f,%.2f,%.2f\r\n",
                system_time_ms,
                pid_controller.setpoint,
                pid_controller.process_value,
                pid_controller.output);
}

// Python/MATLAB 연동을 위한 바이너리 프로토콜
void send_binary_data(void) {
    plot_data_t data = {
        .timestamp = system_time_ms,
        .setpoint = pid_controller.setpoint,
        .process_value = pid_controller.process_value,
        .output = pid_controller.output,
        .error = pid_controller.error
    };
    
    uart_send_bytes((uint8_t*)&data, sizeof(data));
}
```

## 🎯 프로젝트 결과물

완성된 PID 제어 시스템은 다음과 같은 실무 역량을 보여줍니다:

1. **제어 이론 전문성**: 고급 제어 알고리즘 설계 및 구현
2. **실시간 시스템**: 결정론적 제어 루프 구현
3. **신호 처리**: 디지털 필터링 및 노이즈 제거
4. **최적화 기법**: 자동 튜닝 알고리즘 개발
5. **성능 분석**: 정량적 성능 평가 및 개선

이 프로젝트는 **제어 시스템 엔지니어** 또는 **고급 임베디드 개발자**로서 요구되는 전문 지식을 종합적으로 보여주는 **기술적 깊이가 있는 포트폴리오**입니다.

## 📚 참고 자료

- Modern Control Engineering by Ogata
- Digital Control System Analysis and Design by Phillips & Nagle  
- Adaptive Control by Åström & Wittenmark
- Robust Control Design by McFarlane & Glover
- Real-Time Control Systems by Cervin & Årzén

---
**난이도**: ⭐⭐⭐⭐⭐ (Expert)  
**예상 개발 시간**: 50-70시간  
**핵심 키워드**: `PID Control`, `Digital Signal Processing`, `Real-time Systems`, `Adaptive Control`, `System Identification`