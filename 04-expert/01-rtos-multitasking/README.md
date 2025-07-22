# 🚀 Stage 4-1: RTOS 멀티태스킹 시스템

## 📋 프로젝트 개요

**실시간 운영체제(RTOS) 기반의 고급 멀티태스킹 시스템**

ATmega328P에서 협력적/선점적 스케줄링을 구현하여 복수의 독립적인 태스크를 동시 실행하는 전문적인 임베디드 시스템을 개발합니다. 우선순위 기반 스케줄링, 태스크 간 통신, 동기화 메커니즘을 포함한 완전한 RTOS 구현체입니다.

## 🎯 학습 목표

### Core RTOS Concepts
- **태스크 스케줄링**: 협력적 vs 선점적 스케줄링 알고리즘
- **우선순위 관리**: 고정/동적 우선순위 시스템
- **컨텍스트 스위칭**: 레지스터 상태 저장/복원 메커니즘
- **태스크 상태 관리**: Ready, Running, Blocked, Suspended 상태

### Inter-Task Communication
- **메시지 큐**: 태스크 간 비동기 데이터 교환
- **세마포어**: 리소스 접근 제어 및 동기화
- **뮤텍스**: 상호 배제 및 우선순위 역전 방지
- **이벤트 플래그**: 조건 기반 태스크 동기화

### Memory Management
- **스택 관리**: 태스크별 독립적 스택 공간
- **힙 관리**: 동적 메모리 할당/해제
- **메모리 풀**: 고정 크기 블록 관리
- **메모리 보호**: 스택 오버플로우 검출

## ⚙️ 시스템 아키텍처

### Hardware Components
```
ATmega328P @ 16MHz
├── Timer1: 시스템 틱 (1ms) - 스케줄러 인터럽트
├── Timer2: 태스크 시간 측정 및 타임아웃
├── UART: 실시간 시스템 모니터링
├── GPIO: 태스크 활동 표시 LED
└── External Interrupt: 우선순위 태스크 트리거
```

### Software Architecture
```
Application Layer
├── Task 1: LED Blinker (Priority 1 - Low)
├── Task 2: Button Handler (Priority 2 - Medium)
├── Task 3: UART Monitor (Priority 3 - High)
└── Task 4: System Watchdog (Priority 4 - Critical)

RTOS Kernel Layer
├── Task Scheduler (Round-Robin + Priority)
├── Context Switcher (Assembly optimized)
├── Message Queue System
├── Semaphore/Mutex Manager
├── Timer Management
└── Memory Pool Manager

Hardware Abstraction Layer
├── Timer Drivers
├── UART Driver
├── GPIO Driver
└── Interrupt Handlers
```

## 🔧 핵심 기능

### 1. Task Scheduler
```c
// 태스크 제어 블록 (TCB) 구조
typedef struct task_control_block {
    uint8_t task_id;                    // 태스크 ID
    task_state_t state;                 // 현재 상태
    uint8_t priority;                   // 우선순위 (0=최고)
    uint16_t stack_pointer;             // 스택 포인터
    uint8_t *stack_base;                // 스택 베이스 주소
    uint16_t stack_size;                // 스택 크기
    uint32_t cpu_time;                  // CPU 사용 시간
    uint32_t wake_time;                 // 깨어날 시간
    void (*task_function)(void*);       // 태스크 함수
    void *task_parameter;               // 태스크 파라미터
    struct task_control_block *next;    // 다음 태스크 포인터
} tcb_t;

// 스케줄링 알고리즘
void task_scheduler(void);
void context_switch(tcb_t *from, tcb_t *to);
void task_yield(void);
void task_delay(uint32_t ticks);
```

### 2. Inter-Task Communication
```c
// 메시지 큐 시스템
typedef struct message_queue {
    uint8_t *buffer;                    // 메시지 버퍼
    uint16_t size;                      // 큐 크기
    uint16_t head, tail;                // 헤드/테일 포인터
    uint16_t count;                     // 현재 메시지 수
    semaphore_t semaphore;              // 동기화 세마포어
} message_queue_t;

// 세마포어 시스템
typedef struct semaphore {
    int16_t count;                      // 카운터 값
    tcb_t *waiting_list;                // 대기 중인 태스크 리스트
} semaphore_t;

// API 함수들
rtos_result_t queue_send(message_queue_t *queue, void *message, uint32_t timeout);
rtos_result_t queue_receive(message_queue_t *queue, void *message, uint32_t timeout);
rtos_result_t semaphore_take(semaphore_t *semaphore, uint32_t timeout);
rtos_result_t semaphore_give(semaphore_t *semaphore);
```

### 3. Memory Management
```c
// 메모리 풀 관리
typedef struct memory_pool {
    uint8_t *pool_start;                // 풀 시작 주소
    uint16_t block_size;                // 블록 크기
    uint16_t num_blocks;                // 총 블록 수
    uint16_t free_blocks;               // 사용 가능한 블록 수
    uint8_t *free_list;                 // 프리 리스트
} memory_pool_t;

// 스택 오버플로우 검출
#define STACK_CANARY    0xDEADBEEF
void stack_check_canary(tcb_t *task);
bool is_stack_overflow(tcb_t *task);
```

## 📊 성능 지표

### Timing Specifications
- **Context Switch Time**: <50μs (ATmega328P @ 16MHz)
- **Scheduler Overhead**: <2% CPU utilization
- **Interrupt Response**: <10μs for critical tasks
- **Task Creation Time**: <100μs
- **Memory Allocation**: <20μs per block

### Resource Usage
- **Flash Memory**: ~16-20KB (50-62% of 32KB)
- **SRAM Usage**: ~1.2-1.6KB (60-80% of 2KB)
- **Stack per Task**: 128-256 bytes (configurable)
- **Kernel Overhead**: ~400 bytes SRAM

### Scalability Metrics
- **Maximum Tasks**: 8-16 concurrent tasks
- **Message Queue Depth**: 4-16 messages per queue
- **Semaphore Count**: Up to 255 per semaphore
- **Timer Resolution**: 1ms system tick

## 🎮 실제 응용 예제

### Task 1: LED Blinker (Priority 1)
```c
void led_blinker_task(void *parameter) {
    led_config_t *config = (led_config_t*)parameter;
    
    while(1) {
        gpio_toggle(config->led_pin);
        task_delay(config->blink_period);
        
        // CPU 사용률 모니터링
        if(get_cpu_usage() > 80) {
            task_delay(100);  // 시스템 부하 감소
        }
    }
}
```

### Task 2: Button Handler (Priority 2)
```c
void button_handler_task(void *parameter) {
    button_event_t event;
    
    while(1) {
        if(queue_receive(&button_queue, &event, portMAX_DELAY) == RTOS_OK) {
            switch(event.type) {
                case BUTTON_PRESS:
                    semaphore_give(&user_input_semaphore);
                    break;
                case BUTTON_LONG_PRESS:
                    queue_send(&system_queue, &shutdown_msg, 0);
                    break;
            }
        }
    }
}
```

### Task 3: UART Monitor (Priority 3)
```c
void uart_monitor_task(void *parameter) {
    system_stats_t stats;
    
    while(1) {
        // 시스템 통계 수집
        stats.cpu_usage = get_cpu_usage();
        stats.memory_usage = get_memory_usage();
        stats.task_count = get_active_task_count();
        
        // UART로 출력
        uart_printf("CPU: %d%%, MEM: %dB, TASKS: %d\r\n", 
                   stats.cpu_usage, stats.memory_usage, stats.task_count);
        
        task_delay(1000);  // 1초마다 업데이트
    }
}
```

### Task 4: System Watchdog (Priority 4 - Highest)
```c
void watchdog_task(void *parameter) {
    uint32_t last_heartbeat[MAX_TASKS];
    
    while(1) {
        // 모든 태스크의 하트비트 검사
        for(uint8_t i = 0; i < task_count; i++) {
            if((get_system_tick() - last_heartbeat[i]) > WATCHDOG_TIMEOUT) {
                // 태스크 데드락 감지
                uart_printf("Task %d deadlock detected!\r\n", i);
                task_restart(i);
            }
        }
        
        task_delay(500);  // 0.5초마다 검사
    }
}
```

## 🔍 고급 디버깅 및 프로파일링

### Real-time Profiling
```c
// CPU 사용률 측정
typedef struct cpu_profiler {
    uint32_t total_time;
    uint32_t idle_time;
    uint32_t task_time[MAX_TASKS];
    uint8_t current_usage;
} cpu_profiler_t;

// 스택 사용량 분석
typedef struct stack_analyzer {
    uint16_t max_usage[MAX_TASKS];
    uint16_t current_usage[MAX_TASKS];
    uint8_t overflow_count[MAX_TASKS];
} stack_analyzer_t;
```

### Debug Console Commands
```
> task list                    # 모든 태스크 상태 출력
> task stats <id>             # 특정 태스크 통계
> memory usage                # 메모리 사용량 분석
> queue status                # 메시지 큐 상태
> semaphore list             # 세마포어 상태
> cpu profile                # CPU 프로파일링
> stack check                # 스택 오버플로우 검사
```

## 🚀 고급 최적화 기법

### Context Switch Optimization
- **Assembly Language**: 컨텍스트 스위칭을 어셈블리로 최적화
- **Register Banking**: 레지스터 저장/복원 최소화
- **Fast Interrupt**: 고속 인터럽트 처리 경로

### Memory Optimization
- **Zero-Copy Messaging**: 포인터 기반 메시지 전달
- **Stack Sharing**: 유휴 태스크 스택 공유
- **Compressed TCB**: 태스크 제어 블록 압축

### Power Management
- **Dynamic Frequency Scaling**: 부하에 따른 클럭 조정
- **Task Suspension**: 유휴 태스크 일시 정지
- **Sleep Mode Integration**: 모든 태스크 대기 시 절전 모드

## 📈 확장성 및 포팅

### Multi-Core Support (Future)
- SMP (Symmetric Multi-Processing) 준비
- 태스크 affinity 설정
- 코어 간 통신 메커니즘

### Platform Porting Guide
- HAL (Hardware Abstraction Layer) 분리
- 아키텍처별 어셈블리 코드 모듈화
- 컴파일 시간 구성 옵션

## 🎯 프로젝트 결과물

완성된 RTOS는 다음과 같은 실무 역량을 보여줍니다:

1. **시스템 설계**: 복잡한 임베디드 시스템 아키텍처 설계
2. **동시성 제어**: 멀티태스킹 환경에서의 리소스 관리
3. **성능 최적화**: 제한된 자원에서의 효율적인 알고리즘 구현
4. **디버깅 스킬**: 실시간 시스템의 복잡한 버그 추적
5. **문서화**: 전문적인 기술 문서 작성

이 프로젝트는 **시니어 임베디드 개발자**로서 요구되는 핵심 역량들을 종합적으로 보여주는 **포트폴리오의 하이라이트**입니다.

## 📚 참고 자료

- FreeRTOS Design Patterns
- Real-Time Systems by Liu
- AVR Assembly Language Programming
- Embedded Systems Architecture by Tammy Noergaard
- The Art of Multiprocessor Programming

---
**난이도**: ⭐⭐⭐⭐⭐ (Expert)  
**예상 개발 시간**: 40-60시간  
**핵심 키워드**: `RTOS`, `Multitasking`, `Real-time`, `Scheduler`, `IPC`, `Memory Management`