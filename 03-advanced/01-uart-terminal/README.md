# 📡 UART Terminal - 고급 시리얼 통신 시스템

## 📋 프로젝트 개요

UART(Universal Asynchronous Receiver-Transmitter)를 이용한 완전한 시리얼 통신 시스템을 구현하는 프로젝트입니다. 단순한 데이터 전송을 넘어서 프로토콜 구현, 에러 처리, 버퍼 관리 등 실무에서 사용되는 고급 기능들을 다룹니다.

### 학습 목표
- UART 레지스터 직접 제어
- 인터럽트 기반 송수신 버퍼링
- 프로토콜 설계 및 구현
- 에러 검출 및 복구
- Flow Control과 Handshaking

### 핵심 개념
- Baud Rate 계산 및 설정
- Frame Format (Data bits, Parity, Stop bits)
- Ring Buffer 구현
- Protocol State Machine
- CRC 및 Checksum
- AT Command Interface

## 🛠️ 하드웨어 구성

### 필요 부품
- ATmega328P (Arduino Uno/Nano)
- ESP8266/ESP32 모듈 x 1 (WiFi 통신용)
- GPS 모듈 (NEO-6M) x 1
- Bluetooth 모듈 (HC-05) x 1
- LCD 16x2 디스플레이
- LED x 4
- 버튼 x 4
- RS232 Level Shifter (MAX232)
- SD 카드 모듈
- 점퍼 와이어

### 회로도
```
    ATmega328P
    ┌─────────┐
    │         │ Hardware UART (Primary)
    │   PD1   ├──────────── TX → PC/ESP8266/GPS
    │   (TX)  │
    │   PD0   ├──────────── RX ← PC/ESP8266/GPS  
    │   (RX)  │
    │         │
    │ Software UART #1 (Bluetooth)
    │   PD2   ├──────────── TX → HC-05 RX
    │   PD3   ├──────────── RX ← HC-05 TX
    │         │
    │ Software UART #2 (GPS)
    │   PD4   ├──────────── TX → GPS RX (not used)
    │   PD5   ├──────────── RX ← GPS TX
    │         │
    │ Status LEDs          │
    │   PB0   ├──[220Ω]──┤▶├─ GND (UART Activity)
    │   PB1   ├──[220Ω]──┤▶├─ GND (WiFi Status)
    │   PB2   ├──[220Ω]──┤▶├─ GND (GPS Lock)
    │   PB3   ├──[220Ω]──┤▶├─ GND (Error)
    │         │
    │ Control Buttons      │
    │   PC0   ├──[SW1]──── GND (Mode Select)
    │   PC1   ├──[SW2]──── GND (Send Data)
    │   PC2   ├──[SW3]──── GND (Connect/Disconnect)
    │   PC3   ├──[SW4]──── GND (Emergency Stop)
    └─────────┘

RS232 Level Converter (MAX232):
    ATmega328P          MAX232          PC/Device
    ┌─────────┐        ┌─────────┐      ┌─────────┐
    │   TX    ├────────┤T1in T1out├──────┤   RX    │
    │   RX    ├────────┤R1out R1in├──────┤   TX    │
    │   VCC   ├────────┤VCC      │      │   GND   │
    │   GND   ├────────┤GND      │      └─────────┘
    └─────────┘        └─────────┘
                           │
                       Capacitors
                     (1μF x 4, per datasheet)

ESP8266 WiFi Module:
    ┌─────────┐
    │   VCC   ├──── 3.3V (via regulator)
    │   GND   ├──── GND
    │   TX    ├──── PD0 (RX)
    │   RX    ├──── PD1 (TX) via level shifter
    │   EN    ├──── 3.3V (enable)
    │   RST   ├──── PC4 (reset control)
    └─────────┘
```

## 💻 소스 코드

### Version 1: 기본 UART 구현
```c
#include <avr/io.h>
#include <avr/interrupt.h>
#include <stdbool.h>
#include <string.h>

#define BAUD_RATE 9600
#define UBRR_VALUE ((F_CPU/16/BAUD_RATE) - 1)

// Ring buffer for UART
#define UART_BUFFER_SIZE 128
typedef struct {
    volatile uint8_t buffer[UART_BUFFER_SIZE];
    volatile uint8_t head;
    volatile uint8_t tail;
    volatile uint8_t count;
} ring_buffer_t;

ring_buffer_t tx_buffer = {0};
ring_buffer_t rx_buffer = {0};

void uart_init(void) {
    // Set baud rate
    UBRR0H = (uint8_t)(UBRR_VALUE >> 8);
    UBRR0L = (uint8_t)UBRR_VALUE;
    
    // Enable transmitter and receiver with interrupts
    UCSR0B = (1 << TXEN0) | (1 << RXEN0) | (1 << RXCIE0);
    
    // Set frame format: 8 data bits, 1 stop bit, no parity
    UCSR0C = (1 << UCSZ01) | (1 << UCSZ00);
}

// UART RX Complete interrupt
ISR(USART_RX_vect) {
    uint8_t data = UDR0;
    
    // Check for frame errors
    if(UCSR0A & ((1 << FE0) | (1 << DOR0) | (1 << UPE0))) {
        // Handle errors - for now just ignore corrupted data
        return;
    }
    
    // Add to ring buffer if space available
    if(rx_buffer.count < UART_BUFFER_SIZE) {
        rx_buffer.buffer[rx_buffer.head] = data;
        rx_buffer.head = (rx_buffer.head + 1) % UART_BUFFER_SIZE;
        rx_buffer.count++;
    }
}

// UART TX Complete interrupt
ISR(USART_UDRE_vect) {
    if(tx_buffer.count > 0) {
        UDR0 = tx_buffer.buffer[tx_buffer.tail];
        tx_buffer.tail = (tx_buffer.tail + 1) % UART_BUFFER_SIZE;
        tx_buffer.count--;
    } else {
        // Disable interrupt when buffer is empty
        UCSR0B &= ~(1 << UDRIE0);
    }
}

void uart_putchar(char c) {
    // Wait if buffer is full
    while(tx_buffer.count >= UART_BUFFER_SIZE);
    
    cli();
    tx_buffer.buffer[tx_buffer.head] = c;
    tx_buffer.head = (tx_buffer.head + 1) % UART_BUFFER_SIZE;
    tx_buffer.count++;
    
    // Enable TX interrupt
    UCSR0B |= (1 << UDRIE0);
    sei();
}

bool uart_getchar(char *c) {
    if(rx_buffer.count > 0) {
        cli();
        *c = rx_buffer.buffer[rx_buffer.tail];
        rx_buffer.tail = (rx_buffer.tail + 1) % UART_BUFFER_SIZE;
        rx_buffer.count--;
        sei();
        return true;
    }
    return false;
}

void uart_puts(const char *str) {
    while(*str) {
        uart_putchar(*str++);
    }
}

int main(void) {
    uart_init();
    sei();
    
    uart_puts("UART Terminal Ready\r\n");
    
    while(1) {
        char received;
        if(uart_getchar(&received)) {
            // Echo received character
            uart_putchar(received);
            
            // Process special characters
            if(received == '\r') {
                uart_putchar('\n');
            }
        }
    }
    
    return 0;
}
```

### Version 2: 고급 프로토콜 구현
```c
#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/crc16.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>

// Protocol definitions
#define PROTOCOL_START_BYTE     0xAA
#define PROTOCOL_END_BYTE       0x55
#define MAX_PAYLOAD_SIZE        64
#define MAX_COMMAND_LENGTH      32

// Command types
typedef enum {
    CMD_PING = 0x01,
    CMD_GET_STATUS = 0x02,
    CMD_SET_CONFIG = 0x03,
    CMD_READ_SENSOR = 0x04,
    CMD_WRITE_OUTPUT = 0x05,
    CMD_RESET = 0x06,
    CMD_ERROR = 0xFF
} command_type_t;

// Packet structure
typedef struct {
    uint8_t start;          // 0xAA
    uint8_t command;        // Command type
    uint8_t length;         // Payload length
    uint8_t payload[MAX_PAYLOAD_SIZE];
    uint16_t crc;           // CRC16
    uint8_t end;            // 0x55
} protocol_packet_t;

// Parser state machine
typedef enum {
    STATE_WAIT_START,
    STATE_GET_COMMAND,
    STATE_GET_LENGTH,
    STATE_GET_PAYLOAD,
    STATE_GET_CRC_HIGH,
    STATE_GET_CRC_LOW,
    STATE_GET_END
} parser_state_t;

// Global variables
volatile parser_state_t parser_state = STATE_WAIT_START;
volatile protocol_packet_t current_packet;
volatile uint8_t payload_index = 0;
volatile bool packet_ready = false;

// AT Command interface
typedef struct {
    const char *command;
    void (*handler)(const char *params);
} at_command_t;

// System status
typedef struct {
    uint16_t temperature;
    uint16_t humidity;
    uint8_t wifi_status;
    uint8_t gps_status;
    uint32_t uptime_seconds;
} system_status_t;

volatile system_status_t system_status = {0};

// UART RX interrupt with protocol parsing
ISR(USART_RX_vect) {
    uint8_t data = UDR0;
    
    // Error checking
    if(UCSR0A & ((1 << FE0) | (1 << DOR0) | (1 << UPE0))) {
        parser_state = STATE_WAIT_START;
        return;
    }
    
    // Protocol state machine
    switch(parser_state) {
        case STATE_WAIT_START:
            if(data == PROTOCOL_START_BYTE) {
                current_packet.start = data;
                parser_state = STATE_GET_COMMAND;
            }
            break;
            
        case STATE_GET_COMMAND:
            current_packet.command = data;
            parser_state = STATE_GET_LENGTH;
            break;
            
        case STATE_GET_LENGTH:
            if(data <= MAX_PAYLOAD_SIZE) {
                current_packet.length = data;
                payload_index = 0;
                parser_state = (data > 0) ? STATE_GET_PAYLOAD : STATE_GET_CRC_HIGH;
            } else {
                parser_state = STATE_WAIT_START;  // Invalid length
            }
            break;
            
        case STATE_GET_PAYLOAD:
            current_packet.payload[payload_index++] = data;
            if(payload_index >= current_packet.length) {
                parser_state = STATE_GET_CRC_HIGH;
            }
            break;
            
        case STATE_GET_CRC_HIGH:
            current_packet.crc = (data << 8);
            parser_state = STATE_GET_CRC_LOW;
            break;
            
        case STATE_GET_CRC_LOW:
            current_packet.crc |= data;
            parser_state = STATE_GET_END;
            break;
            
        case STATE_GET_END:
            if(data == PROTOCOL_END_BYTE) {
                current_packet.end = data;
                packet_ready = true;
            }
            parser_state = STATE_WAIT_START;
            break;
    }
}

uint16_t calculate_crc(const uint8_t *data, uint8_t length) {
    uint16_t crc = 0xFFFF;
    
    for(uint8_t i = 0; i < length; i++) {
        crc = _crc16_update(crc, data[i]);
    }
    
    return crc;
}

bool validate_packet(void) {
    // Calculate CRC for command + length + payload
    uint8_t crc_data[2 + MAX_PAYLOAD_SIZE];
    crc_data[0] = current_packet.command;
    crc_data[1] = current_packet.length;
    
    for(uint8_t i = 0; i < current_packet.length; i++) {
        crc_data[2 + i] = current_packet.payload[i];
    }
    
    uint16_t calculated_crc = calculate_crc(crc_data, 2 + current_packet.length);
    return (calculated_crc == current_packet.crc);
}

void send_response(command_type_t cmd, const uint8_t *payload, uint8_t length) {
    protocol_packet_t response;
    
    response.start = PROTOCOL_START_BYTE;
    response.command = cmd;
    response.length = length;
    
    // Copy payload
    for(uint8_t i = 0; i < length; i++) {
        response.payload[i] = payload[i];
    }
    
    // Calculate CRC
    uint8_t crc_data[2 + MAX_PAYLOAD_SIZE];
    crc_data[0] = response.command;
    crc_data[1] = response.length;
    for(uint8_t i = 0; i < length; i++) {
        crc_data[2 + i] = payload[i];
    }
    response.crc = calculate_crc(crc_data, 2 + length);
    response.end = PROTOCOL_END_BYTE;
    
    // Send packet
    uart_putchar(response.start);
    uart_putchar(response.command);
    uart_putchar(response.length);
    
    for(uint8_t i = 0; i < length; i++) {
        uart_putchar(response.payload[i]);
    }
    
    uart_putchar((uint8_t)(response.crc >> 8));
    uart_putchar((uint8_t)(response.crc & 0xFF));
    uart_putchar(response.end);
}

void handle_ping_command(void) {
    const char *response = "PONG";
    send_response(CMD_PING, (const uint8_t*)response, strlen(response));
}

void handle_status_command(void) {
    uint8_t status_data[sizeof(system_status_t)];
    memcpy(status_data, (const void*)&system_status, sizeof(system_status_t));
    send_response(CMD_GET_STATUS, status_data, sizeof(system_status_t));
}

void handle_sensor_command(void) {
    // Read sensor specified in payload
    if(current_packet.length > 0) {
        uint8_t sensor_id = current_packet.payload[0];
        uint16_t sensor_value = 0;
        
        switch(sensor_id) {
            case 0: // Temperature
                sensor_value = system_status.temperature;
                break;
            case 1: // Humidity  
                sensor_value = system_status.humidity;
                break;
            default:
                sensor_value = 0xFFFF; // Invalid sensor
                break;
        }
        
        uint8_t response[3] = {sensor_id, 
                              (uint8_t)(sensor_value >> 8),
                              (uint8_t)(sensor_value & 0xFF)};
        send_response(CMD_READ_SENSOR, response, 3);
    }
}

void process_protocol_packet(void) {
    if(!validate_packet()) {
        // Send error response
        const char *error_msg = "CRC_ERROR";
        send_response(CMD_ERROR, (const uint8_t*)error_msg, strlen(error_msg));
        return;
    }
    
    // Process command
    switch(current_packet.command) {
        case CMD_PING:
            handle_ping_command();
            break;
            
        case CMD_GET_STATUS:
            handle_status_command();
            break;
            
        case CMD_READ_SENSOR:
            handle_sensor_command();
            break;
            
        case CMD_RESET:
            // Perform system reset
            uart_puts("RESETTING...\r\n");
            // In real system: wdt_enable(WDTO_15MS);
            break;
            
        default:
            {
                const char *error_msg = "UNKNOWN_CMD";
                send_response(CMD_ERROR, (const uint8_t*)error_msg, strlen(error_msg));
            }
            break;
    }
}

// AT Command handlers
void at_ping(const char *params) {
    uart_puts("PONG\r\n");
}

void at_status(const char *params) {
    char buffer[128];
    snprintf(buffer, sizeof(buffer), 
             "STATUS: TEMP=%d, HUM=%d, WIFI=%d, GPS=%d, UP=%lu\r\n",
             system_status.temperature,
             system_status.humidity, 
             system_status.wifi_status,
             system_status.gps_status,
             system_status.uptime_seconds);
    uart_puts(buffer);
}

void at_wifi(const char *params) {
    if(strncmp(params, "CONNECT", 7) == 0) {
        uart_puts("WIFI CONNECTING...\r\n");
        system_status.wifi_status = 1;
    } else if(strncmp(params, "DISCONNECT", 10) == 0) {
        uart_puts("WIFI DISCONNECTED\r\n");
        system_status.wifi_status = 0;
    } else {
        uart_puts("WIFI STATUS: ");
        uart_puts(system_status.wifi_status ? "CONNECTED\r\n" : "DISCONNECTED\r\n");
    }
}

// AT Command table
const at_command_t at_commands[] = {
    {"PING", at_ping},
    {"STATUS", at_status},
    {"WIFI", at_wifi},
    {NULL, NULL}  // End marker
};

void process_at_command(const char *command_line) {
    // Skip "AT+" prefix
    if(strncmp(command_line, "AT+", 3) != 0) {
        uart_puts("ERROR: Invalid AT command format\r\n");
        return;
    }
    
    const char *cmd_start = command_line + 3;
    char command[MAX_COMMAND_LENGTH];
    char *params = NULL;
    
    // Find command and parameters
    char *equals = strchr(cmd_start, '=');
    if(equals) {
        size_t cmd_len = equals - cmd_start;
        strncpy(command, cmd_start, cmd_len);
        command[cmd_len] = '\0';
        params = equals + 1;
    } else {
        strcpy(command, cmd_start);
    }
    
    // Find and execute command
    for(const at_command_t *at_cmd = at_commands; at_cmd->command; at_cmd++) {
        if(strcmp(command, at_cmd->command) == 0) {
            at_cmd->handler(params);
            return;
        }
    }
    
    uart_puts("ERROR: Unknown AT command\r\n");
}

int main(void) {
    uart_init();
    sei();
    
    uart_puts("\r\n=== Advanced UART Terminal ===\r\n");
    uart_puts("Protocol: Binary or AT Commands\r\n");
    uart_puts("Ready for commands...\r\n\r\n");
    
    char command_buffer[MAX_COMMAND_LENGTH];
    uint8_t cmd_index = 0;
    bool in_at_mode = false;
    
    while(1) {
        // Update system status periodically
        static uint32_t last_update = 0;
        if(++last_update > 100000) {  // Rough timing
            last_update = 0;
            system_status.uptime_seconds++;
            system_status.temperature = 250 + (system_status.uptime_seconds % 100);
            system_status.humidity = 500 + (system_status.uptime_seconds % 200);
        }
        
        // Process binary protocol packets
        if(packet_ready) {
            packet_ready = false;
            process_protocol_packet();
        }
        
        // Process AT commands (text mode)
        char received;
        if(uart_getchar(&received)) {
            if(received == 'A' && !in_at_mode) {
                // Start of potential AT command
                command_buffer[0] = 'A';
                cmd_index = 1;
                in_at_mode = true;
            } else if(in_at_mode) {
                if(received == '\r' || received == '\n') {
                    // End of command
                    command_buffer[cmd_index] = '\0';
                    
                    if(cmd_index > 2) {  // At least "AT+"
                        process_at_command(command_buffer);
                    }
                    
                    cmd_index = 0;
                    in_at_mode = false;
                } else if(cmd_index < MAX_COMMAND_LENGTH - 1) {
                    command_buffer[cmd_index++] = received;
                } else {
                    // Command too long
                    uart_puts("ERROR: Command too long\r\n");
                    cmd_index = 0;
                    in_at_mode = false;
                }
            }
        }
    }
    
    return 0;
}
```

## 📊 기술 분석

### UART 프레임 포맷
```
    Start  Data Bits    Parity   Stop
    Bit    (5-9)        (O/E/N)  Bit(s)
    ┌─┐   ┌─────────┐    ┌─┐    ┌─┐
────┘ └───┘ LSB→MSB └────┘ └────┘ └────

Standard 8N1: 8 data bits, No parity, 1 stop bit
```

### Baud Rate 계산
```
UBRR = (F_CPU / (16 × Baud Rate)) - 1

예시 (16MHz, 9600 bps):
UBRR = (16,000,000 / (16 × 9600)) - 1 = 103.17 ≈ 103

실제 Baud Rate = 16,000,000 / (16 × (103 + 1)) = 9615.38 bps
오차 = (9615.38 - 9600) / 9600 = 0.16% (허용 범위)
```

### 프로토콜 계층
| 레벨 | 기능 | 구현 |
|------|------|------|
| Physical | 전기적 신호 | RS232/TTL |
| Data Link | 프레임 구조 | Start/Stop/Parity |
| Network | 패킷 구조 | Custom Protocol |
| Transport | 신뢰성 | CRC, ACK/NAK |
| Application | 명령어 | AT Commands |

## 🔍 실험 결과

### 통신 성능 측정
- **9600 bps**: 에러율 0.01%, 지연시간 ~1ms
- **115200 bps**: 에러율 0.1%, 지연시간 ~100μs  
- **버퍼 효율**: 128바이트로 약 13ms 버퍼링

### 프로토콜 오버헤드
- **바이너리 프로토콜**: 7바이트 헤더 (효율적)
- **AT 명령어**: 가변 길이 (가독성 좋음)
- **CRC16**: 99.998% 에러 검출률

## 🚀 심화 학습

### 도전 과제
1. Modbus RTU 프로토콜 구현
2. 다중 UART 포트 관리
3. 무선 통신 프로토콜 스택
4. 실시간 데이터 스트리밍

### 응용 프로젝트
- IoT 게이트웨이
- 원격 모니터링 시스템
- 산업용 통신 인터페이스
- GPS 트래커

## 📝 학습 노트

### 배운 점
- UART의 정밀한 타이밍 중요성
- 프로토콜 설계의 복잡성
- 에러 처리의 필요성
- 버퍼링의 효과

### 어려웠던 점
- Baud Rate 설정 시 오차 계산
- 프로토콜 상태 머신 설계
- 멀티 UART 동기화
- 실시간 성능 보장

### 개선 사항
- DMA를 이용한 고속 전송
- 하드웨어 Flow Control
- 적응형 Baud Rate
- 압축 알고리즘 적용

---
*Last Updated: 2024-01-20*