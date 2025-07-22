# 🚀 Embedded Systems Portfolio

[![GitHub](https://img.shields.io/badge/Platform-ATmega328P-blue)]()
[![Language](https://img.shields.io/badge/Language-C%2FC%2B%2B-orange)]()
[![IDE](https://img.shields.io/badge/IDE-Arduino%20%7C%20PlatformIO-green)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()

> 임베디드 시스템 개발자를 꿈꾸며 체계적으로 학습한 프로젝트 포트폴리오입니다.

## 👨‍💻 About Me

- **목표**: 임베디드 시스템 개발자
- **주력 분야**: MCU 프로그래밍, 실시간 시스템, IoT
- **사용 기술**: C/C++, AVR, STM32, FreeRTOS, 각종 통신 프로토콜

## 📚 Learning Path

### [Stage 1: Basic (기초)](./01-basic)
- **기간**: 4주
- **핵심 기술**: GPIO, 디지털 I/O, 기본 타이밍
- **프로젝트**:
  - 🔴 [LED Control](./01-basic/01-led-control) - GPIO 제어 기초
  - 🔘 [Button Input](./01-basic/02-button-input) - 디지털 입력 및 디바운싱
  - 🌈 [PWM Brightness](./01-basic/03-pwm-brightness) - PWM을 이용한 LED 밝기 제어

### [Stage 2: Intermediate (중급)](./02-intermediate)
- **기간**: 4주
- **핵심 기술**: 타이머/카운터, 인터럽트, ADC
- **프로젝트**:
  - ⏱️ [Precision Timer](./02-intermediate/01-precision-timer) - 타이머 직접 제어
  - 🎯 [Interrupt Handler](./02-intermediate/02-interrupt-handler) - 외부/타이머 인터럽트
  - 📊 [ADC Data Logger](./02-intermediate/03-adc-logger) - 아날로그 센서 읽기

### [Stage 3: Advanced (고급)](./03-advanced)
- **기간**: 4주
- **핵심 기술**: UART, I2C, SPI 통신
- **프로젝트**:
  - 📡 [UART Terminal](./03-advanced/01-uart-terminal) - 시리얼 통신 구현
  - 🌡️ [I2C Sensor Network](./03-advanced/02-i2c-sensors) - 다중 센서 연결
  - 💾 [SPI SD Card](./03-advanced/03-spi-sdcard) - SPI 통신으로 SD카드 제어

### [Stage 4: Expert (전문가)](./04-expert)
- **기간**: 4주
- **핵심 기술**: RTOS, 실시간 제어, 최적화
- **프로젝트**:
  - 🔄 [FreeRTOS Multitasking](./04-expert/01-freertos-basic) - RTOS 멀티태스킹
  - 🎛️ [PID Controller](./04-expert/02-pid-controller) - 실시간 PID 제어
  - ⚡ [Power Management](./04-expert/03-power-management) - 저전력 최적화

### [Stage 5: Master (마스터)](./05-master)
- **기간**: 4주
- **핵심 기술**: 시스템 통합, IoT, 분산 제어
- **프로젝트**:
  - 🌐 [IoT Weather Station](./05-master/01-iot-weather) - 완전한 IoT 시스템
  - 🤖 [Robot Control System](./05-master/02-robot-control) - 로봇 제어 시스템
  - 🏭 [Industrial Monitor](./05-master/03-industrial-monitor) - 산업용 모니터링

## 🛠️ Development Environment

### Hardware
- **주 MCU**: ATmega328P (Arduino Uno/Nano)
- **보조 MCU**: STM32F103, ESP32
- **개발 보드**: Arduino Uno, STM32 Nucleo, ESP32 DevKit
- **측정 장비**: 오실로스코프, 로직 분석기, 멀티미터

### Software
- **IDE**: Arduino IDE, PlatformIO, STM32CubeIDE
- **컴파일러**: AVR-GCC, ARM-GCC
- **디버거**: AVR-GDB, OpenOCD, J-Link
- **버전 관리**: Git, GitHub

### Tools & Libraries
- **RTOS**: FreeRTOS
- **통신**: Wire.h (I2C), SPI.h, SoftwareSerial
- **센서 라이브러리**: DHT, DS18B20, MPU6050
- **디버깅**: Serial Monitor, Logic Analyzer

## 📊 Skills Matrix

| 기술 영역 | 숙련도 | 프로젝트 |
|---------|--------|---------|
| C Language | ⭐⭐⭐⭐⭐ | 모든 프로젝트 |
| GPIO Control | ⭐⭐⭐⭐⭐ | LED, Button, PWM |
| Timers/Counters | ⭐⭐⭐⭐ | Precision Timer, PWM |
| Interrupts | ⭐⭐⭐⭐ | Interrupt Handler |
| UART | ⭐⭐⭐⭐ | UART Terminal |
| I2C | ⭐⭐⭐⭐ | Sensor Network |
| SPI | ⭐⭐⭐ | SD Card Interface |
| RTOS | ⭐⭐⭐ | FreeRTOS Projects |
| PID Control | ⭐⭐⭐ | PID Controller |
| IoT | ⭐⭐⭐ | Weather Station |

## 📈 Learning Progress

```
[##########] Stage 1: Basic - 100% Complete
[########--] Stage 2: Intermediate - 80% Complete  
[######----] Stage 3: Advanced - 60% Complete
[####------] Stage 4: Expert - 40% Complete
[##--------] Stage 5: Master - 20% Complete
```

## 🏆 Achievements

- ✅ 50개 이상의 임베디드 프로젝트 완성
- ✅ 데이터시트 기반 개발 능력 습득
- ✅ 실시간 시스템 설계 경험
- ✅ 다양한 통신 프로토콜 구현
- ✅ RTOS 기반 멀티태스킹 구현

## 📚 Study Resources

- [AVR Datasheet](./docs/datasheets/)
- [개발 노트](./docs/notes/)
- [회로도 모음](./docs/schematics/)
- [유용한 도구들](./tools/)

## 🤝 Contact

- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn Profile]
- **Blog**: [Your Tech Blog]

---

*"The best way to predict the future is to invent it." - Alan Kay*