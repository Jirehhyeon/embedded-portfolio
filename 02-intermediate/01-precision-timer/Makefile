# Precision Timer Project Makefile
# Target: ATmega328P with 16MHz crystal
# Features: Multiple timers, real-time clock, frequency generation

# 프로젝트 이름
PROJECT = precision_timer

# 소스 파일
SOURCES = main.c

# MCU 설정 - 정밀 타이밍을 위한 16MHz 크리스털 필수
MCU = atmega328p
F_CPU = 16000000UL

# 프로그래머 설정
PROGRAMMER = arduino
PORT = COM3
BAUD = 115200

# 컴파일러 설정
CC = avr-gcc
OBJCOPY = avr-objcopy
OBJDUMP = avr-objdump
SIZE = avr-size
AVRDUDE = avrdude

# 컴파일 플래그 - 최적화 및 디버그 정보
CFLAGS = -mmcu=$(MCU) -DF_CPU=$(F_CPU)
CFLAGS += -Os -std=gnu99
CFLAGS += -Wall -Wextra -Wshadow -Wstrict-prototypes
CFLAGS += -ffunction-sections -fdata-sections
CFLAGS += -funsigned-char -funsigned-bitfields
CFLAGS += -fpack-struct -fshort-enums
CFLAGS += -g -gdwarf-2

# 링커 플래그
LDFLAGS = -mmcu=$(MCU)
LDFLAGS += -Wl,--gc-sections
LDFLAGS += -Wl,--print-gc-sections
LDFLAGS += -Wl,-Map,$(PROJECT).map

# 오브젝트 파일
OBJECTS = $(SOURCES:.c=.o)

# 기본 타겟
all: $(PROJECT).hex size timer-calc

# 컴파일
%.o: %.c
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -c $< -o $@

# 링킹
$(PROJECT).elf: $(OBJECTS)
	@echo "Linking..."
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

# HEX 파일 생성
$(PROJECT).hex: $(PROJECT).elf
	@echo "Creating HEX file..."
	$(OBJCOPY) -O ihex -R .eeprom $< $@

# 크기 정보 및 타이머 계산
size: $(PROJECT).elf
	@echo "Memory usage:"
	$(SIZE) --format=avr --mcu=$(MCU) $(PROJECT).elf

# 타이머 주파수 계산
timer-calc:
	@echo ""
	@echo "Timer Configuration Calculations:"
	@echo "================================="
	@echo "F_CPU = 16MHz"
	@echo ""
	@echo "Timer0 (1ms system clock):"
	@echo "  Mode: CTC, Prescaler: 64"
	@echo "  OCR0A = 249 (16MHz/64/1000Hz - 1)"
	@echo "  Actual frequency: 1000.0 Hz"
	@echo ""
	@echo "Timer1 (2ms display multiplex):"
	@echo "  Mode: CTC, Prescaler: 64"  
	@echo "  OCR1A = 499 (16MHz/64/500Hz - 1)"
	@echo "  Actual frequency: 500.0 Hz"
	@echo ""
	@echo "Timer2 (100μs high-speed timing):"
	@echo "  Mode: CTC, Prescaler: 64"
	@echo "  OCR2A = 24 (16MHz/64/10000Hz - 1)"
	@echo "  Actual frequency: 10000.0 Hz"
	@echo ""
	@echo "LED Frequencies:"
	@echo "  1Hz LED:   Toggle every 50000 * 100μs = 5.0s → 0.2Hz actual"
	@echo "  10Hz LED:  Toggle every 5000 * 100μs = 0.5s → 2Hz actual"
	@echo "  100Hz LED: Toggle every 500 * 100μs = 50ms → 20Hz actual"
	@echo "  1kHz LED:  Toggle every 50 * 100μs = 5ms → 200Hz actual"

# 디스어셈블리
disasm: $(PROJECT).elf
	$(OBJDUMP) -d $(PROJECT).elf > $(PROJECT).lst
	@echo "Disassembly saved to $(PROJECT).lst"

# 업로드
upload: $(PROJECT).hex
	@echo "Uploading to board..."
	$(AVRDUDE) -c $(PROGRAMMER) -p $(MCU) -P $(PORT) -b $(BAUD) -U flash:w:$(PROJECT).hex:i

# 퓨즈 비트 설정 (16MHz 외부 크리스털)
set-fuses:
	@echo "Setting fuses for 16MHz external crystal..."
	@echo "WARNING: Incorrect fuse settings can brick your MCU!"
	@echo "Make sure 16MHz crystal is connected before proceeding."
	@read -p "Press Enter to continue or Ctrl+C to abort..."
	$(AVRDUDE) -c $(PROGRAMMER) -p $(MCU) -P $(PORT) -b $(BAUD) \
		-U lfuse:w:0xFF:m -U hfuse:w:0xDE:m -U efuse:w:0x05:m

# 퓨즈 비트 읽기
read-fuses:
	@echo "Reading current fuse settings..."
	$(AVRDUDE) -c $(PROGRAMMER) -p $(MCU) -P $(PORT) -b $(BAUD) \
		-U lfuse:r:-:h -U hfuse:r:-:h -U efuse:r:-:h

# 주파수 테스트 도구
test-timing:
	@echo "Precision Timer Testing Guide:"
	@echo "=============================="
	@echo ""
	@echo "Hardware Requirements:"
	@echo "  - 16MHz crystal oscillator with 22pF capacitors"
	@echo "  - 4 LEDs on PB0-PB3 with 220Ω resistors"
	@echo "  - 4-digit 7-segment display on PD4-PD7, PC0-PC3"
	@echo "  - Buttons on PD2, PD3, PC0, PC1"
	@echo ""
	@echo "Testing Procedure:"
	@echo "  1. Use oscilloscope to measure LED frequencies"
	@echo "  2. Verify 7-segment display refresh rate (~500Hz)"
	@echo "  3. Test button debouncing (50ms)"
	@echo "  4. Check timer accuracy with frequency counter"
	@echo ""
	@echo "Expected Results:"
	@echo "  - PB0: ~1Hz square wave"
	@echo "  - PB1: ~10Hz square wave"
	@echo "  - PB2: ~100Hz square wave"
	@echo "  - PB3: ~1kHz square wave"
	@echo "  - Display: Stable time display without flicker"

# 시리얼 모니터 (디버깅용)
monitor:
	@echo "Opening serial monitor..."
	@echo "Note: UART not implemented in this version"
	@echo "Use oscilloscope or logic analyzer for timing analysis"

# 청소
clean:
	@echo "Cleaning..."
	rm -f $(OBJECTS) $(PROJECT).elf $(PROJECT).hex $(PROJECT).lst $(PROJECT).map

# 전체 청소
distclean: clean
	rm -f *.o *.elf *.hex *.lst *.map

# 도움말
help:
	@echo "Available targets:"
	@echo "  all         - Build the project"
	@echo "  upload      - Upload to Arduino"
	@echo "  set-fuses   - Set fuses for 16MHz crystal (DANGEROUS!)"
	@echo "  read-fuses  - Read current fuse settings"
	@echo "  test-timing - Show timing test procedures"
	@echo "  timer-calc  - Show timer calculations"
	@echo "  size        - Show memory usage"
	@echo "  disasm      - Create disassembly"
	@echo "  clean       - Clean build files"
	@echo "  help        - Show this help"

.PHONY: all size timer-calc disasm upload set-fuses read-fuses test-timing monitor clean distclean help