# PWM Brightness Control Project Makefile
# Target: ATmega328P
# Features: PWM generation, RGB LED control, ADC input

# 프로젝트 이름
PROJECT = pwm_brightness

# 소스 파일
SOURCES = main.c

# MCU 설정
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

# 수학 라이브러리 사용
LDLIBS = -lm

# 컴파일 플래그
CFLAGS = -mmcu=$(MCU) -DF_CPU=$(F_CPU)
CFLAGS += -Os -std=gnu99
CFLAGS += -Wall -Wextra -Wshadow -Wstrict-prototypes
CFLAGS += -ffunction-sections -fdata-sections
CFLAGS += -funsigned-char -funsigned-bitfields
CFLAGS += -fpack-struct -fshort-enums

# 링커 플래그
LDFLAGS = -mmcu=$(MCU)
LDFLAGS += -Wl,--gc-sections
LDFLAGS += -Wl,--print-gc-sections
LDFLAGS += $(LDLIBS)

# 오브젝트 파일
OBJECTS = $(SOURCES:.c=.o)

# 기본 타겟
all: $(PROJECT).hex size

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

# 크기 정보 출력
size: $(PROJECT).elf
	@echo "Memory usage:"
	$(SIZE) --format=avr --mcu=$(MCU) $(PROJECT).elf
	@echo "PWM frequency calculation:"
	@echo "Timer0/1 (Fast PWM): 16MHz / (8 × 256) = 7812.5 Hz"

# 디스어셈블리
disasm: $(PROJECT).elf
	$(OBJDUMP) -d $(PROJECT).elf > $(PROJECT).lst
	@echo "Disassembly saved to $(PROJECT).lst"

# 업로드
upload: $(PROJECT).hex
	@echo "Uploading to board..."
	$(AVRDUDE) -c $(PROGRAMMER) -p $(MCU) -P $(PORT) -b $(BAUD) -U flash:w:$(PROJECT).hex:i

# 시리얼 모니터 (ADC 값 확인용)
monitor:
	@echo "Opening serial monitor (if UART debugging enabled)..."
	@echo "Press Ctrl+C to exit"
	python -m serial.tools.miniterm $(PORT) 9600

# PWM 신호 테스트 도구
test-pwm:
	@echo "PWM Test Commands:"
	@echo "Use oscilloscope to measure:"
	@echo "- Red LED (Pin 6): OC0A output"
	@echo "- Green LED (Pin 5): OC0B output"  
	@echo "- Blue LED (Pin 9): OC1A output"
	@echo "Expected frequency: ~7.8 kHz"

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
	@echo "  all       - Build the project"
	@echo "  upload    - Upload to Arduino"
	@echo "  monitor   - Open serial monitor"
	@echo "  test-pwm  - Show PWM testing guide"
	@echo "  size      - Show memory usage"
	@echo "  disasm    - Create disassembly"
	@echo "  clean     - Clean build files"
	@echo "  help      - Show this help"

.PHONY: all size disasm upload monitor test-pwm clean distclean help