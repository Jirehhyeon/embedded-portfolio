# LED Control Project Makefile
# Target: ATmega328P
# Programmer: Arduino as ISP

# 프로젝트 이름
PROJECT = led_control

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

# 오브젝트 파일
OBJECTS = $(SOURCES:.c=.o)

# 기본 타겟
all: $(PROJECT).hex size

# 컴파일
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# 링킹
$(PROJECT).elf: $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

# HEX 파일 생성
$(PROJECT).hex: $(PROJECT).elf
	$(OBJCOPY) -O ihex -R .eeprom $< $@

# 크기 정보 출력
size: $(PROJECT).elf
	$(SIZE) --format=avr --mcu=$(MCU) $(PROJECT).elf

# 디스어셈블리
disasm: $(PROJECT).elf
	$(OBJDUMP) -d $(PROJECT).elf

# 업로드
upload: $(PROJECT).hex
	$(AVRDUDE) -c $(PROGRAMMER) -p $(MCU) -P $(PORT) -b $(BAUD) -U flash:w:$(PROJECT).hex:i

# 퓨즈 비트 읽기
read_fuses:
	$(AVRDUDE) -c $(PROGRAMMER) -p $(MCU) -P $(PORT) -b $(BAUD) -U lfuse:r:-:h -U hfuse:r:-:h -U efuse:r:-:h

# 청소
clean:
	rm -f $(OBJECTS) $(PROJECT).elf $(PROJECT).hex $(PROJECT).lst $(PROJECT).map

# 전체 청소
distclean: clean
	rm -f *.o *.elf *.hex *.lst *.map

.PHONY: all size disasm upload read_fuses clean distclean