# Advanced Interrupt Handler Project Makefile
# Target: ATmega328P
# Features: Multi-source interrupts, priority management, real-time control

PROJECT = interrupt_handler
SOURCES = main.c
MCU = atmega328p
F_CPU = 16000000UL

# Programmer settings
PROGRAMMER = arduino
PORT = COM3
BAUD = 115200

# Compiler tools
CC = avr-gcc
OBJCOPY = avr-objcopy
OBJDUMP = avr-objdump
SIZE = avr-size
AVRDUDE = avrdude

# Math library required for PID controller
LDLIBS = -lm

# Optimized compile flags for real-time performance
CFLAGS = -mmcu=$(MCU) -DF_CPU=$(F_CPU)
CFLAGS += -Os -std=gnu99
CFLAGS += -Wall -Wextra -Wshadow -Wstrict-prototypes
CFLAGS += -ffunction-sections -fdata-sections
CFLAGS += -funsigned-char -funsigned-bitfields
CFLAGS += -fpack-struct -fshort-enums
CFLAGS += -g -gdwarf-2

# Linker flags
LDFLAGS = -mmcu=$(MCU)
LDFLAGS += -Wl,--gc-sections
LDFLAGS += -Wl,--print-gc-sections
LDFLAGS += -Wl,-Map,$(PROJECT).map
LDFLAGS += $(LDLIBS)

OBJECTS = $(SOURCES:.c=.o)

# Default target
all: $(PROJECT).hex size interrupt-analysis

# Compile
%.o: %.c
	@echo "Compiling $< with interrupt optimizations..."
	$(CC) $(CFLAGS) -c $< -o $@

# Link
$(PROJECT).elf: $(OBJECTS)
	@echo "Linking with math library..."
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

# Generate HEX
$(PROJECT).hex: $(PROJECT).elf
	@echo "Creating HEX file..."
	$(OBJCOPY) -O ihex -R .eeprom $< $@

# Memory usage analysis
size: $(PROJECT).elf
	@echo "Memory usage analysis:"
	$(SIZE) --format=avr --mcu=$(MCU) $(PROJECT).elf

# Interrupt vector analysis
interrupt-analysis:
	@echo ""
	@echo "Interrupt Vector Analysis:"
	@echo "=========================="
	@echo "ATmega328P Interrupt Vectors (used in this project):"
	@echo ""
	@echo "Vector  Address  Source              Priority  Handler"
	@echo "------  -------  ------------------  --------  ----------------"
	@echo "0x01    0x002    INT0 (PD2)         1         Encoder A Phase"
	@echo "0x02    0x004    INT1 (PD3)         2         Emergency Stop"
	@echo "0x05    0x00A    PCINT2 (PD4)        5         Ultrasonic Echo"
	@echo "0x0B    0x016    TIMER0_COMPA        11        System Tick 1ms"
	@echo "0x0C    0x018    TIMER0_COMPB        12        (PWM only)"
	@echo "0x0D    0x01A    TIMER1_COMPA        13        Control Loop 1ms"
	@echo "0x15    0x02A    ADC                 21        Temperature"
	@echo ""
	@echo "Interrupt Response Times (estimated):"
	@echo "  INT0/INT1:     4-6 cycles (0.25-0.375 μs @ 16MHz)"
	@echo "  PCINT:         4-8 cycles (0.25-0.5 μs @ 16MHz)"
	@echo "  Timer:         3-5 cycles (0.1875-0.3125 μs @ 16MHz)"
	@echo "  ADC:           5-7 cycles (0.3125-0.4375 μs @ 16MHz)"
	@echo ""
	@echo "Maximum Interrupt Load Analysis:"
	@echo "  Timer0 (1ms):     1000 Hz"
	@echo "  Timer1 (1ms):     1000 Hz"
	@echo "  Encoder (max):    ~500 Hz per phase"
	@echo "  Ultrasonic:       ~20 Hz"
	@echo "  ADC:              ~10 Hz"
	@echo "  Total Est.:       ~3.5K interrupts/sec"

# Disassembly with interrupt vector table
disasm: $(PROJECT).elf
	$(OBJDUMP) -d $(PROJECT).elf > $(PROJECT).lst
	@echo "Disassembly with interrupt vectors saved to $(PROJECT).lst"

# Upload to board
upload: $(PROJECT).hex
	@echo "Uploading interrupt handler system..."
	$(AVRDUDE) -c $(PROGRAMMER) -p $(MCU) -P $(PORT) -b $(BAUD) -U flash:w:$(PROJECT).hex:i

# Real-time performance testing
test-timing:
	@echo "Real-time Performance Testing Guide:"
	@echo "====================================="
	@echo ""
	@echo "Required Test Equipment:"
	@echo "  - Oscilloscope (min 4 channels)"
	@echo "  - Logic analyzer (recommended 8+ channels)"
	@echo "  - Rotary encoder with detents"
	@echo "  - HC-SR04 ultrasonic sensor"
	@echo "  - LM35 temperature sensor"
	@echo ""
	@echo "Test Points:"
	@echo "  Channel 1: PB4 (Activity LED) - 1Hz heartbeat"
	@echo "  Channel 2: PD6 (Buzzer PWM) - Variable frequency"
	@echo "  Channel 3: PD5 (LED PWM) - PID controller output"
	@echo "  Channel 4: PD4 (Echo) - Ultrasonic timing"
	@echo ""
	@echo "Performance Verification:"
	@echo "  1. Verify 1ms timer precision (±1% tolerance)"
	@echo "  2. Check interrupt latency (<10 cycles)"
	@echo "  3. Measure ISR execution time"
	@echo "  4. Verify PID control loop timing"
	@echo "  5. Test emergency stop response (<1ms)"
	@echo ""
	@echo "Critical Timing Requirements:"
	@echo "  - Emergency stop response: <1ms"
	@echo "  - PID update rate: 1000Hz (1ms)"
	@echo "  - Encoder response: <100μs"
	@echo "  - Ultrasonic measurement: 20Hz"

# Interrupt debugging tools
debug-interrupts:
	@echo "Interrupt Debugging Tools:"
	@echo "=========================="
	@echo ""
	@echo "1. Enable debug symbols:"
	@echo "   CFLAGS += -g -gdwarf-2"
	@echo ""
	@echo "2. Use avr-gdb with simavr:"
	@echo "   simavr -g -m atmega328p $(PROJECT).hex &"
	@echo "   avr-gdb $(PROJECT).elf"
	@echo ""
	@echo "3. Interrupt monitoring macros already included:"
	@echo "   - interrupt_stats structure"
	@echo "   - ISR_ENTER/ISR_EXIT macros"
	@echo "   - Nesting level tracking"
	@echo "   - Performance counters"
	@echo ""
	@echo "4. Logic analyzer triggers:"
	@echo "   - Set trigger on interrupt pins"
	@echo "   - Monitor ISR execution patterns"
	@echo "   - Check for interrupt conflicts"

# Stack usage analysis
analyze-stack:
	@echo "Stack Usage Analysis:"
	@echo "===================="
	@echo ""
	@echo "ATmega328P Stack Information:"
	@echo "  Total RAM: 2048 bytes"
	@echo "  Stack grows downward from 0x08FF"
	@echo ""
	@echo "Estimated Stack Usage:"
	@echo "  Main context: ~32 bytes"
	@echo "  ISR context (each): ~10-20 bytes"
	@echo "  Max nesting (4 levels): ~80 bytes"
	@echo "  PID calculations: ~16 bytes"
	@echo "  Safety margin: 100 bytes"
	@echo "  Total required: ~250 bytes"
	@echo ""
	@echo "Remaining for data: ~1800 bytes"
	@echo ""
	@echo "Stack overflow protection:"
	@echo "  - Limit interrupt nesting to 4 levels"
	@echo "  - Monitor interrupt_nesting_level"
	@echo "  - Use atomic operations where possible"

# Clean targets
clean:
	@echo "Cleaning build files..."
	rm -f $(OBJECTS) $(PROJECT).elf $(PROJECT).hex $(PROJECT).lst $(PROJECT).map

distclean: clean
	rm -f *.o *.elf *.hex *.lst *.map

# Help
help:
	@echo "Available targets:"
	@echo "  all              - Build the project"
	@echo "  upload           - Upload to Arduino"
	@echo "  test-timing      - Show timing test procedures"
	@echo "  debug-interrupts - Show interrupt debugging guide"
	@echo "  analyze-stack    - Show stack usage analysis"
	@echo "  interrupt-analysis - Show interrupt vector info"
	@echo "  size             - Show memory usage"
	@echo "  disasm           - Create disassembly"
	@echo "  clean            - Clean build files"
	@echo "  help             - Show this help"

.PHONY: all size interrupt-analysis disasm upload test-timing debug-interrupts analyze-stack clean distclean help