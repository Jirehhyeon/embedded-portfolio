# Advanced UART Terminal Project Makefile
# Target: ATmega328P
# Features: Multi-UART protocol with hardware and software UART

PROJECT = uart_terminal
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

# Math library required for CRC calculations
LDLIBS = -lm

# Communication-optimized compile flags
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
all: $(PROJECT).hex size uart-analysis

# Compile
%.o: %.c
	@echo "Compiling $< with UART optimizations..."
	$(CC) $(CFLAGS) -c $< -o $@

# Link
$(PROJECT).elf: $(OBJECTS)
	@echo "Linking with math library for CRC..."
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

# Generate HEX
$(PROJECT).hex: $(PROJECT).elf
	@echo "Creating HEX file..."
	$(OBJCOPY) -O ihex -R .eeprom $< $@

# Memory usage analysis
size: $(PROJECT).elf
	@echo "Memory usage analysis:"
	$(SIZE) --format=avr --mcu=$(MCU) $(PROJECT).elf

# UART communication analysis
uart-analysis:
	@echo ""
	@echo "UART Terminal System Analysis:"
	@echo "=============================="
	@echo ""
	@echo "Hardware Configuration:"
	@echo "  MCU: ATmega328P @ 16MHz"
	@echo "  Baud Rate: 9600 bps"
	@echo "  Frame Format: 8N1 (8 data, no parity, 1 stop)"
	@echo "  UBRR Value: $$(((16000000/16/9600)-1)) = 103"
	@echo ""
	@echo "UART Interfaces:"
	@echo "  Interface     Pins      Type       Purpose"
	@echo "  -----------   -------   --------   ---------------------"
	@echo "  Hardware      PD0/PD1   HW UART    PC Communication"
	@echo "  Software #1   PD2/PD3   SW UART    Bluetooth (HC-05)"
	@echo "  Software #2   PD4/PD5   SW UART    GPS Module (RX only)"
	@echo ""
	@echo "Protocol Support:"
	@echo "  - Binary Protocol with CRC16 error checking"
	@echo "  - AT Command interface for human interaction"
	@echo "  - Multi-device coordination and switching"
	@echo ""
	@echo "Ring Buffer Configuration:"
	@echo "  Hardware UART: 128 bytes (TX + RX)"
	@echo "  Software UART: 64 bytes each"
	@echo "  Protocol Packets: Max 64 bytes payload"
	@echo ""
	@echo "Command Set:"
	@echo "  Binary Commands     AT Commands"
	@echo "  ---------------     -----------"
	@echo "  CMD_PING           AT+PING"
	@echo "  CMD_GET_STATUS     AT+STATUS"
	@echo "  CMD_READ_SENSOR    AT+WIFI"
	@echo "  CMD_WIFI_CONNECT   AT+BT"
	@echo "  CMD_GPS_REQUEST    AT+GPS"
	@echo "  CMD_RESET          AT+STATS"

# Protocol testing tools
test-protocol:
	@echo "Protocol Testing Guide:"
	@echo "======================="
	@echo ""
	@echo "Required Test Equipment:"
	@echo "  - USB-to-Serial adapter (FTDI, CH340, etc.)"
	@echo "  - ESP8266 WiFi module"
	@echo "  - HC-05 Bluetooth module"
	@echo "  - GPS module (NEO-6M or similar)"
	@echo "  - Terminal software (PuTTY, CoolTerm, etc.)"
	@echo ""
	@echo "Test Connections:"
	@echo "  PC ←→ USB-Serial ←→ ATmega328P (PD0/PD1)"
	@echo "  ATmega328P (PD2/PD3) ←→ HC-05 Bluetooth"
	@echo "  ATmega328P (PD4/PD5) ←→ GPS Module"
	@echo "  ATmega328P (PC4) → ESP8266 Reset"
	@echo ""
	@echo "AT Command Testing:"
	@echo "  1. Connect terminal to COM port at 9600 baud"
	@echo "  2. Send: AT+PING (should respond: PONG)"
	@echo "  3. Send: AT+STATUS (system status)"
	@echo "  4. Send: AT+WIFI=CONNECT (WiFi connection)"
	@echo "  5. Send: AT+GPS=START (GPS activation)"
	@echo "  6. Send: AT+STATS (communication statistics)"
	@echo ""
	@echo "Binary Protocol Testing:"
	@echo "  Packet Format: [AA][CMD][LEN][PAYLOAD][CRC_H][CRC_L][55]"
	@echo "  Example PING: AA 01 00 <CRC16> 55"
	@echo "  Use hex editor or custom test program"

# Communication debugging
debug-uart:
	@echo "UART Debugging Tools:"
	@echo "===================="
	@echo ""
	@echo "1. Logic Analyzer Connections:"
	@echo "   Ch0: PD0 (HW UART RX)"
	@echo "   Ch1: PD1 (HW UART TX)"
	@echo "   Ch2: PD2 (BT UART TX)"
	@echo "   Ch3: PD3 (BT UART RX)"
	@echo "   Ch4: PD4 (GPS UART TX)"
	@echo "   Ch5: PD5 (GPS UART RX)"
	@echo ""
	@echo "2. Oscilloscope Test Points:"
	@echo "   TP1: PD1 (TX signal analysis)"
	@echo "   TP2: PB0 (Status LED - activity indicator)"
	@echo "   TP3: PB1 (WiFi LED - connection status)"
	@echo "   TP4: PB2 (GPS LED - lock indicator)"
	@echo ""
	@echo "3. Serial Monitor Settings:"
	@echo "   Baud Rate: 9600"
	@echo "   Data Bits: 8"
	@echo "   Parity: None"
	@echo "   Stop Bits: 1"
	@echo "   Flow Control: None"
	@echo ""
	@echo "4. Common Issues:"
	@echo "   - Wrong baud rate calculation"
	@echo "   - Buffer overflow in ring buffers"
	@echo "   - CRC mismatch in binary protocol"
	@echo "   - Software UART timing issues"
	@echo "   - Interrupt conflicts"

# Performance analysis
analyze-performance:
	@echo "UART Performance Analysis:"
	@echo "=========================="
	@echo ""
	@echo "Throughput Calculations:"
	@echo "  Hardware UART @ 9600 bps:"
	@echo "    Theoretical: 960 bytes/sec"
	@echo "    With 8N1: 960 bytes/sec"
	@echo "    With protocol overhead: ~800 bytes/sec"
	@echo ""
	@echo "  Software UART @ 9600 bps:"
	@echo "    Bit-banging overhead: ~10% CPU @ 16MHz"
	@echo "    Timing accuracy: ±2% (acceptable)"
	@echo "    Concurrent operation: Limited by CPU"
	@echo ""
	@echo "Buffer Management:"
	@echo "  Ring Buffer Efficiency: O(1) for all operations"
	@echo "  Memory Usage: 320 bytes total (all buffers)"
	@echo "  Overflow Protection: Built-in bounds checking"
	@echo ""
	@echo "Protocol Efficiency:"
	@echo "  Binary Protocol Overhead: 7 bytes per packet"
	@echo "  CRC16 Calculation: ~50 cycles per byte"
	@echo "  AT Command Parsing: Variable (20-100 cycles)"
	@echo ""
	@echo "Interrupt Load:"
	@echo "  RX Interrupt: ~20 cycles per byte"
	@echo "  TX Interrupt: ~15 cycles per byte"
	@echo "  Timer Interrupt: ~40 cycles (1Hz system tick)"
	@echo "  Total: <5% CPU usage at full UART speed"

# CRC calculation verification
test-crc:
	@echo "CRC16 Verification:"
	@echo "=================="
	@echo ""
	@echo "Test Vectors for CRC16-CCITT:"
	@echo "  Input: \"123456789\""
	@echo "  Expected CRC: 0x29B1"
	@echo ""
	@echo "Protocol Test Packets:"
	@echo "  PING Command:"
	@echo "    Data: 01 00 (CMD_PING, 0 length)"
	@echo "    CRC16: 0x1021 (may vary with implementation)"
	@echo "    Full packet: AA 01 00 10 21 55"
	@echo ""
	@echo "  STATUS Command:"
	@echo "    Data: 02 00 (CMD_GET_STATUS, 0 length)"
	@echo "    Calculate CRC with your implementation"
	@echo ""
	@echo "CRC Debug Tips:"
	@echo "  - Verify initial value (0xFFFF)"
	@echo "  - Check polynomial (0x1021 for CCITT)"
	@echo "  - Ensure correct byte order in packet"
	@echo "  - Test with known good CRC calculator"

# Disassembly with UART focus
disasm: $(PROJECT).elf
	$(OBJDUMP) -d $(PROJECT).elf > $(PROJECT).lst
	@echo "Disassembly with UART analysis saved to $(PROJECT).lst"
	@echo ""
	@echo "Key symbols to look for:"
	@echo "  - USART_RX_vect: Hardware UART RX interrupt"
	@echo "  - USART_UDRE_vect: Hardware UART TX interrupt"
	@echo "  - hw_uart_putchar: Main transmission function"
	@echo "  - sw_uart_bt_putchar: Software UART transmission"
	@echo "  - process_protocol_packet: Binary protocol handler"

# Upload to board
upload: $(PROJECT).hex
	@echo "Uploading UART terminal system..."
	$(AVRDUDE) -c $(PROGRAMMER) -p $(MCU) -P $(PORT) -b $(BAUD) -U flash:w:$(PROJECT).hex:i

# Multi-device testing
test-multidevice:
	@echo "Multi-Device Testing Procedure:"
	@echo "==============================="
	@echo ""
	@echo "1. ESP8266 WiFi Module Test:"
	@echo "   a) Power ESP8266 with 3.3V (use level shifter for TX)"
	@echo "   b) Connect ESP RX to MCU PD1, ESP TX to MCU PD0"
	@echo "   c) Send AT+WIFI=CONNECT command"
	@echo "   d) Verify WiFi LED (PB1) activation"
	@echo ""
	@echo "2. HC-05 Bluetooth Module Test:"
	@echo "   a) Connect HC-05 RX to MCU PD2, HC-05 TX to MCU PD3"
	@echo "   b) Send AT+BT=PAIR command"
	@echo "   c) Use smartphone to discover and pair"
	@echo "   d) Test bidirectional communication"
	@echo ""
	@echo "3. GPS Module Test:"
	@echo "   a) Connect GPS TX to MCU PD5 (RX only)"
	@echo "   b) Send AT+GPS=START command"
	@echo "   c) Wait for GPS lock (may take 30+ seconds outdoors)"
	@echo "   d) Verify GPS LED (PB2) activation"
	@echo ""
	@echo "4. Protocol Switching Test:"
	@echo "   a) Send binary packets and AT commands alternately"
	@echo "   b) Verify proper protocol detection and switching"
	@echo "   c) Check CRC error counter with AT+STATS"
	@echo "   d) Test with corrupted packets to verify error handling"

# Clean targets
clean:
	@echo "Cleaning build files..."
	rm -f $(OBJECTS) $(PROJECT).elf $(PROJECT).hex $(PROJECT).lst $(PROJECT).map

distclean: clean
	rm -f *.o *.elf *.hex *.lst *.map

# Help
help:
	@echo "Available targets:"
	@echo "  all               - Build the project"
	@echo "  upload            - Upload to Arduino"
	@echo "  test-protocol     - Show protocol testing guide"
	@echo "  debug-uart        - Show UART debugging guide"
	@echo "  analyze-performance - Show performance analysis"
	@echo "  test-crc          - Show CRC verification guide"
	@echo "  test-multidevice  - Show multi-device testing"
	@echo "  uart-analysis     - Show UART system analysis"
	@echo "  size              - Show memory usage"
	@echo "  disasm            - Create disassembly"
	@echo "  clean             - Clean build files"
	@echo "  help              - Show this help"

.PHONY: all size uart-analysis disasm upload test-protocol debug-uart analyze-performance test-crc test-multidevice clean distclean help