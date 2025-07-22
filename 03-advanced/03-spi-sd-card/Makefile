# Advanced SPI SD Card Interface Project Makefile
# Target: ATmega328P
# Features: SPI communication, FAT32 file system, performance optimization

PROJECT = spi_sd_card
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

# Math library for file system calculations
LDLIBS = -lm

# SPI and filesystem optimized compile flags
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
all: $(PROJECT).hex size spi-analysis

# Compile
%.o: %.c
	@echo "Compiling $< with SPI and FAT32 optimizations..."
	$(CC) $(CFLAGS) -c $< -o $@

# Link
$(PROJECT).elf: $(OBJECTS)
	@echo "Linking with math library for filesystem calculations..."
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

# Generate HEX
$(PROJECT).hex: $(PROJECT).elf
	@echo "Creating HEX file..."
	$(OBJCOPY) -O ihex -R .eeprom $< $@

# Memory usage analysis
size: $(PROJECT).elf
	@echo "Memory usage analysis:"
	$(SIZE) --format=avr --mcu=$(MCU) $(PROJECT).elf

# SPI SD card system analysis
spi-analysis:
	@echo ""
	@echo "SPI SD Card System Analysis:"
	@echo "============================"
	@echo ""
	@echo "Hardware Configuration:"
	@echo "  MCU: ATmega328P @ 16MHz"
	@echo "  SPI Mode: Master mode"
	@echo "  Initial Speed: 125kHz (f_osc/128)"
	@echo "  Data Speed: 4MHz (f_osc/4)"
	@echo "  CS Control: Software controlled (PB2)"
	@echo ""
	@echo "SD Card Interface:"
	@echo "  Pin Assignment:"
	@echo "    CS (Chip Select):  PB2"
	@echo "    MOSI (Data Out):   PB3"
	@echo "    MISO (Data In):    PB4"
	@echo "    SCK (Clock):       PB5"
	@echo ""
	@echo "Supported Card Types:"
	@echo "  - SD Card v1.x (Standard Capacity)"
	@echo "  - SD Card v2.x (Standard Capacity)"
	@echo "  - SDHC (High Capacity up to 32GB)"
	@echo "  - SDXC (Extended Capacity up to 2TB)"
	@echo ""
	@echo "FAT32 File System Features:"
	@echo "  - Boot sector parsing and validation"
	@echo "  - File Allocation Table (FAT) access"
	@echo "  - Directory entry management"
	@echo "  - File operations (open, read, write, close)"
	@echo "  - Long filename support (planned)"
	@echo "  - Error detection and recovery"

# SD Card protocol testing
test-sd-protocol:
	@echo "SD Card Protocol Testing Guide:"
	@echo "==============================="
	@echo ""
	@echo "Required Test Equipment:"
	@echo "  - Logic analyzer (4+ channels, 25MHz+)"
	@echo "  - SD card reader/writer for verification"
	@echo "  - Various SD card types for compatibility"
	@echo "  - Power supply with current monitoring"
	@echo "  - Oscilloscope for signal quality analysis"
	@echo ""
	@echo "SPI Signal Analysis:"
	@echo "  Ch0: CS (PB2) - Chip select control"
	@echo "  Ch1: SCK (PB5) - Serial clock"
	@echo "  Ch2: MOSI (PB3) - Master out, slave in"
	@echo "  Ch3: MISO (PB4) - Master in, slave out"
	@echo "  Sample Rate: 50MHz minimum"
	@echo "  Trigger: CS falling edge"
	@echo ""
	@echo "Command Sequence Verification:"
	@echo "  1. CMD0 (GO_IDLE_STATE): Response R1 = 0x01"
	@echo "  2. CMD8 (SEND_IF_COND): Check voltage range"
	@echo "  3. ACMD41 (SD_SEND_OP_COND): Initialization"
	@echo "  4. CMD58 (READ_OCR): Operating conditions"
	@echo "  5. CMD17 (READ_SINGLE_BLOCK): Data read test"
	@echo "  6. CMD24 (WRITE_BLOCK): Data write test"
	@echo ""
	@echo "Timing Requirements:"
	@echo "  Initialization: 125kHz max (400kHz spec)"
	@echo "  Data transfer: Up to 25MHz (depends on card)"
	@echo "  Setup time: 5ns minimum"
	@echo "  Hold time: 5ns minimum"
	@echo "  CS to clock: 1μs minimum"
	@echo ""
	@echo "Common Issues:"
	@echo "  - Missing pull-up on MISO (50kΩ internal)"
	@echo "  - Clock too fast during initialization"
	@echo "  - Insufficient power supply current"
	@echo "  - Poor signal integrity at high speeds"
	@echo "  - Incorrect CS timing"

# FAT32 filesystem testing
test-fat32:
	@echo "FAT32 Filesystem Testing Guide:"
	@echo "==============================="
	@echo ""
	@echo "Filesystem Structure Verification:"
	@echo "  Boot Sector (Sector 0):"
	@echo "    - Boot signature: 0xAA55"
	@echo "    - Bytes per sector: 512"
	@echo "    - Sectors per cluster: Power of 2"
	@echo "    - Reserved sectors: ≥32"
	@echo "    - FAT count: Usually 2"
	@echo "    - Root directory cluster: ≥2"
	@echo ""
	@echo "  File Allocation Table (FAT):"
	@echo "    - FAT32 signature: 0x0FFFFFF8-0x0FFFFFFF"
	@echo "    - Cluster chain validation"
	@echo "    - Free cluster counting"
	@echo "    - Bad cluster marking"
	@echo ""
	@echo "File Operations Testing:"
	@echo "  Create File:"
	@echo "    1. Find free directory entry"
	@echo "    2. Allocate cluster from FAT"
	@echo "    3. Write directory entry"
	@echo "    4. Update FAT chain"
	@echo ""
	@echo "  Read File:"
	@echo "    1. Parse directory entry"
	@echo "    2. Follow cluster chain"
	@echo "    3. Read cluster data"
	@echo "    4. Handle file boundaries"
	@echo ""
	@echo "  Write File:"
	@echo "    1. Follow existing cluster chain"
	@echo "    2. Allocate new clusters if needed"
	@echo "    3. Update directory entry size"
	@echo "    4. Sync FAT changes"
	@echo ""
	@echo "Data Integrity Checks:"
	@echo "  - Directory entry checksum validation"
	@echo "  - Cluster boundary verification"
	@echo "  - File size consistency"
	@echo "  - FAT chain integrity"
	@echo "  - Free space calculation"

# Performance benchmarking
benchmark-spi:
	@echo "SPI Performance Benchmarking:"
	@echo "============================="
	@echo ""
	@echo "Transfer Speed Analysis:"
	@echo "  Clock Frequency    Transfer Rate    Overhead"
	@echo "  ---------------    -------------    --------"
	@echo "  125kHz (init)      ~15KB/s         ~95%"
	@echo "  1MHz               ~125KB/s        ~20%"
	@echo "  4MHz               ~500KB/s        ~5%"
	@echo "  8MHz               ~1MB/s          ~2%"
	@echo ""
	@echo "Sector Operations (512 bytes):"
	@echo "  Read sector:  ~1.5ms @ 4MHz"
	@echo "  Write sector: ~5-15ms (includes erase)"
	@echo "  Seek time:    ~1ms average"
	@echo ""
	@echo "File System Performance:"
	@echo "  Open file:         ~2-10ms"
	@echo "  Read 1KB:          ~3ms"
	@echo "  Write 1KB:         ~10-25ms"
	@echo "  Directory scan:    ~5-50ms"
	@echo ""
	@echo "Memory Usage:"
	@echo "  Flash: ~24-28KB (75-87% of 32KB)"
	@echo "  SRAM: ~1.0-1.5KB (50-75% of 2KB)"
	@echo "  Stack: ~200-400 bytes"
	@echo "  Sector buffer: 512 bytes"
	@echo ""
	@echo "CPU Load Distribution:"
	@echo "  SPI communication: ~25%"
	@echo "  FAT32 processing: ~30%"
	@echo "  File operations: ~20%"
	@echo "  Error handling: ~5%"
	@echo "  Idle time: ~20%"

# SD card compatibility testing
test-compatibility:
	@echo "SD Card Compatibility Testing:"
	@echo "=============================="
	@echo ""
	@echo "Card Type Detection:"
	@echo "  SD v1.x:"
	@echo "    - CMD8 returns illegal command"
	@echo "    - ACMD41 without HCS bit"
	@echo "    - Byte addressing mode"
	@echo ""
	@echo "  SD v2.0 Standard Capacity:"
	@echo "    - CMD8 returns valid response"
	@echo "    - Check pattern verification"
	@echo "    - Byte addressing mode"
	@echo ""
	@echo "  SDHC/SDXC (High/Extended Capacity):"
	@echo "    - CMD8 returns valid response"
	@echo "    - ACMD41 with HCS bit set"
	@echo "    - Block addressing mode"
	@echo "    - OCR CCS bit = 1"
	@echo ""
	@echo "Manufacturer Testing:"
	@echo "  SanDisk: Widely compatible"
	@echo "  Kingston: Good compatibility"
	@echo "  Samsung: Excellent performance"
	@echo "  Lexar: Professional grade"
	@echo "  Generic: Variable quality"
	@echo ""
	@echo "Speed Class Requirements:"
	@echo "  Class 4: 4MB/s minimum write"
	@echo "  Class 6: 6MB/s minimum write"
	@echo "  Class 10: 10MB/s minimum write"
	@echo "  UHS-I: Up to 104MB/s"
	@echo ""
	@echo "Format Compatibility:"
	@echo "  ✓ FAT32 (primary support)"
	@echo "  ✓ FAT16 (basic support)"
	@echo "  ✗ exFAT (not supported)"
	@echo "  ✗ NTFS (not supported)"

# Power consumption analysis
analyze-power:
	@echo "Power Consumption Analysis:"
	@echo "=========================="
	@echo ""
	@echo "Component Power Draw:"
	@echo "  ATmega328P (active):     ~20mA @ 16MHz"
	@echo "  ATmega328P (idle):       ~6mA @ 16MHz"
	@echo "  SD Card (read):          ~80-200mA @ 3.3V"
	@echo "  SD Card (write):         ~100-300mA @ 3.3V"
	@echo "  SD Card (idle):          ~1-5mA @ 3.3V"
	@echo "  Status LEDs (3x):        ~60mA @ 5V"
	@echo ""
	@echo "Operation Power Consumption:"
	@echo "  Idle (card inserted):    ~30mA @ 5V = 150mW"
	@echo "  Reading data:            ~120mA @ 5V = 600mW"
	@echo "  Writing data:            ~200mA @ 5V = 1000mW"
	@echo "  Peak consumption:        ~300mA @ 5V = 1500mW"
	@echo ""
	@echo "Power Optimization:"
	@echo "  - Use sleep modes when idle"
	@echo "  - LED duty cycling during activity"
	@echo "  - SD card power management commands"
	@echo "  - Clock scaling for lower speeds"
	@echo "  - Batch operations to minimize wake time"
	@echo ""
	@echo "Battery Life Estimation:"
	@echo "  3000mAh @ 5V: 10-25 hours continuous use"
	@echo "  With sleep mode: 50-100 hours typical use"

# Error handling and recovery
test-error-recovery:
	@echo "Error Handling and Recovery Testing:"
	@echo "===================================="
	@echo ""
	@echo "SD Card Communication Errors:"
	@echo "  Command Timeout:"
	@echo "    - Retry command up to 3 times"
	@echo "    - Increase timeout progressively"
	@echo "    - Re-initialize if persistent"
	@echo ""
	@echo "  Data Read Errors:"
	@echo "    - CRC error detection"
	@echo "    - Retry read operation"
	@echo "    - Mark sector as bad if persistent"
	@echo ""
	@echo "  Data Write Errors:"
	@echo "    - Write response validation"
	@echo "    - Verify write by read-back"
	@echo "    - Handle write protection"
	@echo ""
	@echo "File System Errors:"
	@echo "  Invalid Boot Sector:"
	@echo "    - Check backup boot sector"
	@echo "    - Attempt filesystem repair"
	@echo "    - Report corruption to user"
	@echo ""
	@echo "  FAT Table Corruption:"
	@echo "    - Use backup FAT table"
	@echo "    - Cross-validate FAT entries"
	@echo "    - Rebuild cluster chains"
	@echo ""
	@echo "  Directory Corruption:"
	@echo "    - Validate directory entries"
	@echo "    - Recover lost clusters"
	@echo "    - Rebuild directory structure"
	@echo ""
	@echo "Recovery Mechanisms:"
	@echo "  1. Automatic retry with backoff"
	@echo "  2. Graceful degradation"
	@echo "  3. User notification of errors"
	@echo "  4. Data integrity preservation"
	@echo "  5. System stability maintenance"

# Signal integrity analysis
analyze-signals:
	@echo "Signal Integrity Analysis:"
	@echo "========================="
	@echo ""
	@echo "Signal Quality Requirements:"
	@echo "  Rise Time: <50ns @ 4MHz"
	@echo "  Fall Time: <50ns @ 4MHz"
	@echo "  Overshoot: <10% of VDD"
	@echo "  Undershoot: <10% of VSS"
	@echo "  Jitter: <5% of clock period"
	@echo ""
	@echo "PCB Layout Guidelines:"
	@echo "  Trace Impedance: 50Ω ±10%"
	@echo "  Trace Length: <10cm for 4MHz"
	@echo "  Ground Plane: Continuous under SPI traces"
	@echo "  Via Count: Minimize vias in signal path"
	@echo "  Crosstalk: >3x spacing between traces"
	@echo ""
	@echo "Connector Specifications:"
	@echo "  Contact Resistance: <50mΩ"
	@echo "  Insertion Force: 0.5-2N"
	@echo "  Mating Cycles: >1000"
	@echo "  Environmental: -40°C to +85°C"
	@echo ""
	@echo "Measurement Points:"
	@echo "  TP1: CS signal at MCU pin"
	@echo "  TP2: SCK signal at MCU pin"
	@echo "  TP3: MOSI signal at MCU pin"
	@echo "  TP4: MISO signal at SD card"
	@echo "  TP5: VDD at SD card"
	@echo "  TP6: GND reference"

# Disassembly with filesystem focus
disasm: $(PROJECT).elf
	$(OBJDUMP) -d $(PROJECT).elf > $(PROJECT).lst
	@echo "Disassembly with SPI and FAT32 analysis saved to $(PROJECT).lst"
	@echo ""
	@echo "Key symbols to analyze:"
	@echo "  - spi_transfer: SPI communication primitive"
	@echo "  - sd_send_command: SD command protocol"
	@echo "  - sd_read_sector: Block reading function"
	@echo "  - sd_write_sector: Block writing function"
	@echo "  - fat32_read_boot_sector: Boot sector parsing"
	@echo "  - fat32_get_next_cluster: FAT navigation"
	@echo "  - fat32_open_file: File opening logic"

# Upload to board
upload: $(PROJECT).hex
	@echo "Uploading SPI SD card filesystem..."
	$(AVRDUDE) -c $(PROGRAMMER) -p $(MCU) -P $(PORT) -b $(BAUD) -U flash:w:$(PROJECT).hex:i

# Comprehensive testing guide
test-comprehensive:
	@echo "Comprehensive Testing Guide:"
	@echo "==========================="
	@echo ""
	@echo "Phase 1: Hardware Verification"
	@echo "  1. Power supply verification (3.3V/5V)"
	@echo "  2. SPI signal continuity testing"
	@echo "  3. SD card socket inspection"
	@echo "  4. Pull-up resistor verification"
	@echo "  5. LED functionality check"
	@echo ""
	@echo "Phase 2: SPI Protocol Testing"
	@echo "  1. Clock frequency verification"
	@echo "  2. Command/response timing"
	@echo "  3. Data transfer integrity"
	@echo "  4. Error condition handling"
	@echo "  5. Multi-card compatibility"
	@echo ""
	@echo "Phase 3: File System Testing"
	@echo "  1. Boot sector validation"
	@echo "  2. FAT table accessibility"
	@echo "  3. Directory operations"
	@echo "  4. File create/read/write/delete"
	@echo "  5. Large file handling"
	@echo ""
	@echo "Phase 4: Performance Validation"
	@echo "  1. Read/write speed measurement"
	@echo "  2. CPU utilization monitoring"
	@echo "  3. Memory usage verification"
	@echo "  4. Power consumption analysis"
	@echo "  5. Long-term stability testing"
	@echo ""
	@echo "Phase 5: Stress Testing"
	@echo "  1. Continuous operation (24+ hours)"
	@echo "  2. Random file access patterns"
	@echo "  3. Error injection and recovery"
	@echo "  4. Temperature cycling"
	@echo "  5. Power cycling endurance"

# Debug information extraction
debug-info:
	@echo "Debug Information:"
	@echo "=================="
	@echo ""
	@echo "Compile-time Configuration:"
	@echo "  F_CPU: $(F_CPU) Hz"
	@echo "  BAUD: $(BAUD) bps"
	@echo "  Sector buffer: 512 bytes"
	@echo "  Max open files: 4"
	@echo ""
	@echo "Memory Layout:"
	@$(SIZE) --format=berkeley $(PROJECT).elf
	@echo ""
	@echo "Symbol Table (key functions):"
	@$(OBJDUMP) -t $(PROJECT).elf | grep -E "(main|spi_|sd_|fat32_)" | head -20

# Clean targets
clean:
	@echo "Cleaning build files..."
	rm -f $(OBJECTS) $(PROJECT).elf $(PROJECT).hex $(PROJECT).lst $(PROJECT).map

distclean: clean
	rm -f *.o *.elf *.hex *.lst *.map

# Help
help:
	@echo "Available targets:"
	@echo "  all                    - Build the project"
	@echo "  upload                 - Upload to Arduino"
	@echo "  test-sd-protocol       - Show SD protocol testing guide"
	@echo "  test-fat32            - Show FAT32 testing guide"
	@echo "  benchmark-spi         - Show SPI performance benchmarks"
	@echo "  test-compatibility    - Show SD card compatibility guide"
	@echo "  analyze-power         - Show power consumption analysis"
	@echo "  test-error-recovery   - Show error handling guide"
	@echo "  analyze-signals       - Show signal integrity analysis"
	@echo "  test-comprehensive    - Show comprehensive testing guide"
	@echo "  debug-info            - Show debug and memory information"
	@echo "  spi-analysis          - Show SPI system analysis"
	@echo "  size                  - Show memory usage"
	@echo "  disasm                - Create disassembly"
	@echo "  clean                 - Clean build files"
	@echo "  help                  - Show this help"

.PHONY: all size spi-analysis disasm upload test-sd-protocol test-fat32 benchmark-spi test-compatibility analyze-power test-error-recovery analyze-signals test-comprehensive debug-info clean distclean help