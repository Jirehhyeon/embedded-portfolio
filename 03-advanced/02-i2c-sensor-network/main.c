/*
 * Advanced I2C Sensor Network System
 * 
 * Hardware: ATmega328P
 * Features: Multi-sensor I2C network with sensor fusion and data logging
 * 
 * Connected Sensors:
 * - BMP280: Pressure/Temperature sensor (0x76)
 * - MPU6050: 6-axis IMU sensor (0x68)  
 * - DS3231: RTC with temperature (0x68 - different registers)
 * - 24LC256: EEPROM for data logging (0x50)
 * - PCF8574: I/O expander for LEDs (0x20)
 * - SSD1306: OLED display (0x3C)
 * - LCD: 16x2 with I2C backpack (0x27)
 * 
 * This program demonstrates:
 * - Complete I2C protocol implementation
 * - Multi-device address management
 * - Sensor data fusion with Kalman filtering
 * - Real-time data logging to EEPROM
 * - Network health monitoring and recovery
 * - Intelligent sensor calibration
 * 
 * Author: [Your Name]
 * Date: 2024-01-20
 */

#include <avr/io.h>
#include <avr/interrupt.h>
#include <avr/eeprom.h>
#include <util/delay.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// I2C Configuration
#define I2C_FREQ        100000UL    // 100kHz Standard Mode
#define TWBR_VALUE      ((F_CPU / I2C_FREQ) - 16) / 2

// Sensor I2C Addresses
#define BMP280_ADDR     0x76
#define MPU6050_ADDR    0x68
#define DS3231_ADDR     0x68    // Same as MPU6050 but different registers
#define EEPROM_ADDR     0x50
#define PCF8574_ADDR    0x20
#define OLED_ADDR       0x3C
#define LCD_ADDR        0x27

// Hardware pin definitions
#define INT_PIN         PD2     // MPU6050 interrupt
#define RTC_SQW_PIN     PD3     // DS3231 square wave
#define IO_INT_PIN      PD4     // PCF8574 interrupt

#define STATUS_LED      PB0     // Network status
#define ERROR_LED       PB1     // Error indicator
#define DATA_LED        PB2     // Data activity
#define FUSION_LED      PB3     // Sensor fusion active

// BMP280 Registers
#define BMP280_CHIP_ID      0xD0
#define BMP280_RESET        0xE0
#define BMP280_STATUS       0xF3
#define BMP280_CTRL_MEAS    0xF4
#define BMP280_CONFIG       0xF5
#define BMP280_PRESS_MSB    0xF7
#define BMP280_PRESS_LSB    0xF8
#define BMP280_PRESS_XLSB   0xF9
#define BMP280_TEMP_MSB     0xFA
#define BMP280_TEMP_LSB     0xFB
#define BMP280_TEMP_XLSB    0xFC

// MPU6050 Registers
#define MPU6050_WHO_AM_I    0x75
#define MPU6050_PWR_MGMT_1  0x6B
#define MPU6050_PWR_MGMT_2  0x6C
#define MPU6050_CONFIG      0x1A
#define MPU6050_GYRO_CONFIG 0x1B
#define MPU6050_ACCEL_CONFIG 0x1C
#define MPU6050_ACCEL_XOUT_H 0x3B
#define MPU6050_TEMP_OUT_H  0x41
#define MPU6050_GYRO_XOUT_H 0x43

// DS3231 Registers
#define DS3231_SECONDS      0x00
#define DS3231_MINUTES      0x01
#define DS3231_HOURS        0x02
#define DS3231_DAY          0x03
#define DS3231_DATE         0x04
#define DS3231_MONTH        0x05
#define DS3231_YEAR         0x06
#define DS3231_CONTROL      0x0E
#define DS3231_STATUS       0x0F
#define DS3231_TEMP_MSB     0x11
#define DS3231_TEMP_LSB     0x12

// Data Structures
typedef struct {
    float temperature;      // °C
    float pressure;         // hPa  
    float altitude;         // meters
    uint32_t timestamp;
    bool valid;
    uint8_t error_count;
} bmp280_data_t;

typedef struct {
    int16_t accel_x, accel_y, accel_z;  // Raw accelerometer data
    int16_t gyro_x, gyro_y, gyro_z;     // Raw gyroscope data
    float temperature;                   // °C
    float accel_g[3];                   // Acceleration in g
    float gyro_dps[3];                  // Angular velocity in °/s
    uint32_t timestamp;
    bool valid;
    uint8_t error_count;
} mpu6050_data_t;

typedef struct {
    uint8_t year, month, day;
    uint8_t hour, minute, second;
    uint8_t day_of_week;
    float temperature;                   // °C
    uint32_t unix_timestamp;
    bool valid;
    uint8_t error_count;
} ds3231_data_t;

typedef struct {
    float q;        // Process noise covariance
    float r;        // Measurement noise covariance
    float x;        // Estimated value
    float p;        // Error covariance
    float k;        // Kalman gain
} kalman_filter_t;

typedef struct {
    float fused_temperature;    // Multi-sensor temperature fusion
    float fused_altitude;       // Barometric + accelerometer altitude
    float orientation[3];       // Roll, Pitch, Yaw (degrees)
    float angular_velocity[3];  // Angular rates (°/s)
    float linear_acceleration[3]; // Linear acceleration (m/s²)
    bool fusion_valid;
    uint32_t last_update;
} sensor_fusion_t;

typedef struct {
    uint32_t timestamp;
    float temperature;
    float pressure;
    float altitude;
    int16_t accel[3];
    int16_t gyro[3];
    float orientation[3];
    uint8_t sensor_status;
    uint8_t checksum;
} __attribute__((packed)) data_log_entry_t;

typedef struct {
    bmp280_data_t bmp280;
    mpu6050_data_t mpu6050;
    ds3231_data_t rtc;
    sensor_fusion_t fusion;
    uint32_t system_time_ms;
    uint8_t active_sensors;
    uint8_t network_errors;
    uint8_t i2c_bus_errors;
    bool network_healthy;
    bool logging_enabled;
} sensor_network_t;

// Global Variables
sensor_network_t sensor_network = {0};
kalman_filter_t temp_filter = {0.01f, 0.1f, 25.0f, 1.0f, 0.0f};
kalman_filter_t altitude_filter = {0.1f, 1.0f, 0.0f, 1.0f, 0.0f};
kalman_filter_t orientation_filters[3] = {
    {0.01f, 0.1f, 0.0f, 1.0f, 0.0f},  // Roll
    {0.01f, 0.1f, 0.0f, 1.0f, 0.0f},  // Pitch  
    {0.05f, 0.5f, 0.0f, 1.0f, 0.0f}   // Yaw
};

// EEPROM logging
#define EEPROM_LOG_START    0x0000
#define EEPROM_LOG_SIZE     32768
#define LOG_ENTRY_SIZE      sizeof(data_log_entry_t)
#define MAX_LOG_ENTRIES     (EEPROM_LOG_SIZE / LOG_ENTRY_SIZE)

uint16_t current_log_index = 0;
bool eeprom_full = false;

// BMP280 calibration data
typedef struct {
    uint16_t dig_T1;
    int16_t dig_T2, dig_T3;
    uint16_t dig_P1;
    int16_t dig_P2, dig_P3, dig_P4, dig_P5, dig_P6, dig_P7, dig_P8, dig_P9;
} bmp280_calib_t;

bmp280_calib_t bmp280_calib = {0};

// ==========================================
// I2C Low-Level Functions
// ==========================================

void i2c_init(void) {
    TWSR = 0x00;  // No prescaler
    TWBR = TWBR_VALUE;
    TWCR = (1 << TWEN);  // Enable TWI
}

uint8_t i2c_start(uint8_t address) {
    TWCR = (1 << TWINT) | (1 << TWSTA) | (1 << TWEN);
    while(!(TWCR & (1 << TWINT)));
    
    if((TWSR & 0xF8) != 0x08) {
        sensor_network.i2c_bus_errors++;
        return 1;  // Start condition failed
    }
    
    TWDR = address;
    TWCR = (1 << TWINT) | (1 << TWEN);
    while(!(TWCR & (1 << TWINT)));
    
    uint8_t status = TWSR & 0xF8;
    if(status != 0x18 && status != 0x40) {
        sensor_network.i2c_bus_errors++;
        return 1;  // Address not acknowledged
    }
    
    return 0;  // Success
}

void i2c_stop(void) {
    TWCR = (1 << TWINT) | (1 << TWSTO) | (1 << TWEN);
    while(TWCR & (1 << TWSTO));
}

uint8_t i2c_write(uint8_t data) {
    TWDR = data;
    TWCR = (1 << TWINT) | (1 << TWEN);
    while(!(TWCR & (1 << TWINT)));
    
    if((TWSR & 0xF8) != 0x28) {
        sensor_network.i2c_bus_errors++;
        return 1;  // Data not acknowledged
    }
    return 0;
}

uint8_t i2c_read_ack(void) {
    TWCR = (1 << TWINT) | (1 << TWEN) | (1 << TWEA);
    while(!(TWCR & (1 << TWINT)));
    return TWDR;
}

uint8_t i2c_read_nack(void) {
    TWCR = (1 << TWINT) | (1 << TWEN);
    while(!(TWCR & (1 << TWINT)));
    return TWDR;
}

uint8_t i2c_read_register(uint8_t device_addr, uint8_t reg_addr) {
    uint8_t data = 0;
    
    if(i2c_start(device_addr << 1) == 0) {
        if(i2c_write(reg_addr) == 0) {
            if(i2c_start((device_addr << 1) | 1) == 0) {
                data = i2c_read_nack();
            }
        }
    }
    i2c_stop();
    
    return data;
}

bool i2c_write_register(uint8_t device_addr, uint8_t reg_addr, uint8_t data) {
    bool success = false;
    
    if(i2c_start(device_addr << 1) == 0) {
        if(i2c_write(reg_addr) == 0) {
            if(i2c_write(data) == 0) {
                success = true;
            }
        }
    }
    i2c_stop();
    
    return success;
}

bool i2c_read_multiple(uint8_t device_addr, uint8_t reg_addr, uint8_t *buffer, uint8_t count) {
    bool success = false;
    
    if(i2c_start(device_addr << 1) == 0) {
        if(i2c_write(reg_addr) == 0) {
            if(i2c_start((device_addr << 1) | 1) == 0) {
                for(uint8_t i = 0; i < count - 1; i++) {
                    buffer[i] = i2c_read_ack();
                }
                buffer[count - 1] = i2c_read_nack();
                success = true;
            }
        }
    }
    i2c_stop();
    
    return success;
}

// ==========================================
// BMP280 Sensor Functions
// ==========================================

bool bmp280_read_calibration(void) {
    uint8_t calib_data[24];
    
    if(!i2c_read_multiple(BMP280_ADDR, 0x88, calib_data, 24)) {
        return false;
    }
    
    bmp280_calib.dig_T1 = (calib_data[1] << 8) | calib_data[0];
    bmp280_calib.dig_T2 = (int16_t)((calib_data[3] << 8) | calib_data[2]);
    bmp280_calib.dig_T3 = (int16_t)((calib_data[5] << 8) | calib_data[4]);
    
    bmp280_calib.dig_P1 = (calib_data[7] << 8) | calib_data[6];
    bmp280_calib.dig_P2 = (int16_t)((calib_data[9] << 8) | calib_data[8]);
    bmp280_calib.dig_P3 = (int16_t)((calib_data[11] << 8) | calib_data[10]);
    bmp280_calib.dig_P4 = (int16_t)((calib_data[13] << 8) | calib_data[12]);
    bmp280_calib.dig_P5 = (int16_t)((calib_data[15] << 8) | calib_data[14]);
    bmp280_calib.dig_P6 = (int16_t)((calib_data[17] << 8) | calib_data[16]);
    bmp280_calib.dig_P7 = (int16_t)((calib_data[19] << 8) | calib_data[18]);
    bmp280_calib.dig_P8 = (int16_t)((calib_data[21] << 8) | calib_data[20]);
    bmp280_calib.dig_P9 = (int16_t)((calib_data[23] << 8) | calib_data[22]);
    
    return true;
}

bool bmp280_init(void) {
    // Check chip ID
    uint8_t chip_id = i2c_read_register(BMP280_ADDR, BMP280_CHIP_ID);
    if(chip_id != 0x58) {  // BMP280 chip ID
        return false;
    }
    
    // Read calibration data
    if(!bmp280_read_calibration()) {
        return false;
    }
    
    // Configure sensor
    if(!i2c_write_register(BMP280_ADDR, BMP280_CONFIG, 0xA0)) {  // Standby 1000ms, filter off
        return false;
    }
    
    if(!i2c_write_register(BMP280_ADDR, BMP280_CTRL_MEAS, 0x27)) {  // Temp x1, Press x1, Normal mode
        return false;
    }
    
    return true;
}

int32_t bmp280_compensate_temperature(int32_t adc_T) {
    int32_t var1, var2, T;
    
    var1 = ((((adc_T >> 3) - ((int32_t)bmp280_calib.dig_T1 << 1))) * 
            ((int32_t)bmp280_calib.dig_T2)) >> 11;
    var2 = (((((adc_T >> 4) - ((int32_t)bmp280_calib.dig_T1)) * 
              ((adc_T >> 4) - ((int32_t)bmp280_calib.dig_T1))) >> 12) * 
            ((int32_t)bmp280_calib.dig_T3)) >> 14;
    
    T = var1 + var2;
    return (T * 5 + 128) >> 8;
}

uint32_t bmp280_compensate_pressure(int32_t adc_P, int32_t t_fine) {
    int64_t var1, var2, p;
    
    var1 = ((int64_t)t_fine) - 128000;
    var2 = var1 * var1 * (int64_t)bmp280_calib.dig_P6;
    var2 = var2 + ((var1 * (int64_t)bmp280_calib.dig_P5) << 17);
    var2 = var2 + (((int64_t)bmp280_calib.dig_P4) << 35);
    var1 = ((var1 * var1 * (int64_t)bmp280_calib.dig_P3) >> 8) + 
           ((var1 * (int64_t)bmp280_calib.dig_P2) << 12);
    var1 = (((((int64_t)1) << 47) + var1)) * ((int64_t)bmp280_calib.dig_P1) >> 33;
    
    if(var1 == 0) {
        return 0; // avoid exception caused by division by zero
    }
    
    p = 1048576 - adc_P;
    p = (((p << 31) - var2) * 3125) / var1;
    var1 = (((int64_t)bmp280_calib.dig_P9) * (p >> 13) * (p >> 13)) >> 25;
    var2 = (((int64_t)bmp280_calib.dig_P8) * p) >> 19;
    p = ((p + var1 + var2) >> 8) + (((int64_t)bmp280_calib.dig_P7) << 4);
    
    return (uint32_t)p;
}

bool bmp280_read_data(bmp280_data_t *data) {
    uint8_t raw_data[6];
    
    if(!i2c_read_multiple(BMP280_ADDR, BMP280_PRESS_MSB, raw_data, 6)) {
        data->error_count++;
        return false;
    }
    
    int32_t adc_P = ((uint32_t)raw_data[0] << 12) | 
                    ((uint32_t)raw_data[1] << 4) | 
                    ((uint32_t)raw_data[2] >> 4);
    
    int32_t adc_T = ((uint32_t)raw_data[3] << 12) | 
                    ((uint32_t)raw_data[4] << 4) | 
                    ((uint32_t)raw_data[5] >> 4);
    
    // Compensate temperature
    int32_t t_fine = ((((adc_T >> 3) - ((int32_t)bmp280_calib.dig_T1 << 1))) * 
                      ((int32_t)bmp280_calib.dig_T2)) >> 11;
    t_fine += (((((adc_T >> 4) - ((int32_t)bmp280_calib.dig_T1)) * 
                 ((adc_T >> 4) - ((int32_t)bmp280_calib.dig_T1))) >> 12) * 
               ((int32_t)bmp280_calib.dig_T3)) >> 14;
    
    data->temperature = (float)bmp280_compensate_temperature(adc_T) / 100.0f;
    
    // Compensate pressure
    uint32_t pressure_raw = bmp280_compensate_pressure(adc_P, t_fine);
    data->pressure = (float)pressure_raw / 25600.0f;  // Convert to hPa
    
    // Calculate altitude (international barometric formula)
    data->altitude = 44330.0f * (1.0f - powf(data->pressure / 1013.25f, 0.1903f));
    
    data->timestamp = sensor_network.system_time_ms;
    data->valid = true;
    
    return true;
}

// ==========================================
// MPU6050 Sensor Functions
// ==========================================

bool mpu6050_init(void) {
    // Check WHO_AM_I
    uint8_t who_am_i = i2c_read_register(MPU6050_ADDR, MPU6050_WHO_AM_I);
    if(who_am_i != 0x68) {
        return false;
    }
    
    // Reset device
    if(!i2c_write_register(MPU6050_ADDR, MPU6050_PWR_MGMT_1, 0x80)) {
        return false;
    }
    _delay_ms(100);
    
    // Wake up device
    if(!i2c_write_register(MPU6050_ADDR, MPU6050_PWR_MGMT_1, 0x00)) {
        return false;
    }
    
    // Configure gyroscope (±250°/s)
    if(!i2c_write_register(MPU6050_ADDR, MPU6050_GYRO_CONFIG, 0x00)) {
        return false;
    }
    
    // Configure accelerometer (±2g)
    if(!i2c_write_register(MPU6050_ADDR, MPU6050_ACCEL_CONFIG, 0x00)) {
        return false;
    }
    
    // Set sample rate divider (1kHz / (1 + 7) = 125Hz)
    if(!i2c_write_register(MPU6050_ADDR, 0x19, 0x07)) {
        return false;
    }
    
    // Configure DLPF (Digital Low Pass Filter)
    if(!i2c_write_register(MPU6050_ADDR, MPU6050_CONFIG, 0x06)) {
        return false;
    }
    
    return true;
}

bool mpu6050_read_data(mpu6050_data_t *data) {
    uint8_t raw_data[14];
    
    if(!i2c_read_multiple(MPU6050_ADDR, MPU6050_ACCEL_XOUT_H, raw_data, 14)) {
        data->error_count++;
        return false;
    }
    
    // Parse raw data
    data->accel_x = (int16_t)((raw_data[0] << 8) | raw_data[1]);
    data->accel_y = (int16_t)((raw_data[2] << 8) | raw_data[3]);
    data->accel_z = (int16_t)((raw_data[4] << 8) | raw_data[5]);
    
    int16_t temp_raw = (int16_t)((raw_data[6] << 8) | raw_data[7]);
    data->temperature = (float)temp_raw / 340.0f + 36.53f;
    
    data->gyro_x = (int16_t)((raw_data[8] << 8) | raw_data[9]);
    data->gyro_y = (int16_t)((raw_data[10] << 8) | raw_data[11]);
    data->gyro_z = (int16_t)((raw_data[12] << 8) | raw_data[13]);
    
    // Convert to physical units
    data->accel_g[0] = (float)data->accel_x / 16384.0f;  // ±2g scale
    data->accel_g[1] = (float)data->accel_y / 16384.0f;
    data->accel_g[2] = (float)data->accel_z / 16384.0f;
    
    data->gyro_dps[0] = (float)data->gyro_x / 131.0f;   // ±250°/s scale
    data->gyro_dps[1] = (float)data->gyro_y / 131.0f;
    data->gyro_dps[2] = (float)data->gyro_z / 131.0f;
    
    data->timestamp = sensor_network.system_time_ms;
    data->valid = true;
    
    return true;
}

// ==========================================
// DS3231 RTC Functions
// ==========================================

uint8_t bcd_to_decimal(uint8_t bcd) {
    return ((bcd >> 4) * 10) + (bcd & 0x0F);
}

uint8_t decimal_to_bcd(uint8_t decimal) {
    return ((decimal / 10) << 4) | (decimal % 10);
}

bool ds3231_init(void) {
    // Configure control register
    if(!i2c_write_register(DS3231_ADDR, DS3231_CONTROL, 0x04)) {  // 1Hz square wave
        return false;
    }
    
    // Clear status register
    if(!i2c_write_register(DS3231_ADDR, DS3231_STATUS, 0x00)) {
        return false;
    }
    
    return true;
}

bool ds3231_read_time(ds3231_data_t *data) {
    uint8_t time_data[7];
    
    if(!i2c_read_multiple(DS3231_ADDR, DS3231_SECONDS, time_data, 7)) {
        data->error_count++;
        return false;
    }
    
    data->second = bcd_to_decimal(time_data[0] & 0x7F);
    data->minute = bcd_to_decimal(time_data[1] & 0x7F);
    data->hour = bcd_to_decimal(time_data[2] & 0x3F);
    data->day_of_week = bcd_to_decimal(time_data[3] & 0x07);
    data->day = bcd_to_decimal(time_data[4] & 0x3F);
    data->month = bcd_to_decimal(time_data[5] & 0x1F);
    data->year = bcd_to_decimal(time_data[6]);
    
    // Read temperature
    uint8_t temp_data[2];
    if(i2c_read_multiple(DS3231_ADDR, DS3231_TEMP_MSB, temp_data, 2)) {
        data->temperature = (float)temp_data[0] + ((temp_data[1] >> 6) * 0.25f);
    }
    
    // Calculate Unix timestamp (simplified)
    data->unix_timestamp = (uint32_t)data->year * 365 * 24 * 3600 +
                          (uint32_t)data->month * 30 * 24 * 3600 +
                          (uint32_t)data->day * 24 * 3600 +
                          (uint32_t)data->hour * 3600 +
                          (uint32_t)data->minute * 60 +
                          (uint32_t)data->second;
    
    data->valid = true;
    return true;
}

// ==========================================
// PCF8574 I/O Expander Functions
// ==========================================

bool pcf8574_write(uint8_t data) {
    bool success = false;
    
    if(i2c_start(PCF8574_ADDR << 1) == 0) {
        if(i2c_write(data) == 0) {
            success = true;
        }
    }
    i2c_stop();
    
    return success;
}

uint8_t pcf8574_read(void) {
    uint8_t data = 0xFF;
    
    if(i2c_start((PCF8574_ADDR << 1) | 1) == 0) {
        data = i2c_read_nack();
    }
    i2c_stop();
    
    return data;
}

void update_status_leds(void) {
    uint8_t led_pattern = 0xFF;  // All LEDs off (PCF8574 is active low)
    
    // Sensor status LEDs
    if(sensor_network.bmp280.valid) led_pattern &= ~0x01;
    if(sensor_network.mpu6050.valid) led_pattern &= ~0x02;
    if(sensor_network.rtc.valid) led_pattern &= ~0x04;
    if(sensor_network.fusion.fusion_valid) led_pattern &= ~0x08;
    
    // Network health
    if(sensor_network.network_healthy) led_pattern &= ~0x10;
    if(sensor_network.logging_enabled) led_pattern &= ~0x20;
    
    // Error indication (blinking)
    static uint8_t blink_counter = 0;
    if(sensor_network.network_errors > 0) {
        if((++blink_counter & 0x0F) < 8) {
            led_pattern &= ~0x40;  // Error LED on
        }
    }
    
    pcf8574_write(led_pattern);
}

// ==========================================
// Kalman Filter Implementation
// ==========================================

float kalman_update(kalman_filter_t *filter, float measurement) {
    // Prediction step
    filter->p = filter->p + filter->q;
    
    // Update step
    filter->k = filter->p / (filter->p + filter->r);
    filter->x = filter->x + filter->k * (measurement - filter->x);
    filter->p = (1.0f - filter->k) * filter->p;
    
    return filter->x;
}

// ==========================================
// Sensor Fusion Functions
// ==========================================

void update_sensor_fusion(void) {
    sensor_fusion_t *fusion = &sensor_network.fusion;
    
    if(!sensor_network.bmp280.valid || !sensor_network.mpu6050.valid) {
        fusion->fusion_valid = false;
        return;
    }
    
    // Temperature fusion
    float temp_sum = 0.0f;
    uint8_t temp_count = 0;
    
    if(sensor_network.bmp280.valid) {
        temp_sum += sensor_network.bmp280.temperature;
        temp_count++;
    }
    if(sensor_network.mpu6050.valid) {
        temp_sum += sensor_network.mpu6050.temperature;
        temp_count++;
    }
    if(sensor_network.rtc.valid) {
        temp_sum += sensor_network.rtc.temperature;
        temp_count++;
    }
    
    if(temp_count > 0) {
        fusion->fused_temperature = kalman_update(&temp_filter, temp_sum / temp_count);
    }
    
    // Altitude fusion
    if(sensor_network.bmp280.valid) {
        fusion->fused_altitude = kalman_update(&altitude_filter, 
                                              sensor_network.bmp280.altitude);
    }
    
    // Orientation calculation (simplified complementary filter)
    if(sensor_network.mpu6050.valid) {
        mpu6050_data_t *imu = &sensor_network.mpu6050;
        
        // Calculate roll and pitch from accelerometer
        float accel_roll = atan2f(imu->accel_g[1], imu->accel_g[2]) * 180.0f / M_PI;
        float accel_pitch = atan2f(-imu->accel_g[0], 
                                  sqrtf(imu->accel_g[1]*imu->accel_g[1] + 
                                       imu->accel_g[2]*imu->accel_g[2])) * 180.0f / M_PI;
        
        // Apply Kalman filtering
        fusion->orientation[0] = kalman_update(&orientation_filters[0], accel_roll);
        fusion->orientation[1] = kalman_update(&orientation_filters[1], accel_pitch);
        
        // Yaw integration (gyroscope only, will drift)
        static float yaw_integrator = 0.0f;
        float dt = 0.008f;  // 125Hz sample rate
        yaw_integrator += imu->gyro_dps[2] * dt;
        fusion->orientation[2] = kalman_update(&orientation_filters[2], yaw_integrator);
        
        // Angular velocity (filtered)
        fusion->angular_velocity[0] = imu->gyro_dps[0];
        fusion->angular_velocity[1] = imu->gyro_dps[1];
        fusion->angular_velocity[2] = imu->gyro_dps[2];
        
        // Linear acceleration (gravity compensated)
        fusion->linear_acceleration[0] = (imu->accel_g[0] + sinf(fusion->orientation[1] * M_PI / 180.0f)) * 9.81f;
        fusion->linear_acceleration[1] = (imu->accel_g[1] - sinf(fusion->orientation[0] * M_PI / 180.0f) * 
                                         cosf(fusion->orientation[1] * M_PI / 180.0f)) * 9.81f;
        fusion->linear_acceleration[2] = (imu->accel_g[2] - cosf(fusion->orientation[0] * M_PI / 180.0f) * 
                                         cosf(fusion->orientation[1] * M_PI / 180.0f)) * 9.81f;
    }
    
    fusion->last_update = sensor_network.system_time_ms;
    fusion->fusion_valid = true;
}

// ==========================================
// Data Logging Functions
// ==========================================

uint8_t calculate_checksum(const data_log_entry_t *entry) {
    uint8_t checksum = 0;
    const uint8_t *data = (const uint8_t*)entry;
    
    for(uint16_t i = 0; i < sizeof(data_log_entry_t) - 1; i++) {
        checksum ^= data[i];
    }
    
    return checksum;
}

bool write_eeprom_page(uint16_t address, const uint8_t *data, uint8_t length) {
    if(i2c_start(EEPROM_ADDR << 1)) return false;
    
    // Write address (16-bit)
    if(i2c_write((uint8_t)(address >> 8))) { i2c_stop(); return false; }
    if(i2c_write((uint8_t)(address & 0xFF))) { i2c_stop(); return false; }
    
    // Write data
    for(uint8_t i = 0; i < length; i++) {
        if(i2c_write(data[i])) { i2c_stop(); return false; }
    }
    
    i2c_stop();
    _delay_ms(5);  // EEPROM write cycle time
    
    return true;
}

bool log_sensor_data(void) {
    if(!sensor_network.logging_enabled || eeprom_full) {
        return false;
    }
    
    data_log_entry_t entry = {0};
    
    entry.timestamp = sensor_network.system_time_ms;
    
    if(sensor_network.bmp280.valid) {
        entry.temperature = sensor_network.bmp280.temperature;
        entry.pressure = sensor_network.bmp280.pressure;
        entry.altitude = sensor_network.bmp280.altitude;
    }
    
    if(sensor_network.mpu6050.valid) {
        entry.accel[0] = sensor_network.mpu6050.accel_x;
        entry.accel[1] = sensor_network.mpu6050.accel_y;
        entry.accel[2] = sensor_network.mpu6050.accel_z;
        entry.gyro[0] = sensor_network.mpu6050.gyro_x;
        entry.gyro[1] = sensor_network.mpu6050.gyro_y;
        entry.gyro[2] = sensor_network.mpu6050.gyro_z;
    }
    
    if(sensor_network.fusion.fusion_valid) {
        entry.orientation[0] = sensor_network.fusion.orientation[0];
        entry.orientation[1] = sensor_network.fusion.orientation[1];
        entry.orientation[2] = sensor_network.fusion.orientation[2];
    }
    
    entry.sensor_status = sensor_network.active_sensors;
    entry.checksum = calculate_checksum(&entry);
    
    uint16_t eeprom_address = EEPROM_LOG_START + (current_log_index * LOG_ENTRY_SIZE);
    
    if(current_log_index >= MAX_LOG_ENTRIES) {
        eeprom_full = true;
        return false;
    }
    
    bool success = write_eeprom_page(eeprom_address, (const uint8_t*)&entry, LOG_ENTRY_SIZE);
    if(success) {
        current_log_index++;
        PORTB ^= (1 << DATA_LED);  // Toggle data LED
    }
    
    return success;
}

// ==========================================
// Network Management Functions
// ==========================================

void scan_i2c_network(void) {
    uint8_t found_devices = 0;
    
    // Test each potential device
    if(i2c_start(BMP280_ADDR << 1) == 0) found_devices |= 0x01;
    i2c_stop();
    
    if(i2c_start(MPU6050_ADDR << 1) == 0) found_devices |= 0x02;
    i2c_stop();
    
    if(i2c_start(DS3231_ADDR << 1) == 0) found_devices |= 0x04;
    i2c_stop();
    
    if(i2c_start(EEPROM_ADDR << 1) == 0) found_devices |= 0x08;
    i2c_stop();
    
    if(i2c_start(PCF8574_ADDR << 1) == 0) found_devices |= 0x10;
    i2c_stop();
    
    sensor_network.active_sensors = found_devices;
    sensor_network.network_healthy = (found_devices >= 0x07);  // At least 3 sensors
}

void update_sensor_network(void) {
    static uint32_t last_update = 0;
    static uint32_t log_timer = 0;
    
    if(sensor_network.system_time_ms - last_update >= 8) {  // 125Hz update rate
        last_update = sensor_network.system_time_ms;
        
        // Update BMP280
        if(!bmp280_read_data(&sensor_network.bmp280)) {
            sensor_network.network_errors++;
        }
        
        // Update MPU6050
        if(!mpu6050_read_data(&sensor_network.mpu6050)) {
            sensor_network.network_errors++;
        }
        
        // Update sensor fusion
        update_sensor_fusion();
    }
    
    // Slower updates for RTC and logging
    if(sensor_network.system_time_ms - log_timer >= 1000) {  // 1Hz
        log_timer = sensor_network.system_time_ms;
        
        // Update RTC
        if(!ds3231_read_time(&sensor_network.rtc)) {
            sensor_network.network_errors++;
        }
        
        // Log data every minute
        static uint8_t log_counter = 0;
        if(++log_counter >= 60) {
            log_counter = 0;
            if(!log_sensor_data()) {
                sensor_network.network_errors++;
            }
        }
        
        // Update status LEDs
        update_status_leds();
    }
}

void diagnose_and_recover_network(void) {
    static uint32_t last_diagnosis = 0;
    
    if(sensor_network.system_time_ms - last_diagnosis >= 10000) {  // Every 10 seconds
        last_diagnosis = sensor_network.system_time_ms;
        
        // Re-scan network
        scan_i2c_network();
        
        // Attempt recovery if network is unhealthy
        if(!sensor_network.network_healthy || sensor_network.i2c_bus_errors > 100) {
            // Reset I2C bus
            i2c_init();
            _delay_ms(100);
            
            // Re-initialize sensors
            if(sensor_network.active_sensors & 0x01) bmp280_init();
            if(sensor_network.active_sensors & 0x02) mpu6050_init();
            if(sensor_network.active_sensors & 0x04) ds3231_init();
            
            // Reset error counters
            sensor_network.i2c_bus_errors = 0;
            sensor_network.network_errors = 0;
            
            PORTB ^= (1 << ERROR_LED);  // Flash error LED
        }
    }
}

// ==========================================
// System Timer and Main Functions
// ==========================================

ISR(TIMER1_COMPA_vect) {
    sensor_network.system_time_ms++;
    
    // Toggle status LED every second
    static uint16_t status_counter = 0;
    if(++status_counter >= 1000) {
        status_counter = 0;
        PORTB ^= (1 << STATUS_LED);
    }
    
    // Toggle fusion LED when fusion is active
    if(sensor_network.fusion.fusion_valid) {
        static uint8_t fusion_counter = 0;
        if(++fusion_counter >= 100) {
            fusion_counter = 0;
            PORTB ^= (1 << FUSION_LED);
        }
    }
}

void init_system_timer(void) {
    // Timer1: CTC mode, 1ms interval
    TCCR1B = (1 << WGM12) | (1 << CS11) | (1 << CS10);  // CTC, prescaler 64
    OCR1A = 249;  // 1ms @ 16MHz
    TIMSK1 = (1 << OCIE1A);
}

void init_gpio(void) {
    // LED outputs
    DDRB |= (1 << STATUS_LED) | (1 << ERROR_LED) | (1 << DATA_LED) | (1 << FUSION_LED);
    PORTB &= ~((1 << STATUS_LED) | (1 << ERROR_LED) | (1 << DATA_LED) | (1 << FUSION_LED));
    
    // Interrupt inputs
    DDRD &= ~((1 << INT_PIN) | (1 << RTC_SQW_PIN) | (1 << IO_INT_PIN));
    PORTD |= (1 << INT_PIN) | (1 << RTC_SQW_PIN) | (1 << IO_INT_PIN);  // Pull-ups
}

int main(void) {
    // Initialize hardware
    init_gpio();
    i2c_init();
    init_system_timer();
    
    sei();  // Enable global interrupts
    
    _delay_ms(100);  // Allow sensors to stabilize
    
    // Initialize sensors
    bool init_success = true;
    
    PORTB |= (1 << STATUS_LED);  // Show initialization
    
    if(bmp280_init()) {
        sensor_network.active_sensors |= 0x01;
    } else {
        init_success = false;
    }
    
    if(mpu6050_init()) {
        sensor_network.active_sensors |= 0x02;
    } else {
        init_success = false;
    }
    
    if(ds3231_init()) {
        sensor_network.active_sensors |= 0x04;
    } else {
        init_success = false;
    }
    
    // Scan for additional devices
    scan_i2c_network();
    
    // Enable logging if EEPROM is available
    sensor_network.logging_enabled = (sensor_network.active_sensors & 0x08) != 0;
    
    // Initialize status display
    if(init_success && sensor_network.network_healthy) {
        // Success pattern
        for(uint8_t i = 0; i < 3; i++) {
            pcf8574_write(0x00);  // All LEDs on
            _delay_ms(200);
            pcf8574_write(0xFF);  // All LEDs off
            _delay_ms(200);
        }
    } else {
        // Error pattern
        PORTB |= (1 << ERROR_LED);
        for(uint8_t i = 0; i < 5; i++) {
            pcf8574_write(0xAA);  // Alternating pattern
            _delay_ms(300);
            pcf8574_write(0x55);
            _delay_ms(300);
        }
    }
    
    PORTB &= ~(1 << STATUS_LED);
    
    // Main sensor network loop
    while(1) {
        // Core sensor network operations
        update_sensor_network();
        
        // Network health monitoring
        diagnose_and_recover_network();
        
        // Alert conditions
        if(sensor_network.fusion.fusion_valid) {
            // Temperature alerts
            if(sensor_network.fusion.fused_temperature > 40.0f) {
                PORTB |= (1 << ERROR_LED);
            } else if(sensor_network.fusion.fused_temperature < 0.0f) {
                PORTB |= (1 << ERROR_LED);
            } else {
                PORTB &= ~(1 << ERROR_LED);
            }
            
            // Motion alerts (high acceleration)
            float accel_magnitude = sqrtf(
                sensor_network.fusion.linear_acceleration[0] * sensor_network.fusion.linear_acceleration[0] +
                sensor_network.fusion.linear_acceleration[1] * sensor_network.fusion.linear_acceleration[1] +
                sensor_network.fusion.linear_acceleration[2] * sensor_network.fusion.linear_acceleration[2]
            );
            
            if(accel_magnitude > 20.0f) {  // High acceleration detected
                pcf8574_write(0x0F);  // Alert pattern
                _delay_ms(100);
            }
            
            // Orientation alerts (excessive tilt)
            if(fabs(sensor_network.fusion.orientation[0]) > 45.0f || 
               fabs(sensor_network.fusion.orientation[1]) > 45.0f) {
                pcf8574_write(0xF0);  // Tilt alert
                _delay_ms(100);
            }
        }
        
        // Background processing delay
        _delay_ms(1);
    }
    
    return 0;
}